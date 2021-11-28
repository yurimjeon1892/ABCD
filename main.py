import os, sys
import gc

import torch.nn.parallel
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter
import numpy as np

import shutil

import data_loader
import nets

from utils.helper import query_yes_no, load_config
from iterater import iterater
from eval import test, val

import yaml

def main():

    # ensure numba JIT is on
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    # parse arguments
    global args
    with open(sys.argv[1], 'r') as stream:
        args = yaml.safe_load(stream)

    if args['resume'] is not False:
        args = load_config(args['resume'], args)

    cuda = torch.cuda.is_available()
    if cuda :
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("=> using '{}' for computation.".format(device))

    # -------------------- logging args --------------------
    print("=> checking ckpt dir...")
    if args['test'] is False and os.path.exists(args['ckpt_dir']):
        if args['resume'] is False:
            to_continue = query_yes_no(
                '=> Attention!!! ckpt_dir {' + args['ckpt_dir'] + '} already exists!\n' 
                + '=> Whether to continue?',
                default=None)
            if to_continue:
                for root, dirs, files in os.walk(args['ckpt_dir'], topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
            else:
                sys.exit(1)
        elif args['pre_trained'] is not False:
            to_continue = query_yes_no(
                '=> Attention!!! ckpt_dir {' + args['ckpt_dir'] +
                '} already exists! Whether to continue?',
                default=None)
            if to_continue:
                for root, dirs, files in os.walk(args['ckpt_dir'], topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
            else:
                sys.exit(1)
    if not args['test']:
        os.makedirs(args['ckpt_dir'], mode=0o777, exist_ok=True)
        shutil.copyfile(sys.argv[1], os.path.join(args['ckpt_dir'], 'config.yaml'))
        summary = SummaryWriter(args['ckpt_dir'])        

    # -------------------- dataset & loader --------------------
    loader = {}
    if not args['test']:
        train_dataset = data_loader.__dict__[args['dataset']](
            split='train',
            args=args
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=args['workers'],
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )
        loader['train'] = train_loader

        val_dataset = data_loader.__dict__[args['dataset']](
            split='valid',
            args=args
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=args['workers'],
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )
        loader['validate'] = val_loader

    elif args['test'] == 'valid' or args['test'] == 'test':
        test_dataset = data_loader.__dict__[args['dataset']](
            split=args['test'],
            args=args
        )
        # print('val_dataset: ' + str(val_dataset))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=args['workers'],
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )
    # -------------------- create model --------------------
    print("=> creating model and optimizer ... ", end='')
    model = nets.__dict__[args['arch']](args).to(device)

    if not args['test']:
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params,
                                     lr=args['lr'],
                                     weight_decay=args['weight_decay'])
        # define loss functions
        # criterion = nets.__dict__[args['loss']]()
        criterion = nets.Criterion(args)
    print("=> completed.")
    print("=> total parameters: {:.5f}M".format(
        sum(p.numel() for p in model.parameters())/1000000.0))

    model = torch.nn.DataParallel(model)
    # -------------------- resume --------------------
    if args['test']:
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = torch.load(args['resume'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)            
            print("=> completed.")
            print("=> start iter {}, min loss {}"
                  .format(checkpoint['iter'], checkpoint['min_loss']))  
            args['iter'] = checkpoint['iter']
            if args['test'] == 'valid':
                val(test_loader, model, args)
            elif args['test'] == 'test':
                test(test_loader, model, args)
            return
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))
            return
    elif args['resume']:
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = torch.load(args['resume'])
            args['iter'] = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> completed.")
            print("=> start iter {}, min loss {}"
                  .format(checkpoint['iter'], checkpoint['min_loss']))  
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))
            return
    elif args['pre_trained']:
        if os.path.isfile(args['pre_trained']):
            print("=> loading checkpoint '{}'".format(args['pre_trained']))
            checkpoint = torch.load(args['pre_trained'])

            pretrained_dict = checkpoint['state_dict']
            model_dict = model.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(pretrained_dict, strict=False)

            print("=> completed.")
        else:
            print("=> no checkpoint found at '{}'".format(args['pre_trained']))
            return

    # -------------------- main loop --------------------
    it_dict = {}
    if args['resume']:
        it_dict['iter'] = args['iter'] + 1
        it_dict['min_train_loss'] = None
        it_dict['best_train_iter'] = None
        it_dict['min_val_loss'] = checkpoint['min_loss']
        it_dict['best_val_iter'] = args['iter']
    else:
        it_dict['iter'] = 0
        it_dict['min_train_loss'] = None
        it_dict['best_train_iter'] = None
        it_dict['min_val_loss'] = None
        it_dict['best_val_iter'] = None

    while it_dict['iter'] < args['epochs'] * len(loader['train']):
        it_dict = \
            iterater(loader, model, criterion, optimizer, args, summary, it_dict)
        gc.collect()
    return

if __name__ == '__main__':
    main()
