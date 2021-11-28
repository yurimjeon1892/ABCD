#!/bin/bash

files=(
        data_depth_selection.zip
        data_depth_velodyne.zip
        data_depth_annotated.zip
)

for i in ${files[@]}; do
        fullname=$i
	echo "Downloading: "$fullname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'$fullname
        unzip -o $fullname
        rm $fullname
done
