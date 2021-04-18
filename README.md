# EppDevTool
This repository contains train/eval/dev code for development of Eppipolar Occlusion Detection Algorithm

## Dataset Address
Please download an organized version of vrkitti2 from [google drive](https://drive.google.com/file/d/1sRUCkcKPXVhyBWHhe2qVCYBkVi8RsE6I/view?usp=sharing)

## Demos
You can run a demo frame via using command
```Shell
python dev_tool.py --dataset_root=your_vrkitti2root
```

## KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

## Different Kitti gt depthmap
We have four different groundtruth depthmaps, three for comparison and one for evaluation.
For training, you need to donwload Kitti semidense gt [google drive](https://drive.google.com/file/d/1sRUCkcKPXVhyBWHhe2qVCYBkVi8RsE6I/view?usp=sharing), Kitti raw lidarscan mapped gt and kitti filtered gt.
For evaluation, you need to download an organized version of kitti stereo15 dataset here.

## Training
You can train using command as follow:
```Shell
python exp_kitti_sync/train_mDnet.py --model_name kittimD --split semidense_eigen_full --data_path [your_kitti_dataset] --gt_path [your_selected_gt] --batch_size 2 --num_epochs 20 --height 320 --width 1024 --val_gt_path [your_selected_gt] --num_layers 50
```
