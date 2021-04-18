# EppDevTool
This repository contains train/eval/dev code for development of Eppipolar Occlusion Detection Algorithm

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
For training, you need to donwload [Kitti semidense gt](https://drive.google.com/file/d/1m0dVq5Y88tuRLqXCh3Vl3ED_9wvCfBpb/view?usp=sharing), [Kitti raw lidarscan mapped gt](https://drive.google.com/file/d/1McXOiD9XChVO1ezetv4koF30sWJ88uGd/view?usp=sharing) and [kitti filtered gt](https://drive.google.com/file/d/1w7y9kvGbKHxkWYPS9_e8W37oMCDtriL2/view?usp=sharing)
For evaluation, you need to download an organized version of kitti stereo15 dataset [here](https://drive.google.com/file/d/12GBcFL7PUHijZUj7on4AWFclf6OCAySf/view?usp=sharing).

## Training
You can train using command as follow:
```Shell
python exp_kitti_sync/train_mDnet.py --model_name kittimD --split semidense_eigen_full --data_path [your_kitti_dataset] --gt_path [your_selected_gt] --batch_size 2 --num_epochs 20 --height 320 --width 1024 --val_gt_path [your_selected_gt] --num_layers 50
```

## Evaluation
You can evaluate using command as follow:
```Shell
python exp_kitti_sync/eval_mDnet.py --data_path [your_kitti_dataset] --gt_path [your_kitti_stereo15_organized_path] --height 320 --width 1024 --num_layers 50 --load_weights_folder_depth [your_downloaded_pretrained_weights]
```
The three pretrained models trained under different groundtruth depthmaps can be downloaded here:[Kitti semidense gt](https://drive.google.com/drive/folders/14DDmIoOUSxjQShOwM4whfU9-Xfquz4DR?usp=sharing), [Kitti raw lidarscan mapped gt](https://drive.google.com/drive/folders/1wpeM5kQmCbmNBp6CHkJAgmLhFDYl68cK?usp=sharing) and [kitti filtered gt](https://drive.google.com/drive/folders/1CtVpOE6V3LH4myLaUAJ9sQIrwD98K4Pg?usp=sharing)
