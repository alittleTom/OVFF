# OVFF
Code of paper "Pose Recognition of 3D Human Shapes via Multi-View CNN with Ordered View Feature Fusion"

Paper Download: https://doi.org/10.3390/electronics9091368

## Prerequisites
* CUDA and CuDNN (changing the code to run on CPU should require few changes)
* Python 3.6
* Tensorflow-gpu 1.9

## Datasets Download and Generate
SH-RE and SH-SY: http://www.cs.cf.ac.uk/shaperetrieval/shrec14/

FAUST: http://faust.is.tue.mpg.de/overview

HPRD: Link: https://pan.baidu.com/s/177328DDAuvUjUV7vPp7tag  Password: gpta

RAD: Link: https://pan.baidu.com/s/1uFDYtmxWq8bc3OtgCiqAuA  Password: 2lam 


Generate dataset views: you can run the script /Others/generateViews_sync.py

Generate rotation dataset (eg: SH-RE-RO): you can run the script /Others/generatePose.py

Generate trainList and testList: you can run the script /Others/generatelist.py

## Download network parameters
Alexnet_imagenet Link: https://pan.baidu.com/s/1CJ_RfJF6e269Je0lKTzkjQ   password: rfqj

Please copy alexnet_imagenet.npy to ./classification and ./retrieval before run code.

## Training and Testing
In each experiment, you can find train.py and test.py used to train and test the network

## Citation
If you use our work, please cite our paper

'''
Wang, H.; He, P.; Li, N.; Cao, J. Pose Recognition of 3D Human Shapes via Multi-View CNN with Ordered View Feature Fusion. Electronics 2020, 9, 1368.
'''

## Contact
If you have any problem about this implementation, please feel free to contact via:

hpcalifornia AT 163 DOT com
