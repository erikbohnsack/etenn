# ETENN 

This is the implementation of the End-to-End Neural Network (ETENN)
for the Master Thesis `Multi-Object Tracking using either Deep Learning or PMBM filtering` by Erik Bohnsack and Adam Lilja 
at Chalmers University of Technology.

The implementation is inspired from `Fast and Furious: Real Time End-to-End 3D Detection, Tracking and MotionForecasting with a Single Convolutional Net`
by Luo et al. and `IntentNet: Learning to Predict Intentionfrom Raw Sensor Data` by Casas et al. Since there is no code available 
from these two mentioned papers, this could be used in an attempt to replicate or continue to work on the finidngs of the above. 

On top of using the Fast and Furious network input processing, we borrowed code from [PointPillars:](https://github.com/nutonomy/second.pytorch)`Fast Encoders for Object Detection from Point Clouds` by Lang et al.,
and implemented PointPillars Feature encoding input processing as well.

We have only tested it on Ubuntu 16.04 and Python 3.7, using PyTorch with CUDA. 

## Note

The network never generalized from training due to two reasons:
1. Coordinate Transform mishap
    1. The input frames and the training labels for the future frames are not transformed
    to the same coordinate system using the ego motion of the ego vehicle. This simplifies the task for the network, needing less data.
    It shouldn't be theoretically impossible to learn without this, but we forgot to do it until it was too late. 
1. Data
    1. Fast and Furious uses a private dataset which is 2 order of magnitudes bigger than KITTI. 
    For this task the KITTI tracking dataset (without Data Augmentation which we did not have time to implement) was simply not enough.
    
    
## Requirements

`python 3.7` 


1. torch
1. pyyaml
1. mayavi
1. numba
1. visdom [https://github.com/facebookresearch/visdom]
1. torchviz [https://github.com/szagoruyko/pytorchviz]
1. graphviz: `sudo apt-get install graphviz`

## Train

Check `train.py` or `train_pp.py` for the `PointPillars` version

## Evaluation

Check `eval.py` or `eval_pp.py` for the `PointPillars` version