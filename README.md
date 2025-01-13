# NOTNet
The implementation of the paper "[Brightness-Adjustable Low-Light Image Enhancement via Noise-Only Training Scheme](https://ieeexplore.ieee.org/abstract/document/10211231/)"

NOTNet is inspired by [NoiSER](https://arxiv.org/abs/2211.04700v3), and our implementation is developed from [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE/).
Thanks for your attention to these work!


## Requirements
+ Python 3.8
+ Pytorch 2.1.0
+ torchvision 0.16.0
+ opencv 4.5.3
+ cuda 12.0

All you need is basic environment.

## Train
```
python train_with_noises.py 
```

## Test
```
python test_with_real_images.py --V=0.6
```
Please adjust the V value manually to meet your need.
