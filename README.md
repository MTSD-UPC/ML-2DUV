# ML-2DUV
This repository contains the analysis source demo code used in the paper "[Machine learning recognition of protein secondary structures based on two-dimensional spectroscopic descriptors](https://www.pnas.org/doi/full/10.1073/pnas.2202713119)" by Hao Ren et al.

<img src="workflow.jpg"/>

## Download dataset
We provide two kinds of datasets in different sizes.

We recommend downloading datasets of small size which just 1.4 GB: [BaiDu Drive](https://pan.baidu.com/s/1VYTjBFhtAza4Jybajdkhsw?pwd=PNAS)(extract code:PNAS)

You can also download a full dataset in size of 30 GB: [DCAIKU](http://dcaiku.com:13000/)

All Spectral data was simulated using method from our another repository ([2duv_tutorial](https://github.com/MTSD-UPC/2duv_tutorial))
## Data structure

Take 1.4G dataset as an example:

- original
  - original_dataset.npz
    - twoduv
    - la
    - cd
    - labels
  - original_transfer_dataset.npz
    - ...
 - homologous
   - homologous_dataset.npz
     - ...
   - homologous_transfer_dataset.npz
     - ...
 - nonhomologous
   - nonhomologous_dataset.npz
     - ...
   - nonhomologous_transfer_dataset.npz
     - ...


## Prerequisite
```
Python 3.X
Tensorflow>=2.4.0
keras-tuner>=1.0.2
scikit-learn>=0.22
scikit-image>=0.16
numpy
pandas
```
## Download code
```shell
git clone https://github.com/MTSD-UPC/ML-2DUV.git
cd ML-2DUV
```

