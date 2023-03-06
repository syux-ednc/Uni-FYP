# FYP
Text Recognition, Computer Vision, Deep Learning, CRNN

# Overview

Setting Up of Environment

Preparing Dataset

Training of the Model

(Deployment) Testing of the Model 

Possible Errors Encountered when referencing this code

## Setting Up of Environment

The project was built and ran on Paperspace Gradient, a paid-online platform for machine learning development.

As such, most of the dependencies specified here XXXX are already available in the platform except the following:

1. Install Warp-ctc Loss by XXX, Github Reference: 

```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
cd ..
cd pytorch_binding
python setup.py install
```


2. Install LMDB

```
pip install LMDB
```

3. Install Openpyxl

```
pip install openpyxl
```

## Preparing the Dataset

In this project, we use the [MJSynth Dataset](https://www.robots.ox.ac.uk/~vgg/data/text/)

Afterwich, you need to convert the dataset to .mdb format before it can be trained.

You can use this method to do so :

OR

Get the converted dataset from [here](https://www.dropbox.com/sh/i39abvnefllx2si/AAAbAYRvxzRp3cIE5HzqUw3ra?dl=0)
