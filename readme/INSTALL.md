# Installation


The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6, CUDA 10.0, and [PyTorch]((http://pytorch.org/)) v1.0.
It should be compatible with PyTorch <=1.4 and python >=0.4 (you will need to switch DCNv2 version for PyTorch <1.0).

This installation process will work if you want to use a hourglass backbone. To use DLA, you'll need to follow [CenterTrack](https://github.com/xingyizhou/CenterTrack)'s instruction to install and compile DCNv2.

After installing Anaconda and cloning this repo:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create --name PolyTrack python=3.6
    ~~~
    And activate the environment.
    
    ~~~
    conda activate PolyTrack
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch torchvision -c pytorch
    ~~~
    
2. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
