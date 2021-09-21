# PolyTrack: Tracking with Bounding Polygons

## Abstract
In this paper, we present a novel method called PolyTrack for fast multi-object tracking and segmentation using bounding polygons. Polytrack detects objects by producing heatmaps of their center keypoint. For each of them, a rough segmentation is done by computing a bounding polygon over each instance instead of the traditional bounding box. Tracking is done by taking two consecutive frames as input and computing a center offset for each object detected in the first frame to predict their location in the second frame. A Kalman filter is also applied to reduce the number of ID switches. Since our target application is automated driving systems, we apply our method on urban environment videos. We train and evaluate PolyTrack on the MOTS and KITTIMOTS dataset.


## Example results

Video examples from the [KITTI MOTS](http://www.cvlibs.net/datasets/kitti/eval_mots.php) test set:

<p align="center"> <img src='readme/demo_video_cars.gif' align="center" height="230px"> </p> 

<p align="center"> <img src='readme/demo_video_mix.gif' align="center" height="230px"> </p> 

## Model

<img src="readme/imgs/architecture.png">

An overview of the PolyTrack architecture. The network takes as input the image at time t, I(t), the image at time t-1, I(t-1), as well as the heatmap at time t-1, H(t-1). Features are produced by the backbone and then used by five different network heads. The center heatmaps head is used for detecting and classifying objects, the polygon head is used for the segmentation part, the depth head is used to produce a relative depth between objects, the tracking head is used to produce an offset between frames at time t-1 and time t and finally the offset head is used for correctly upsampling images.

a) Generated Heatmap       | b) Generated Output
:-------------------------:|:-------------------------:
<img src="readme/imgs/hm.png" height=300> | <img src="readme/imgs/output.png" height=300>

a): The center heatmap produced by the network to detect objects, b): the output of our method: a bounding polygon for each object, a class label, a track id as well as an offset from the previous frame.

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Demo
TODO

## Benchmark Evaluation and Training

After [installation](readme/INSTALL.md), follow the instructions in [DATA.md](readme/DATA.md) to setup the datasets. Then check [GETTING_STARTED.md](readme/GETTING_STARTED.md) to reproduce the results in the paper.
We provide scripts for all the experiments in the [experiments](experiments) folder.

## License

CenterTrack is developed upon [CenterNet](https://github.com/xingyizhou/CenterNet). Both codebases are released under MIT License themselves. Some code of CenterNet are from third-parties with different licenses, please check the CenterNet repo for details. In addition, this repo uses [py-motmetrics](https://github.com/cheind/py-motmetrics) for MOT evaluation and [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) for nuScenes evaluation and preprocessing. See [NOTICE](NOTICE) for detail. Please note the licenses of each dataset. Most of the datasets we used in this project are under non-commercial licenses.

