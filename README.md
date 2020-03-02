# Using panoramic videos for multi-person localization and tracking in a 3D panoramic coordinate

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Dataset Download Link](#dataset-download-link)
* [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Run code](#run-code)
* [Demos](#demos)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## About the Project ([ArXiv](https://arxiv.org/pdf/1911.10535.pdf))
3D panoramic multi-person localization and tracking are prominent in many applications, however, conventional methods using LiDAR equipment could be economically expensive and also computationally inefficient due to the processing of point cloud data. In this work, we propose an effective and efficient approach at a low cost. First, we obtain panoramic videos with four normal cameras. Then, we transform human locations from a 2D panoramic image coordinate to a 3D panoramic camera coordinate using camera geometry and human bio-metric property (i.e., height). Finally, we generate 3D tracklets by associating human appearance and 3D trajectory. We verify the effectiveness of our method on three datasets including a new one built by us, in terms of 3D single-view multi-person localization, 3D single-view multi-person tracking, and 3D panoramic multi-person localization and tracking. 


## Dataset Download Link:
  [Download](https://mega.nz/#!BtYx1ACa!B24sxHQ8hC7t1hVDTJJ4RSBwZDtHiXxoazVpjVSbsro)

## Getting Started
### Installation
The code was tested on Ubuntu 18.04, with Anaconda Python 3.6 and PyTorch v1.1.0.

You may need to install requirements.txt by
```sh
pip3 install requirements.txt
```
### Run code
1. Download data and put them to /data folder
2. Download [model weight](https://drive.google.com/open?id=1AGo6qc1xOiC-DnY0K1Xx824uB9F3Mwzp) and put it to /reid folder
3. Run pano_detector.ipynb to generate and save 2D detection boxes.
4. Run tracking.ipynb to generate and save tracking links.
5. Run generate_video.ipynb to generate visulation videos.

## Demos:
![](pictures/tracking_1.gif)
![](pictures/tracking_2.gif)

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.

## Citation
```
@inproceedings{yang2020mplt,
  title={Using panoramic videos for multi-person localization and tracking in a 3D panoramic coordinate},
  author={Fan Yang, Feiran Li, Yang Wu, Sakriani Sakti, and Satoshi Nakamura},
  booktitle={International Conference on Acoustics, Speech, and Signal Processing},
  year={2020}
}
```

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements (parts of our code are heavily borrowed from)
* [Tiny Person re-id](https://github.com/lulujianjie/person-reid-tiny-baseline)
* [PifPaf](https://github.com/vita-epfl/openpifpaf)
* [DeepSort](https://github.com/vita-epfl/openpifpaf)
* [3D MOT Metric (used for evaluation)](https://github.com/shijieS/mot-metric)
