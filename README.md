# Using panoramic videos for multi-person localization and tracking in a 3D panoramic coordinate

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Run code](#run-code)
* [Demos](#demos)
* [Dataset Download Link](#dataset-download-link)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## About the Project
In this research, we develop a framework for multi-person localization and tracking in a 3D panoramic coordinate by using panoramic RGB videos. It has not been approached before due to difficulties arising from the framework's complexity. To approach it, we simplify the target goal by dividing the entire framework into four basic modules as Pose Detection Module, Geometry Transformation Module, Appearance Re-identification Module and Tracking Module, and then seamlessly incorporating them together. To evaluate our framework and promote related studies, we also propose a panoramic video dataset with persons' 3D trajectories available.

## Getting Started
### Installation
The code was tested on Ubuntu 18.04, with Anaconda Python 3.6 and PyTorch v1.1.0. NVIDIA GPUs are needed for both training and testing. 

You also need to install openpifpaf by
```sh
pip3 install openpifpaf
```
### Run code
  TBA

## Demos:
![](tracking_1.gif)
![](tracking_2.gif)

## Dataset Download Link:
  TBA
 
<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Tiny Person re-id](https://github.com/lulujianjie/person-reid-tiny-baseline)
* [PifPaf](https://github.com/vita-epfl/openpifpaf)
* [DeepSort](https://github.com/vita-epfl/openpifpaf)
* [3D MOT Metric (used for evaluation)](https://github.com/shijieS/mot-metric)
