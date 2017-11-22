Заменяем nngraph на nn


# How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)

This is the training code for 2D-FAN and 3D-FAN decribed in "How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)" paper. Please visit [our](https://www.adrianbulat.com) webpage or read bellow for instructions on how to run the code.

Pretrained models are available on our page.

**Demo code: <https://www.github.com/1adrianb/2D-and-3D-face-alignment>**

Note: If you are interested in a binarized version, capable of running on devices with limited resources please also check <https://github.com/1adrianb/binary-face-alignment> for a demo.

## Requirments

- Install the latest [Torch7](http://torch.ch/docs/getting-started.html) version (for Windows, please follow the instructions available [here](https://github.com/torch/distro/blob/master/win-files/README.md))

### Packages

- [cutorch](https://github.com/torch/cutorch)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [cudnn](https://github.com/soumith/cudnn.torch)
- [xlua](https://github.com/torch/xlua)
- [image](https://github.com/torch/image)
- [paths](https://github.com/torch/paths)
- [matio](https://github.com/soumith/matio-ffi.torch)

## Setup

1. Clone the github repository and install all the dependencies mentiones above.

```bash

git  clone https://github.com/1adrianb/face-alignment-training
cd face-alignment-training
```

2. Download the 300W-LP dataset from the authors webpage. In order to train on your own data the dataloader.lua file needs to be adapted.

3. Download the 300W-LP annotations converted to t7 format from [here](https://www.adrianbulat.com/downloads/FaceAlignment/landmarks.zip), extract it and move the ```landmarks``` folder to the root of the 300W-LP dataset.

## Usage

In order to run the demo please download the required models available bellow and the associated data.

```bash
th main.lua -data path_to_300W_LP_dataset
```

In order to see all the available options please run:

```bash
th main.lua --help
```

## Citation

```
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
```

## Acknowledgements

This pipeline is build around the ImageNet training code avaialable at <https://github.com/facebook/fb.resnet.torch> and HourGlass(HG) code available at https://github.com/anewell/pose-hg-train
