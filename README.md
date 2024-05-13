# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Setup

1. How to get flowers image folder provided by Udacity:

Image categories are found in cat_to_name.json and flower images can be downloaded in the gziped tar file flower_data.tar.gz.

```bash
make setup
```

This command only automates the download and extraction of flowers data (provided by Udacity) to a flower folder on this project using two commands to do so:
```bash
mkdir -p flowers
curl https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz -o flowers/flower_data.tar.gz
tar -zxf flowers/flower_data.tar.gz -C flowers/
```

If by any chance you need to download and/or extract the files to a different folder, customize the lines above and run it accordingly.

After the setup, you should have the following files:
```
aipnd-project/
│   cat_to_name.json
│   CODEOWNERS
│   Image Classifier Project.html
│   Image Classifier Project.ipynb
│   LICENSE
│   Makefile
│   predict.py
│   train.py
│   utils.py
└───flowers/
│   │   flower_data.tar.gz
│   └───test/
│   └───train/
│   └───valid/
```



## How to train (train.py)

Use the help command to see all supported parameters:
```bash
python ./train.py -h
```

Using **vgg16** (default) and train on **CPU**:
```bash
python ./train.py ./flowers/train/
```

Using **densenet121** to train on **GPU** with **100** node layer:
```bash
python ./train.py ./flowers/train --gpu --arch "densenet121" --hidden_units 100 --epochs 5
```

Additional hidden layers with checkpoint saved to densenet201 folder.
```bash
python ./train.py ./flowers/train --gpu --arch=densenet201 --hidden_units 1280 640 --save_folder densenet201
```

## How to predict (predict.py)

Use the help command to see all supported parameters:
```bash
python ./predict.py -h
```

Basic Prediction for the image `flowers/valid/1/image_06739.jpg`
```bash
python ./predict.py flowers/valid/1/image_06739.jpg checkpoint.pth
```

Prediction with Top 10 Probabilities
```bash
python ./predict.py flowers/valid/1/image_06739.jpg checkpoint.pth --tok_k 10
```

Prediction using GPU
```bash
python ./predict.py flowers/valid/1/image_06739.jpg checkpoint.pth --gpu
```

## Part 1 Development Notebook

### Image Classifier Project.ipynb

To review the  [Image Classifier Project.ipynb] notebook, launch **Jupyter Notebook** from the project root:

```bash
make notebook
```
This command only automates Jupyter Noobook initialization using the command: `jupyter notebook`


#### Files
- [x] Image Classifier Project.html
- [x] Image Classifier Project.ipynb


## Part 2 Scripts

### [train.py](train.py)

**Options:**

- Set folder to save checkpoints
    - `python train.py data_dir --save_folder save_folder`
- Choose architecture between `vgg` or `densenet` models
    - `python train.py data_dir --arch "vgg13"`
- Set hyperparameters
    - `python train.py data_dir --learning_rate 0.01 --hidden_units 512 256 --epochs 20`
- Use GPU for training
    - `python train.py data_dir --gpu`

**Help** - `python ./train.py -h`:
```plain
usage: python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units 3136 --epochs 5

Train and save an image classification model.

positional arguments:
  data_folder

optional arguments:
  -h, --help            show this help message and exit
  --save_folder SAVE_FOLDER   folder to save training checkpoint file (default:
                        .)
  --save_name SAVE_NAME
                        Checkpoint filename. (default: checkpoint)
  --categories_json CATEGORIES_JSON
                        Path to file containing the categories. (default:
                        cat_to_name.json)
  --arch ARCH           Supported architectures: vgg11, vgg13, vgg16, vgg19,
                        densenet121, densenet169, densenet161, densenet201
                        (default: vgg16)
  --gpu                 Use GPU (default: False)

hyperparameters:
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --hidden_units HIDDEN_UNITS [HIDDEN_UNITS ...], -hu HIDDEN_UNITS [HIDDEN_UNITS ...]
                        Hidden layer units (default: [3136, 784])
  --epochs EPOCHS       Epochs (default: 1)
```

### [predict.py](predict.py)

- Basic usage
    - `python predict.py /path/to/image checkpoint`
- Options
    - Return top KK most likely classes
        - `python predict.py input checkpoint --top_k 3`
    - Use a mapping of categories to real name
        - `python predict.py input checkpoint --category_names cat_to_name.json`
    - Use GPU for inference
        - `python predict.py input checkpoint --gpu`

**Help** - `python ./predict.py -h`:
```plain
usage: python ./predict.py /path/to/image.jpg checkpoint.pth

Image prediction.

positional arguments:
  path_to_image         Path to image file.
  checkpoint_file       Path to checkpoint file.

optional arguments:
  -h, --help            show this help message and exit
  --save_folder SAVE_FOLDER   folder to save training checkpoint file (default:
                        .)
  --top_k TOP_K         Return top KK most likely classes. (default: 5)
  --category_names CATEGORIES_JSON
                        Path to file containing the categories. (default:
                        cat_to_name.json)
  --gpu                 Use GPU (default: False)
```

#### Files
- [x] [train.py](train.py)
- [x] [predict.py](predict.py)
- [x] [utils.py](utils.py)
  - Content: Base class to extend commum methods between train.py and predict.py and arguments parsers classes
