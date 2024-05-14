#!/usr/bin/env python3
""" utils.py
Udacity AI Programming with Python Project final project: Part 2
Submission for Luis Jr.
This file trains a new network on a specified data set.
"""
__author__ = "Luis Jr <jose.luis@iclinic.com.br>"
__version__ = "1.0.0"
__license__ = "MIT"

import argparse
import json
import torch


class BaseClassUtil(object):
    def setup_device(self):
        device = torch.device("cpu")

        if self.cli_args.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU is not available. Using CPU.")

        return device

    def load_categories(self):
        with open(self.cli_args.categories_json, "r") as f:
            return json.load(f)


class TrainingArgs(object):
    def __init__(self):
        self.supported_arch = [
            "vgg11", "vgg13", "vgg16", "vgg19",
            "densenet121", "densenet169", "densenet161", "densenet201"
        ]

    def get_args(self):
        """
        Get argument parser for train cli.
        Returns an argparse parser.
        """

        parser = argparse.ArgumentParser(
            description="Train and save an image classification model.",
            usage=(
                "python ./train.py ./flowers/train "
                "--gpu --learning_rate 0.001 "
                "--hidden_units 3136 --epochs 5"
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument("data_folder", action="store")

        parser.add_argument(
            "--save_folder",
            action="store",
            default=".",
            dest="save_folder",
            type=str,
            help="Folder to save training checkpoint file",
        )

        parser.add_argument(
            "--save_name",
            action="store",
            default="checkpoint",
            dest="save_name",
            type=str,
            help="Checkpoint filename.",
        )

        parser.add_argument(
            "--categories_json",
            action="store",
            default="cat_to_name.json",
            dest="categories_json",
            type=str,
            help="Path to file containing the categories.",
        )

        parser.add_argument(
            "--arch",
            action="store",
            default="vgg16",
            dest="arch",
            type=str,
            help="Supported architectures: " + ", ".join(self.supported_arch),
        )

        parser.add_argument(
            "--gpu",
            action="store_true",
            dest="use_gpu",
            default=False,
            help="Use GPU"
        )

        hp = parser.add_argument_group("hyperparameters")

        hp.add_argument(
            "--learning_rate",
            action="store",
            default=0.001,
            type=float,
            help="Learning rate"
        )

        hp.add_argument(
            "--hidden_units", "-hu",
            action="store",
            dest="hidden_units",
            default=[3136, 784],
            type=int,
            nargs="+",
            help="Hidden layer units"
        )

        hp.add_argument(
            "--epochs",
            action="store",
            dest="epochs",
            default=1,
            type=int,
            help="Epochs"
        )

        parser.parse_args()
        return parser


class PredictArgs(object):
    def get_args(self):
        parser = argparse.ArgumentParser(
            description="Image prediction.",
            usage="python ./predict.py /path/to/image.jpg checkpoint.pth",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        parser.add_argument(
            "path_to_image",
            help="Path to image file.",
            action="store"
        )

        parser.add_argument(
            "checkpoint_file",
            help="Path to checkpoint file.",
            action="store"
        )

        parser.add_argument(
            "--save_folder",
            action="store",
            default=".",
            dest="save_folder",
            type=str,
            help="Folder to save training checkpoint file",
        )

        parser.add_argument(
            "--top_k",
            action="store",
            default=5,
            dest="top_k",
            type=int,
            help="Return top KK most likely classes.",
        )

        parser.add_argument(
            "--category_names",
            action="store",
            default="cat_to_name.json",
            dest="categories_json",
            type=str,
            help="Path to file containing the categories.",
        )

        parser.add_argument(
            "--gpu",
            action="store_true",
            dest="use_gpu",
            default=False,
            help="Use GPU"
        )

        parser.parse_args()
        return parser
