#!/usr/bin/env python3
""" train.py
Udacity AI Programming with Python Project final project: Part 2
Submission for Luis Jr.
This file trains a new network on a specified data set.
"""
__author__ = "Luis Jr <jose.luis@iclinic.com.br>"
__version__ = "1.0.0"
__license__ = "MIT"

import os
import sys

import torch

from collections import OrderedDict
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import BaseClassUtil, TrainingArgs


class Train(BaseClassUtil):

    def __init__(self):
        # ARGS
        train_args = TrainingArgs()
        parser = train_args.get_args()
        parser.add_argument(
            "--version",
            action="version",
            version=__version__
        )
        self.cli_args = parser.parse_args()

        # prep data loader
        self.expected_means = [0.485, 0.456, 0.406]
        self.expected_std = [0.229, 0.224, 0.225]
        self.max_image_size = 224
        self.batch_size = 32

        self.is_vgg = "vgg" in self.cli_args.arch
        self.is_densenet = "densenet" in self.cli_args.arch
        self.is_supported = all([self.is_vgg, self.is_densenet])

        self.densenet_input = {
            "densenet121": 1024,
            "densenet169": 1664,
            "densenet161": 2208,
            "densenet201": 1920
        }

    def setup_transforms(self):
        return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.25),
                    transforms.RandomRotation(25),
                    transforms.RandomGrayscale(p=0.02),
                    transforms.RandomResizedCrop(self.max_image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        self.expected_means,
                        self.expected_std
                    )
                ]
            )

    def get_training_dataloader(self, training_dataset):
        return DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def execute(self):
        """
            Image Classification Network Trainer
        """
        self.check_folder(self.cli_args.data_folder)
        self.check_folder(self.cli_args.save_folder, create=True)

        cat_to_name = self.load_categories()

        # set output to the number of categories
        output_size = len(cat_to_name)
        print(f"Images are labeled with {output_size} categories.")

        training_transforms = self.setup_transforms()

        training_dataset = datasets.ImageFolder(
            self.cli_args.data_folder,
            transform=training_transforms
        )

        self.training_dataloader = self.get_training_dataloader(
            training_dataset
        )

        self.nn_model = self.make_model(self.cli_args)

        input_size = 0

        # Input size from current classifier if VGG
        if self.is_vgg:
            input_size = self.nn_model.classifier[0].in_features

        elif self.is_densenet:
            input_size = self.densenet_input[self.cli_args.arch]

        self.prevent_backpropagation()

        classifier = nn.Sequential(
            self.setup_sequential(input_size, output_size)
        )

        self.nn_model.classifier = classifier
        self.nn_model.zero_grad()
        self.criterion = nn.NLLLoss()
        print(
            "Setting optimizer learning rate to "
            f"{self.cli_args.learning_rate}."
        )
        self.optimizer = optim.Adam(
            self.nn_model.classifier.parameters(),
            lr=self.cli_args.learning_rate
        )

        self.device = self.setup_device()

        self.nn_model = self.send_model_to_device()

        data_set_len = len(self.training_dataloader.batch_sampler)

        self.chk_every = 50

        self.train()

        self.print_training_messages(data_set_len)

    def check_folder(self, folder, create=False):
        if not create and not os.path.isdir(folder):
            print(f"Data folder {folder} does not exist")
            exit(1)
        elif create and not os.path.isdir(folder):
            print(f"Folder {folder} does not exist. Creating...")
            os.makedirs(folder)

    def make_model(self):
        if self.is_supported:
            print("Choose VGG or Densenet")
            exit(1)

        print(f"Using a pre-trained {self.cli_args.arch} network.")
        return models.__dict__[self.cli_args.arch](pretrained=True)

    def prevent_backpropagation(self):
        for param in self.nn_model.parameters():
            param.requires_grad = False

    def setup_sequential(self, input_size, output_size):
        od = OrderedDict()
        hidden_sizes = self.cli_args.hidden_units
        print(f"Building a {len(hidden_sizes)} hidden layer classifier with inputs {hidden_sizes}")

        hidden_sizes.insert(0, input_size)

        for i in range(len(hidden_sizes) - 1):
            od[f"fc{i+1}"] = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            od[f"relu{i+1}"] = nn.ReLU()
            od[f"dropout{i+1}"] = nn.Dropout(p=0.15)

        od["output"] = nn.Linear(hidden_sizes[i + 1], output_size)
        od["softmax"] = nn.LogSoftmax(dim=1)

        return od

    def send_model_to_device(self):
        print(f"Sending model to device {self.device}.")
        self.nn_model = self.nn_model.to(self.device)
        return self.nn_model

    def train(self):
        for e in range(self.cli_args.epochs):
            e_loss = 0
            prev_chk = 0
            total = 0
            correct = 0
            print(f"\nEpoch {e+1} of {self.cli_args.epochs}\n=========")
            for ii, (images, labels) in enumerate(self.training_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.nn_model.forward(images)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                e_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                itr = (ii + 1)
                if itr % self.chk_every == 0:
                    avg_loss = f"avg. loss: {e_loss/itr:.4f}"
                    acc = f"accuracy: {correct/total:.2%}"
                    print(f"  Batches {prev_chk:03} to {itr:03}: {avg_loss}, {acc}.")
                    prev_chk = (ii + 1)

        print(f"{self.cli_args.epochs} epochs done. Saving.")

        self.nn_model.class_to_idx = self.training_dataset.class_to_idx
        model_state = {
            "epoch": self.cli_args.epochs,
            "state_dict": self.nn_model.state_dict(),
            "optimizer_dict": self.optimizer.state_dict(),
            "classifier": self.nn_model.classifier,
            "class_to_idx": self.nn_model.class_to_idx,
            "arch": self.cli_args.arch
        }

        save_location = (
            f"{self.cli_args.save_folder}/{self.cli_args.save_name}.pth"
        )
        print(f"Saving checkpoint to {save_location}")

        torch.save(model_state, save_location)

    def print_training_messages(self, data_set_len):
        print(f"Training with the device: {self.device}")
        print(
            f"Training on {data_set_len} of "
            f"{self.training_dataloader.batch_size}."
        )
        print(
            "Showing average loss and accuracy for epoch "
            f"after every {self.chk_every} batches."
        )


if __name__ == "__main__":
    try:
        train = Train()
        train.execute()
    except KeyboardInterrupt:
        sys.exit(0)
