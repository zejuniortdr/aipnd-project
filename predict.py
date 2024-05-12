#!/usr/bin/env python3
""" predict.py
Udacity AI Programming with Python Project final project: Part 2
Submission for Luis Jr.
This file trains a new network on a specified data set.
"""
__author__ = "Luis Jr <jose.luis@iclinic.com.br>"
__version__ = "1.0.0"
__license__ = "MIT"

import torch
import warnings

from PIL import Image
from torchvision import models
from torchvision import transforms

from utils import PredictArgs, Util


class Predict(Util):
    def __init__(self):
        predict_args = PredictArgs()
        parser = predict_args.get_args()
        parser.add_argument(
            "--version",
            action="version",
            version=__version__
        )
        self.cli_args = parser.parse_args()

    def execute(self):
        """
            Image Classification Prediction
        """

        # Start with CPU
        device = self.setup_device()

        # load categories
        self.load_categories()

        # load model
        self.chkp_model = self.load_checkpoint(
            device,
            self.cli_args.checkpoint_file
        )

        self.top_prob, self.top_classes = self.predict(
            self.cli_args.path_to_image,
            self.chkp_model,
            self.cli_args.top_k
        )

        self.label = self.top_classes[0]
        self.prob = self.top_prob[0]

        self.print_parameters()

    def predict(self, image_path, model, topk=5):
        model.eval()
        model.cpu()

        image = self.process_image(image_path)

        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model.forward(image)
            top_prob, top_labels = torch.topk(output, topk)
            top_prob = top_prob.exp()

        class_to_idx_inv = {
            model.class_to_idx[k]: k for k in model.class_to_idx
        }
        mapped_classes = list()

        for label in top_labels.numpy()[0]:
            mapped_classes.append(class_to_idx_inv[label])

        return top_prob.numpy()[0], mapped_classes

    def load_checkpoint(self, device, file="checkpoint.pth"):
        """
        Loads model checkpoint saved by train.py
        """
        model_state = torch.load(
            file,
            map_location=lambda storage, loc: storage
        )

        model = models.__dict__[model_state["arch"]](pretrained=True)
        model = model.to(device)

        model.classifier = model_state["classifier"]
        model.load_state_dict(model_state["state_dict"])
        model.class_to_idx = model_state["class_to_idx"]

        return model

    def process_image(self, image):
        """
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        """
        expects_means = [0.485, 0.456, 0.406]
        expects_std = [0.229, 0.224, 0.225]

        pil_image = Image.open(image).convert("RGB")

        in_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(expects_means, expects_std)
            ]
        )
        pil_image = in_transforms(pil_image)

        return pil_image

    def print_parameters(self):
        print("Parameters\n=========")

        print(f"Image  : {self.cli_args.path_to_image}")
        print(f"Model  : {self.cli_args.checkpoint_file}")
        print(f"Device : {self.device}")

        print("\nPrediction\n=========")

        print(f"Flower      : {self.cat_to_name[self.label]}")
        print(f"Label       : {self.label}")
        print(f"Probability : {self.prob*100:.2f}%")

        print("\nTop K\n=========")

        for i in range(len(self.top_prob)):
            print(
                f"{self.cat_to_name[self.top_classes[i]]:<25} "
                f"{self.top_prob[i]:.2%}"
            )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predict = Predict()
        predict.execute()
