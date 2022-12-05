# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""MedMNIST aggregator for PyTorch."""

import logging

from flame.config import Config
from flame.dataset import Dataset # Not sure why we need this.
from flame.mode.horizontal.top_aggregator import TopAggregator
import torch
import torchvision
import medmnist
from medmnist import INFO
from sklearn.metrics import accuracy_score


logger = logging.getLogger(__name__)

class CNN(torch.nn.Module):
    """CNN Class"""

    def __init__(self, num_classes):
        """Initialize."""
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = torch.nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PyTorchMedMNistAggregator(TopAggregator):
    """PyTorch MedMNist Aggregator"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None
        self.dataset: Dataset = None # Not sure why we need this.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_size = 0

        self.train_loader = None
        self.val_loader = None

        self.epochs = self.config.hyperparameters['epochs']
        self.batch_size = self.config.hyperparameters['batchSize']

    def initialize(self):
        """Initialize."""
        #self.model = torchvision.models.resnet50()
        self.model = CNN(num_classes=9)
        self.moptimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_data(self) -> None:
        """MedMNIST Pathology Dataset
        The dataset is kindly released by Jakob Nikolas Kather, Johannes Krisam, et al. (2019) in their paper 
        "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study",
        and made available through Yang et al. (2021) in 
        "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis".
        Dataset Repo: https://github.com/MedMNIST/MedMNIST
        """
        info = INFO["pathmnist"]
        DataClass = getattr(medmnist, info['python_class'])
 
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_dataset = DataClass(split='train', transform=data_transform, download=True)
        val_dataset = DataClass(split='val', transform=data_transform, download=True)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4 * torch.cuda.device_count(),
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4 * torch.cuda.device_count(),
            pin_memory=True,
            drop_last=True
        )

        self.dataset_size = len(train_dataset)

    def train(self) -> None:
        """Train a model."""
        # Implement this if training is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate a model."""
        logger.info("Started Evaluating of Model")
        self.model.eval()
        # loss_lst = list()
        # labels = torch.tensor([],device=self.device)
        # labels_pred = torch.tensor([],device=self.device)
        # with torch.no_grad():
        #     for data, label in self.train_loader:
        #         data, label = data.to(self.device), label.to(self.device)
        #         output = self.model(data)
        #         loss = self.criterion(output, label.squeeze())
        #         loss_lst.append(loss.item())
        #         labels_pred = torch.cat([labels_pred, output.argmax(dim=1)], dim=0)
        #         labels = torch.cat([labels, label], dim=0)

        # labels_pred = labels_pred.cpu().detach().numpy()
        # labels = labels.cpu().detach().numpy()
        # val_acc = accuracy_score(labels, labels_pred)

        # val_loss = sum(loss_lst) / len(loss_lst)
        val_acc = 11

        val_loss = 12
        self.update_metrics({"Val Loss": val_loss, "Val Accuracy": val_acc})
        logger.info(f"Test Loss: {val_loss}")
        logger.info(f"Test Accuracy: {val_acc}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('config', nargs='?', default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchMedMNistAggregator(config)
    a.compose()
    a.run()