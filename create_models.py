from inspect import signature
import os

import numpy as np
# mlflow
import mlflow.sklearn
import mlflow.pytorch
# random forest imports
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# dnn imports
import torch
import torch.nn as nn
import torch.optim as optim

from clflow.data.loader import load_as_dataloader, load_as_arrays


class CustomRandomForest(RandomForestClassifier):

    def predict(self, X):
        # flatten out the feature in one dim
        if len(X.shape) > 2:
            n_feat = np.product(X.shape) // X.shape[0]
            X = X.reshape(X.shape[0], n_feat)
        return super().predict(X)

    def fit(self, X, y, sample_weight=None):
        # flatten out the feature in one dim
        if len(X.shape) > 2:
            n_feat = np.product(X.shape) // X.shape[0]
            X = X.reshape(X.shape[0], n_feat)
        return super().fit(X, y, sample_weight=sample_weight)


class Net(nn.Module):
    def __init__(self, nr_classes, data_shape):
        linear_tensors = 4096 if data_shape[1] != 28 else 2304
        super(Net, self).__init__()

        self.encoder = nn.Sequential(*[
            # Conv Layer block 1
            nn.Conv2d(in_channels=1 if len(data_shape) == 3 else data_shape[-1], out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(linear_tensors, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        ])

        self.classifier = nn.Sequential(*[
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=-1)
        ])

    def forward(self, x):
        embedding = self.encoder(x)
        return self.classifier(embedding)


def train_random_forest(data):

    X_train, y_train, X_test, y_test = load_as_arrays(data)

    model_dir = os.path.join('clflow', 'models', f'rf_{data}')

    if not os.path.exists(model_dir):

        rf=CustomRandomForest(n_estimators=100)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        print(accuracy_score(y_test, y_pred))

        mlflow.sklearn.save_model(rf, model_dir)


def train_neural_network(data):

    train_loader, test_loader = load_as_dataloader(data, batch_size=64)

    model_dir = os.path.join('clflow', 'models', f'dnn_{data}')

    if not os.path.exists(model_dir):

        # input = {
        #     'name': info["name"],
        #     'dtype': str(train_loader.dataset.data.dtype).split('.')[1],
        #     'shape': np.array([-1, 28, 28, 1]) if info["name"] == 'MNIST' else np.array([-1, 32, 32, 3])
        # }
        # output = {
        #     'shape': np.array([-1, 10]),
        #     'dtype': 'float32'
        # }
        # signature = infer_signature(input, output)
        # inputs: '[{"name": "images", "dtype": "uint8", "shape": [-1, 28, 28, 1]}]'
        # outputs: '[{"shape": [-1, 10], "dtype": "float32"}]'

        epochs = 10

        net = Net(10, train_loader.dataset.data.shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)

        net.to(device)

        for epoch in range(epochs):
            net.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                pred = net(data)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                correct = 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    pred = net(data).data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
                train_acc = 100. * correct / len(train_loader.dataset)

                correct = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = net(data).data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
                test_acc = 100. * correct / len(test_loader.dataset)

                print(f'EPOCH {epoch} train acc {train_acc:5.3} test acc {test_acc:5.3f} ')
        
        mlflow.pytorch.save_model(net, model_dir)

# train model

# for data in ['MNIST', 'CIFAR10', 'FashionMNIST']:

#     train_random_forest(data)
#     train_neural_network(data)
