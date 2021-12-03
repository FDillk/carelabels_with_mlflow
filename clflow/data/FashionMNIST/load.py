import torchvision
import torch


def load():

    train = torchvision.datasets.FashionMNIST('./tmp', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5,), (0.5,))
        ])
    )

    test = torchvision.datasets.FashionMNIST('./tmp', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5,), (0.5,))
        ])
    )

    return train, test
