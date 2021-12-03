import torchvision
import torchvision.transforms as transforms


def load():

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train = torchvision.datasets.CIFAR10('./tmp', train=True, download=True, transform=transform)

    test = torchvision.datasets.CIFAR10('./tmp', train=False, download=True, transform=transform)

    return train, test
