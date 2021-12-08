import torchvision.datasets
from torchvision.transforms import Resize, ToTensor
from Architecture import *


def check_version():
    print("===== Vision Transformer Requirements =====")
    print("============= Library Version =============")
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print("Numpy Version: ", np.__version__)
    print("===========================================")


def check_device():
    print("========== Device Configurations ==========")
    if torch.cuda.is_available():
        print("Device: Graphic Processing Unit (GPU) ")
    else:
        print("Device: Central Processing Unit (CPU) ")
    print("===========================================")


def process_image():
    pass


def get_cifar10_dataset(width, height):
    transform = transforms.Compose([Resize((width, height)), ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                                 transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                                transform=transform)
    return train_dataset, test_dataset


def load_data(dataset, batch_size, shuffle):
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_data

