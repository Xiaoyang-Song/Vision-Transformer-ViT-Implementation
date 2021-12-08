import torch
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


def get_cifar_10_dataset(width, height):
    transform = transforms.Compose([Resize((width, height)), ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                                 transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                                transform=transform)
    return train_dataset, test_dataset


def load_data(dataset, batch_size, shuffle):
    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_data


# TODO: Delete the following test codes later
def simple_test_patch_embedding():
    batch_size = 8
    cifar_train, cifar_test = get_cifar_10_dataset(32, 32)
    loader = load_data(cifar_train, batch_size, False)
    single_image = cifar_train.__getitem__(0)[0]
    single_image = torch.repeat_interleave(torch.unsqueeze(single_image, dim=0), batch_size, dim=0)
    hidden_size = 123
    patch_embedding = PatchEmbedding(hidden_size, 32, 32, 4, 3)
    print(patch_embedding(single_image).shape)

# simple_test_patch_embedding()
