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
