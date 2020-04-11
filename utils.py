import torch
import torch.nn.functional as F
import torchvision
import json
import math
import os
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import requests


def get_device(gpu=0):
    r"""Get the :class`torch.device` to use; specify device with :attr:`gpu`.

    Args:
        gpu (int, optional): Index of the GPU device; specify ``None`` to
            force CPU. Default: ``0``.

    Returns:
        :class:`torch.device`: device to use.
    """
    device = torch.device(
        f'cuda:{gpu}'
        if torch.cuda.is_available() and gpu is not None
        else 'cpu')
    return device

def get_example_data(img_file, arch, shape=224):
    """Get example data to demonstrate visualization techniques.

    Args:
        arch (str, optional): name of torchvision.models architecture.
            Default: ``'vgg16'``.
        shape (int or tuple of int, optional): shape to resize input image to.
            Default: ``224``.

    Returns:
        (:class:`torch.nn.Module`, :class:`torch.Tensor`, int, int): a tuple
        containing

            - a convolutional neural network model in evaluation mode.
            - a sample input tensor image.
            - the ImageNet category id of an object in the image.
            - the ImageNet category id of another object in the image.

    """

    # Get a network pre-trained on ImageNet.
    model = torchvision.models.__dict__[arch](pretrained=True)

    # Switch to eval mode to make the visualization deterministic.
    model.eval()

    # We do not need grads for the parameters.
    for param in model.parameters():
        param.requires_grad_(False)

    # response = requests.get(url)
    # img = Image.open(BytesIO(response.content))
    img=Image.open(img_file)
    # Pre-process the image and convert into a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(shape),
        torchvision.transforms.CenterCrop(shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    x = transform(img).unsqueeze(0)

    # Move model and input to device.
    # from torchray.utils import get_device
    dev = get_device()
    model = model.to(dev)
    x = x.to(dev)

    return  model, x
