import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Apparently better to choose a pre-trained model that is lower resolution than the
# fine-tune database's images, i.e. for this choose a 224 model.
# Probably vit_base_patch32_224.augreg_in21k_ft_in1k
# 256 x 240
trainDataset = ImageFolder("/home/groups/comp3710/ADNI", transform=transform)