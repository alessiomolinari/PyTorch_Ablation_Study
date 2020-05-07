import torchvision.models as models
from pytorch_ablation.ablator import Ablator

file = open("../txt_models/vgg16.txt", "w")

vgg16 = models.vgg16(pretrained=True)
print(Ablator._get_module_list(vgg16), file=file)

inception = models.inception_v3()

file = open("../txt_models/inception.txt", "w")
print(Ablator._get_module_list(inception), file=file)

print(Ablator._get_module_list(inception))

