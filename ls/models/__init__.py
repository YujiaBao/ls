from .mnistcnn import mnistcnn
from .mlp import mlp
from .bert import bert
from .textcnn import textcnn

# Add support for all torchvision classificaiton models
from .alexnet import alexnet

from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large

from .densenet import densenet121, densenet161, densenet169, densenet201

from .efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, \
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

from .googlenet import googlenet

from .inception import inception_v3

from .mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3

from .mobilenetv2 import mobilenet_v2

from .mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large

from .regnet import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, \
    regnet_y_3_2gf, regnet_y_8gf, regnet_y_16gf, regnet_y_32gf, \
    regnet_y_128gf, regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, \
    regnet_x_3_2gf, regnet_x_8gf, regnet_x_16gf, regnet_x_32gf

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
    resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, \
    wide_resnet101_2

from .shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, \
    shufflenet_v2_x1_5, shufflenet_v2_x2_0

from .squeezenet import squeezenet1_0, squeezenet1_1

from .swin_transformer import swin_t, swin_s, swin_b

from .vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, \
    vgg19_bn

from .vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

# Define the buildin_models list
from .build import ModelFactory
builtin_models = list(ModelFactory.registry.keys())
