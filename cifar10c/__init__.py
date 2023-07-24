from .resnet import resnet_32x4
from .wideresnet import wrn_40_2

model_dict = {
    'wrn_40_2': wrn_40_2,
    'rn_32x4': resnet_32x4
}