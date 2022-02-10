import PIL.Image as Img
from torchvision.transforms.functional import InterpolationMode

INTERPOLATION_STRING_TO_TYPE = {
    'nearest': InterpolationMode.NEAREST,
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS
                                }