import torch
import math
import warnings
import torchvision.transforms as transforms

from torch.nn.modules.utils import _ntuple
from PIL import Image
from PIL import ImageOps


def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    r"""Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    Ripped from master branch of PyTorch, which is not stable yet. Just need this fn.
    https://pytorch.org/docs/master/_modules/torch/nn/functional.html#interpolate

    Not ideal but will work for now.
    """

    def _check_size_scale_factor(dim):
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if size is not None and scale_factor is not None:
            raise ValueError('only one of size or scale_factor should be defined')
        if scale_factor is not None and isinstance(scale_factor, tuple)\
                and len(scale_factor) != dim:
            raise ValueError('scale_factor shape must match input shape. '
                             'Input is {}D, scale_factor size is {}'.format(dim, len(scale_factor)))

    def _output_size(dim):
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)]

    if mode in ('nearest', 'area'):
        if align_corners is not None:
            raise ValueError("align_corners option can only be set with the "
                             "interpolating modes: linear | bilinear | trilinear")
    else:
        if align_corners is None:
            warnings.warn("Default upsampling behavior when mode={} is changed "
                          "to align_corners=False since 0.4.0. Please specify "
                          "align_corners=True if the old behavior is desired. "
                          "See the documentation of nn.Upsample for details.".format(mode))
            align_corners = False

    if input.dim() == 3 and mode == 'bilinear':
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    elif input.dim() == 4 and mode == 'bilinear':
        return torch._C._nn.upsample_bilinear2d(input, _output_size(2), align_corners)
    elif input.dim() == 5 and mode == 'bilinear':
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")
    else:
        raise NotImplementedError("Input Error: Only 3D, 4D and 5D input Tensors supported"
                                  " (got {}D) for the modes: nearest | linear | bilinear | trilinear"
                                  " (got {})".format(input.dim(), mode))


class resizeDifferentiableNormalize(object):
    """
    NOT IMPLEMENTED. WIP
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        :param size: (imgW, imgH)
        :param interpolation:
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        :param img: 3D image tensor of shape (1, height, width)
                    or 2D image tensor of shape (height, width)
        :return: resized image tensor to self.size
        """
        return_viewed = False
        if img.shape[0] == 1:
            img = img[0]
            return_viewed = True

        new_img = torch.ones(self.size).to(dtype=torch.float32)

        # Resize image as necessary to new height, maintaining aspect ratio
        o_size = img.shape
        # Width / height aspect ration
        AR = o_size[0] / float(o_size[1])
        new_width = int(round(AR * self.size[1]))
        new_height = self.size[1]
        img = img.resize((new_width, new_height), self.interpolation)

        # Now pad to new width, as target width is guaranteed to be larger than width if keep aspect ratio is true
        o_size = img.shape
        delta_w = self.size[0] - o_size[0]
        delta_h = self.size[1] - o_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_im = ImageOps.expand(img, padding, "white")

        img.sub_(0.5).div_(0.5)

        return img