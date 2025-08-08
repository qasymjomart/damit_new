import monai
from monai.transforms import (
    Crop,
    Randomizable,
    Compose,
    OneOf,
    CropForeground,
    ToTensor
)

import math
from monai.data.utils import get_random_patch, get_valid_patch_size
from torch.nn.functional import interpolate

class RandomResizedCrop3d(Crop, Randomizable):
    def __init__(
        self,
        size,
        in_slice_scale,
        cross_slice_scale,
        interpolation='trilinear',
        aspect_ratio=(0.9, 1/0.9),
    ):
        """
        Adapting torch RandomResizedCrop to 3D data by separating in-slice/in-plane and cross-slice dimensions.

        Args:
            size: Size of output image.
            in_slice_scale: Range of the random size of the cropped in-slice/in-plane dimensions.
            cross_slice_scale: Range of the random size of the cropped cross-slice dimensions.
            interpolation: 3D interpolation method, defaults to 'trilinear'.
            aspect_ratio: Range of aspect ratios of the cropped in-slice/in-plane dimensions.
        """
        super().__init__()
        self.size = size
        self.in_slice_scale = in_slice_scale
        self.cross_slice_scale = cross_slice_scale
        self.interpolation = interpolation
        self.aspect_ratio = aspect_ratio
        self._slices: tuple[slice, ...] = ()

    def get_in_slice_crop(self, height, width):
        """
        Adapted from torchvision RandomResizedCrop, applied to the in-slice/in-plane dimensions
        """
        area = height * width

        log_ratio = math.log(self.aspect_ratio[0]), math.log(self.aspect_ratio[1])
        for _ in range(10):
            target_area = area * self.R.uniform(*self.in_slice_scale)
            aspect_ratio = math.exp(self.R.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                return h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.aspect_ratio):
            w = width
            h = int(round(w / min(self.aspect_ratio)))
        elif in_ratio > max(self.aspect_ratio):
            h = height
            w = int(round(h * max(self.aspect_ratio)))
        else:  # whole image
            w = width
            h = height
        return h, w

    def randomize(self, img_size):
        # first two dimensions are dicom slice dims/in-plane dims, third is number of slices
        height, width, depth = img_size

        # get in-slice crop size
        crop_h, crop_w = self.get_in_slice_crop(height, width)

        # get cross-slice crop size
        crop_d = int(round(depth * self.R.uniform(*self.cross_slice_scale)))

        crop_size = (crop_h, crop_w, crop_d)
        valid_size = get_valid_patch_size(img_size, crop_size)
        self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img, lazy=False):
        self.randomize(img.shape[1:])
        cropped = super().__call__(img=img, slices=self._slices)
        resized = interpolate(cropped.unsqueeze(0), size=self.size, mode=self.interpolation).squeeze(0)
        return resized


class CropForegroundSwapSliceDims(CropForeground):
    """
    Same functionality as CropForeground, but permutes in-plane dimensions to first two spatial dims for
    RandomResizedCrop3d.
    """
    @staticmethod
    def get_permutation(shape_or_spacing):
        # get permutation for how to swap slice axes, add small tolerance
        if abs(shape_or_spacing[0] - shape_or_spacing[1]) < 1e-2:
            permutation = (0, 1, 2, 3)
        elif abs(shape_or_spacing[0] - shape_or_spacing[2]) < 1e-2:
            permutation = (0, 1, 3, 2)
        elif abs(shape_or_spacing[1] - shape_or_spacing[2]) < 1e-2:
            permutation = (0, 2, 3, 1)
        else:
            permutation = None
        return permutation

    def __call__(self, img_dict, mode=None, lazy=None, **pad_kwargs):
        # get image spacing and spatial dims
        try:
            img_spacing = img_dict['spacing']
        except KeyError:
            img_spacing = None
        img = img_dict['image']
        spatial_dims = img.shape[1:]

        # try getting from pixel spacing first, NOTE: verified that at least two dims have similar spacing in datasets
        if img_spacing is not None:
            perm = self.get_permutation(img_spacing)
        else:
            perm = self.get_permutation(spatial_dims)

        if perm is None:
            raise RuntimeError('Could not determine slice dimension permutation')

        # swap slice dims
        img = img.permute(*perm)

        # crop foreground
        return super().__call__(img, mode, lazy, **pad_kwargs)

class DataAugmentation3DINO:
    def __init__(self):
        """
        Initialize the 3DINO data augmentation pipeline.

        Args:
            size: Size of output image.
            in_slice_scale: Range of the random size of the cropped in-slice/in-plane dimensions.
            cross_slice_scale: Range of the random size of the cropped cross-slice dimensions.
            interpolation: Interpolation method for resizing.
        """
        self.first_trans = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image",]),
            monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple([1.75, 1.75, 1.75])),
            # monai.transforms.Resized(keys=["image"], spatial_size=tuple([128, 128, 128])),
        ])
        
        self.global_transforms = Compose([
            monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
            CropForegroundSwapSliceDims(select_fn=lambda x: x > -1),
            RandomResizedCrop3d(size=(128, 128, 128), 
                                in_slice_scale=(0.48, 1.0), 
                                cross_slice_scale=(0.5, 1.0)),
        ])
        
        self.local_transforms = Compose([
            monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
            CropForegroundSwapSliceDims(select_fn=lambda x: x > -1),
            RandomResizedCrop3d(size=(64, 64, 64), 
                                in_slice_scale=(0.16, 0.48), 
                                cross_slice_scale=(0.2, 0.5),),
        ])
        
        # noise, contrast, blurring
        gaussian_transforms = OneOf(
            [
                monai.transforms.RandAdjustContrast(prob=0.8, gamma=(0.5, 2)),
                monai.transforms.RandGaussianNoise(prob=0.8, std=0.002),
                monai.transforms.RandHistogramShift(num_control_points=10, prob=0.8),
            ]
        )

        global_transfo1_extra = OneOf(
            [
                monai.transforms.RandGaussianSmooth(prob=1.0),
                monai.transforms.RandGaussianSharpen(prob=1.0),
            ]
        )

        global_transfo2_extra = monai.transforms.Compose(
            [
                OneOf(
                    [
                        monai.transforms.RandGaussianSmooth(prob=0.1),
                        monai.transforms.RandGaussianSharpen(prob=0.1),
                    ]
                ),
                monai.transforms.RandGibbsNoise(prob=0.2)
            ]
        )

        local_transfo_extra = monai.transforms.RandGaussianSmooth(prob=0.5)

        self.global_transfo1 = Compose([gaussian_transforms, global_transfo1_extra, ToTensor()])
        self.global_transfo2 = Compose([gaussian_transforms, global_transfo2_extra, ToTensor()])
        self.local_transfo = Compose([gaussian_transforms, local_transfo_extra, ToTensor()])
        
    def __call__(self, img_dict):
        output = []

        # image = self.load_and_normalize(image_path)
        image = self.first_trans(img_dict)

        # global crops:
        im1_base = self.global_transforms(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.global_transforms(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output.append(global_crop_1)
        output.append(global_crop_2)

        # local crops:
        local_crops = [
            self.local_transfo(self.local_transforms(image)) for _ in range(8)
        ]
        output.extend(local_crops)

        # "label" expected, but return nothing
        return output