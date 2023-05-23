import skimage
import torch

class Rescale:
    """Rescale the image and label to the specified output size.

    Args:
        output_size (int or tuple): Desired output size. If an int, the smaller
            dimension of the image will be rescaled proportionally. If a tuple,
            output_size should be a tuple of (new_height, new_width).

    Example:
        >>> import numpy as np
        >>> sample = {'image': np.random.rand(100, 200, 3), 'label': 'cat'}
        >>> rescale_transform = Rescale(256)
        >>> transformed_sample = rescale_transform(sample)
        >>> transformed_sample['image'].shape
        (128, 256, 3)
        >>> transformed_sample['label']
        'cat'
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = skimage.transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}
    
class ToTensor:
    """Convert a dictionary of 'image' and 'label' to PyTorch Tensors.

    Converts the 'image' and 'label' values in the input dictionary to PyTorch
    Tensors. The 'image' value is transposed to match the format expected by
    PyTorch (C x H x W). The 'image' and 'label' values are then converted to
    float Tensors.

    Example:
        >>> import numpy as np
        >>> sample = {'image': np.random.rand(3, 100, 200), 'label': 1}
        >>> to_tensor_transform = ToTensor()
        >>> transformed_sample = to_tensor_transform(sample)
        >>> transformed_sample['image'].shape
        torch.Size([3, 100, 200])
        >>> transformed_sample['label']
        tensor([1.])
    """
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': torch.Tensor([label]).float()}