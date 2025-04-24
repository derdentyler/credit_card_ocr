from PIL import Image
import random
import torch

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor:
    """
    Преобразование PIL.Image -> torch.Tensor для image,
    остальные поля оставляем без изменений.
    """
    def __call__(self, sample):
        image, boxes, labels = sample['image'], sample['boxes'], sample['labels']
        image = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
             .view(*image.size[::-1], 3)
             .permute(2, 0, 1)
             .float() / 255.0
            ).numpy()
        )
        return {'image': image, 'boxes': boxes, 'labels': labels}

class RandomHorizontalFlip:
    """
    С вероятностью 0.5 зеркально отражает изображение и соответственные xmin/xmax.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, boxes, labels = sample['image'], sample['boxes'], sample['labels']
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            w, _ = image.size
            # инверсия bbox по горизонтали
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
            return {'image': image, 'boxes': boxes, 'labels': labels}
        return sample
