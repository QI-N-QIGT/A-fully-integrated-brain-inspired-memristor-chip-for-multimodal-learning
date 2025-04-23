import torchvision
from PIL import Image

class MNIST(torchvision.datasets.MNIST):
    def __init__(self,root='', train=False, transform=None):
        super().__init__(root, train, transform,download=False)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
