from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

ds_path = "/mnt/hdd_1tb/datasets/cifar10"
tv_ds = CIFAR10(ds_path)
image_rgb = tv_ds[0][0]
image_rgb_tensor = to_tensor(image_rgb)

image_ycbcr = image_rgb.convert("YCbCr")
image_ycbcr_tensor = to_tensor(image_ycbcr)

save_image(image_ycbcr_tensor, "image_ycbcr_tensor.png")

for channel in image_ycbcr_tensor:
    print(f"{channel.mean()=}, {channel.std()=}, {channel.min()=}, {channel.max()=}")
