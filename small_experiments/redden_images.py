from torchvision.datasets import FashionMNIST
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor

from dctn.tb_logging import add_good_bad_border, add_good_bad_bar, add_y_dots

ds = FashionMNIST("/mnt/hdd_1tb/datasets/fashionmnist", transform=to_tensor)
img_orig = ds[3][0]  # 1 x 28 x 28
good = 0.98
save_image(add_good_bad_border(img_orig, good), f"add_good_bad_border_{good=}.png")
save_image(add_good_bad_bar(img_orig, good), f"add_good_bad_bar_{good=}.png")
index = ds[3][1]
print(index)
save_image(
    add_y_dots(add_good_bad_border(img_orig, good), index),
    f"y_dots_add_good_bad_border_{good=}.png",
)
save_image(
    add_y_dots(add_good_bad_bar(img_orig, good), index),
    f"y_dots_add_good_bad_bar_{good=}.png",
)
