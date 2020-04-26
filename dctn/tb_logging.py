import math

import torch
import torch.nn.functional as F


def add_good_bad_border(img: torch.Tensor, how_good: float) -> torch.Tensor:
    assert img.ndim == 3
    assert img.shape[0] == 1
    padding = 3
    red_channel = F.pad(img, (padding,) * 4, value=1.0 - how_good)
    green_channel = F.pad(img, (padding,) * 4, value=how_good)
    blue_channel = F.pad(img, (padding,) * 4, value=0.0)
    img_result = torch.cat((red_channel, green_channel, blue_channel))
    return img_result


def add_good_bad_bar(img: torch.Tensor, how_good: float) -> torch.Tensor:
    padding = 3
    padded = F.pad(img.expand(3, *img.shape[1:]), (padding,) * 4, value=1.0)
    new_width = padded.shape[2]

    if not math.isfinite(how_good):
        # Pink is #FF54FF
        padded[0, -padding:] = 0xFF / 255.0
        padded[1, -padding:] = 0x54 / 255.0
        padded[2, -padding:] = 0xFF / 255.0
    else:
        bar_width = round(how_good * new_width)
        if how_good >= 0.5:
            channel = 1
            value = (how_good - 0.5) * 2
            bar_width = round((how_good - 0.5) * 2 * new_width)
        else:
            channel = 0
            value = (0.5 - how_good) * 2
            bar_width = round((0.5 - how_good) * 2 * new_width)
        for c in range(3):
            if c == channel:
                padded[c, -padding:, :bar_width] = value
            else:
                padded[c, -padding:, :bar_width] = 0.0
    return padded


def add_y_dots(img: torch.Tensor, y: int, padding: int = 3) -> torch.Tensor:
    assert img.ndim == 3
    assert img.shape[0] == 3
    assert img.shape[2] >= y * 2
    new_img = img.clone()
    for i in range(y):
        new_img[2, :padding, 2 * i] = 1.0
        new_img[:2, :padding, 2 * i] = 0.0
    return new_img
