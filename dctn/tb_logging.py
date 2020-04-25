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

    bar_width = round(how_good * new_width)
    if how_good >= 0.5:
        channel = 1
        value = (how_good - 0.5) * 2
    else:
        channel = 0
        value = (0.5 - how_good) * 2
    for c in range(3):
        if c == channel:
            padded[c, -padding:, :bar_width] = value
        else:
            padded[c, -padding:, :bar_width] = 0.0
    return padded
