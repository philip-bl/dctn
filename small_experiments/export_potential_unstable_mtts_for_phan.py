import torch
import os
from scipy.io import savemat

sd_85_percent_accuracy_on_mnist = torch.load(
    "/mnt/important/experiments/dctn_mnist/2020-02-25T21-49-26/dctnmnistmodel_UTC2020-02-25T18:49:29_model_34721_cross_entropy_loss=0.4238284.pth",
    "cpu",
)

sd_bigger_min_random_eye = torch.load(
    "/mnt/important/experiments/dctn_mnist/2020-03-14T11-44-26/dctnmnistmodel_UTC2020-03-14T08:44:29_model_505_cross_entropy_loss=2.193019.pth",
    "cpu",
)

model_85_percent_accuracy_on_mnist = [
    [
        [
            sd_85_percent_accuracy_on_mnist[f"conv_sbses.0.strings.0.cores.{i}"].numpy()
            for i in range(9)
        ],
        [
            sd_85_percent_accuracy_on_mnist[f"conv_sbses.0.strings.1.cores.{i}"].numpy()
            for i in range(9)
        ],
    ],
    [
        [
            sd_85_percent_accuracy_on_mnist[f"conv_sbses.1.strings.0.cores.{i}"].numpy()
            for i in range(9)
        ]
    ],
]

model_bigger_min_random_eye = [
    [
        [
            sd_bigger_min_random_eye[f"conv_sbses.0.strings.0.cores.{i}"].numpy()
            for i in range(9)
        ],
        [
            sd_bigger_min_random_eye[f"conv_sbses.0.strings.1.cores.{i}"].numpy()
            for i in range(9)
        ],
    ],
    [
        [
            sd_bigger_min_random_eye[f"conv_sbses.1.strings.0.cores.{i}"].numpy()
            for i in range(9)
        ],
        [
            sd_bigger_min_random_eye[f"conv_sbses.1.strings.1.cores.{i}"].numpy()
            for i in range(9)
        ],
    ],
    [
        [
            sd_bigger_min_random_eye[f"conv_sbses.2.strings.0.cores.{i}"].numpy()
            for i in range(9)
        ],
        [
            sd_bigger_min_random_eye[f"conv_sbses.2.strings.1.cores.{i}"].numpy()
            for i in range(9)
        ],
    ],
    [
        [
            sd_bigger_min_random_eye[f"conv_sbses.3.strings.0.cores.{i}"].numpy()
            for i in range(9)
        ]
    ],
]


dir = "/mnt/important/experiments/export_of_mtts_for_phan/model_85_percent_accuracy_on_mnist"
for i, layer in enumerate(model_85_percent_accuracy_on_mnist):
    layer_dir = os.path.join(dir, f"layer_{i}")
    os.mkdir(layer_dir)
    for j, snake_mtt_sbs in enumerate(layer):
        savemat(
            os.path.join(layer_dir, f"snake_mtt_sbs_{j}.mat"),
            {f"core_{k}": snake_mtt_sbs[k] for k in range(9)},
        )

dir = "/mnt/important/experiments/export_of_mtts_for_phan/model_bigger_min_random_eye"
for i, layer in enumerate(model_bigger_min_random_eye):
    layer_dir = os.path.join(dir, f"layer_{i}")
    os.mkdir(layer_dir)
    for j, snake_mtt_sbs in enumerate(layer):
        savemat(
            os.path.join(layer_dir, f"snake_mtt_sbs_{j}.mat"),
            {f"core_{k}": snake_mtt_sbs[k] for k in range(9)},
        )
