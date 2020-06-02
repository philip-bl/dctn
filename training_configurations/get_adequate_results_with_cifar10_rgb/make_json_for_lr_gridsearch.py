import os

from make_plot_training_json_for_dir import main

main.main(
    (
        "/mnt/important/experiments/cifar10/ycbcr_plus_constant_channel",
        os.path.expanduser(
            "~/projects/dctn/small_experiments/plots/10_cifar10_ycbcr_const_channel_zeromeanscaling_one_eps_K=3/json.json"
        ),
        "-k",
        "lr",
        "-k",
        "reg_coeff",
    ),
    standalone_mode=False,
)
