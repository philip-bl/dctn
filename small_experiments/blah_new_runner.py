from new_runner import main

main.main(
  (
    "--experiments-dir", "/tmp/new_runner_test/",
    "--ds-type", "fashionmnist",
    "--ds-path", "/mnt/hdd_1tb/datasets/fashionmnist",
    "--kernel-size", "4",
    "--out-size", "4",
    "--batch-size", "128",
    # "--load-model-state", "/mnt/important/experiments/eps_plus_linear_fashionmnist/out_size=4_model/model_sd.pth"
  ),
  standalone_mode=False)



# STOP HERE, run interact, do the lines below
from dctn.align import align
from dctn.rank_one_tensor import RankOneTensorsBatch
train_ds = train_dl.dataset
x = train_dl.dataset.x.double()
x_windows = torch.cat(
  tuple(
    torch.stack(tuple(align(x_slice, kwargs["kernel_size"])), dim=0)
    for x_slice in x[:, 22000:44000].split(kwargs["batch_size"], dim=1)),
  dim=1)
x_windows_r1t = RankOneTensorsBatch(x_windows, factors_dim=0, coordinates_dim=4)
assert x_windows_r1t.ncoordinates == 2**16
print(f"{x_windows_r1t.mean_over_batch()=}")
print(f"{x_windows_r1t.std_over_batch()=}")
from dctn.utils import transform_dataset
x_transformed = transform_dataset(model[0].double(), x.to(dev))
print(f"{x_transformed.mean()=}")
print(f"{x_transformed.std()=}")
