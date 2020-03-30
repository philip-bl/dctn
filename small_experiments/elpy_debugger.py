import click
import os
import sys
import torch

def fizzbuzz(n: int) -> None:
  for i in range(1, n+1):
    print(str(i) + ": ", end="")
    if i % 3 == 0:
      print("fizz", end="")
    if i % 5 == 0:
      print("buzz", end="")
    print()


@click.command()
@click.argument("n", type=int)
def main(n):
  print(f"{torch.cuda.device_count()=}")
  fizzbuzz(n)


def fail():
  return 3 + "a"

if __name__ == "__main__":
  if sys.argv == [""]:
    print("populating argv manually")
    sys.argv.append("19")
    print(f"{sys.argv=}")
  else:
    print(f"{sys.argv=}")
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  print(f"{os.environ=}")
  main()
