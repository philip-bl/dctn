import click


@click.command()
@click.argument("n", type=int)
@click.option("--kek", type=str)
def main(n, kek):
    print(f"{n=}")
    print(f"{kek=}")
    1 / 0


if __name__ == "__main__":
    # main()
    main.main(("--kek", "keepo", "99"), standalone_mode=False)
