import pandas as pd
import numpy as np
from pathlib import Path
import click


def split_data(in_file: Path, img_dir: Path, out_dir: Path, fraction=0.8):

    df = pd.read_csv(str(in_file))
    print(df.shape)
    print(df.head)

    train, test = np.split(
        df.sample(frac=1, random_state=42), [int(fraction * len(df))]
    )

    for sub_df, dataset in zip([train, test], ["train", "test"]):

        (out_dir / dataset).mkdir(parents=True)
        sub_df.to_csv(out_dir / f"{dataset}/entries.csv")

        for img_file in sub_df["Image_ID"]:
            filename = f"{img_file}.JPG"
            (out_dir / f"{dataset}/{filename}").symlink_to(img_dir / filename)


@click.command()
@click.option("--in_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--img_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--out_dir", type=click.Path(file_okay=False))
@click.option("--fraction", default=0.8)
def cli(in_file, img_dir, out_dir, fraction):
    split_data(
        Path(in_file).absolute(),
        Path(img_dir).absolute(),
        Path(out_dir).absolute(),
        fraction,
    )


if __name__ == "__main__":
    cli()
