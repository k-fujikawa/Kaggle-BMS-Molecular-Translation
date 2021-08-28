import click
import pandas as pd

from nncomp_molecule.preprocessors import (
    normalize_inchi_batch,
    disable_rdlogger,
)


@click.command()
@click.argument("filename")
@click.option("--in-column", default="InChI")
@click.option("--out-column", default="normed_InChI")
def main(filename, in_column, out_column):
    disable_rdlogger()
    df = pd.read_csv(filename)
    df[out_column] = normalize_inchi_batch(df[in_column])
    df["is_valid"] = ~df[out_column].isna()
    df[out_column] = df[out_column].where(df.is_valid, df[in_column])
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
