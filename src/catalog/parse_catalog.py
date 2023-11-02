import pandas as pd
import numpy as np
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "GLADE+reduced.txt")
chunksize = 100000
chunk_step_progress_message = 10

dtypes = {
    "GWGC_flag": str,
    "Hyperleda_flag": str,
    "2MASS_flag": str,
    "WISE_flag": str,
    "SDSS_flag": str,
    "Quasar_flag": str,
    "ra": np.float64,
    "dec": np.float64,
    "m_K": np.float64,
    "m_K_err": np.float64,
    "z_helio": np.float64,
    "z_cmb": np.float64,
    "peculiar_velocity_correction_flag": "Int64",
    "peculiar_velocity_err": np.float64,
    "z_helio_err": np.float64,
    "redshift_dl_flag": "Int64",
}

reader_args = dict(
    sep=" ",
    names=dtypes.keys(),
    dtype=dtypes,
    header=None,
    false_values=["null"],
    chunksize=chunksize,
)


def catalog_mask(df):
    return np.logical_and.reduce(
        [
            (df["z_cmb"].notnull()),
            (df["z_cmb"] >= 0),
            (df["Quasar_flag"] == "G"),
            ((df["redshift_dl_flag"] == 1) | (df["redshift_dl_flag"] == 3)),
            ~((df["peculiar_velocity_correction_flag"] == 0) & (df["z_cmb"] < 0.05)),
        ]
    )


def filter_chunk(df):
    # Remove galaxies with non-positive redshift
    df = df[df["z_cmb"].notnull() & (df["z_cmb"] >= 0)]
    # Remove quasars or clusters
    df = df[df["Quasar_flag"] == "G"]
    # Remove galaxies with no redshift or redshift calculated from dl
    df = df[(df["redshift_dl_flag"] == 1) | (df["redshift_dl_flag"] == 3)]
    # Remove close by galaxies without peculiar velocity corrections
    df = df[~((df["peculiar_velocity_correction_flag"] == 0) & (df["z_cmb"] < 0.05))]
    return df


if __name__ == "__main__":
    catalog = pd.DataFrame()
    with pd.read_csv(filename, **reader_args) as reader:
        for index, chunk in enumerate(reader):
            # print(f"chunk has {chunk.shape[0]} rows")
            catalog = pd.concat([catalog, filter_chunk(chunk)], ignore_index=True)
            if (index + 1) % chunk_step_progress_message == 0:
                print(f"Parsed chunk number {index + 1}")
                print(f"Parsed catalog has {catalog.shape[0]} rows")
                print(f"{chunksize * (index + 1) - catalog.shape[0]} rows filtered out")
                print("-------------------------------------------")

    print("catalog has a total of " + str(len(catalog)) + " objects.")
