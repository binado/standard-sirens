import h5py


def create_or_overwrite_dataset(file: h5py.File, dataset, data):
    """
    Overwrite existing dataset with new data for a hdf5 file.
    If such dataset does not exist, create one and populate it
    """
    try:
        # See https://stackoverflow.com/a/22925117
        old_data = file[dataset]
        old_data[...] = data
    except KeyError:
        file.create_dataset(dataset, data=data)


def write_to_file(file: str, prefix="", attrs: dict = None, **data):
    with h5py.File(file, "a") as f:
        for key, dataset in data.items():
            create_or_overwrite_dataset(f, prefix + key, data=dataset)

        if attrs is not None and isinstance(attrs, dict):
            f.attrs.update(**attrs)
