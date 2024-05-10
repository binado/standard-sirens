from os.path import join
import pickle
import h5py

string_dtype = h5py.string_dtype(encoding="utf-8")


def pickle_write(filename, obj):
    with open(filename, "r+b") as f:
        pickle.dump(obj, f)


def pickle_read(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def create_or_overwrite_dataset(file: h5py.File, dataset, data, *args, **kwargs):
    """
    Overwrite existing dataset with new data for a hdf5 file.
    If such dataset does not exist, create one and populate it
    """
    try:
        # See https://stackoverflow.com/a/22925117
        del file[dataset]
    except KeyError:
        pass
    finally:
        file.create_dataset(dataset, *args, data=data, **kwargs)


def write_to_hdf5(file: str, data: dict, dtypes: dict, *args, prefix="", attrs: dict = None, **kwargs):
    with h5py.File(file, "a") as f:
        for key, dataset in data.items():
            # Make h5py correctly interpret data as string
            dtype = dtypes[key] if dtypes[key] != str else string_dtype
            create_or_overwrite_dataset(f, join(prefix, key), dataset, *args, dtype=dtype, **kwargs)

        if attrs is not None and isinstance(attrs, dict):
            f.attrs.update(**attrs)


def set_dataset_attrs(file: str, dataset: str = None, **attrs):
    with h5py.File(file, "a") as f:
        group = f[dataset] if dataset is not None else f
        group.attrs.update(**attrs)


def get_dataset_attrs(file: str, dataset: str = None):
    with h5py.File(file, "r") as f:
        group = f[dataset] if dataset is not None else f
        return group.attrs
