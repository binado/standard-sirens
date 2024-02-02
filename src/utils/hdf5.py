from os.path import join
import h5py

string_dtype = h5py.string_dtype(encoding="utf-8")


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


def write_to_file(file: str, data: dict, dtypes: dict, *args, prefix="", attrs: dict = None, **kwargs):
    default_kwargs = dict(maxshape=(None,))
    default_kwargs.update(**kwargs)
    with h5py.File(file, "a") as f:
        for key, dataset in data.items():
            # Make h5py correctly interpret data as string
            dtype = dtypes[key] if dtypes[key] != str else string_dtype
            create_or_overwrite_dataset(f, join(prefix, key), dataset, *args, dtype=dtype, **default_kwargs)

        if attrs is not None and isinstance(attrs, dict):
            f.attrs.update(**attrs)


# def traverse_node(file: h5py.File, node, data, prefix: str, *args, **kwargs):
#     if isinstance(node, dict):
#         for key, obj in node.items():
#             traverse_node(file, obj, data, join(prefix, key), *args, **kwargs)
#     elif isinstance(node, list):
#         for dataset in node:
#             create_or_overwrite_dataset(file, join(prefix, dataset), data[dataset], *args, **kwargs)
#     elif isinstance(node, str):
#         create_or_overwrite_dataset(file, join(prefix, node), data[node], *args, **kwargs)


# def write_from_scheme(file: str, node: dict, data, attrs: dict = None, **kwargs):
#     default_kwargs = dict(dtype="f8", maxshape=(None,))
#     default_kwargs.update(**kwargs)
#     prefix = ""
#     with h5py.File(file, "a") as f:
#         traverse_node(f, node, data, prefix, **kwargs)
#         if attrs is not None and isinstance(attrs, dict):
#             f.attrs.update(**attrs)
