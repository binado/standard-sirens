We use the [GLADE+](https://glade.elte.hu/) catalog, which combines data from six astronomical catalogues.

The raw data can be downloaded with the [`get_catalog`](../scripts/get_catalog.sh) script. Post-processing is implemented in the [`parse_catalog`](../scripts/parse_catalog.py) script. After installing the package, run

```bash
sirenslib-parse_catalog --help
```

for more details.

This [`notebook`](../../notebooks/visualize_catalog.ipynb) helps visualizing the parsed data.
