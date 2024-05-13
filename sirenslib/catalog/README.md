We use the [GLADE+](https://glade.elte.hu/) catalog, which combines data from six astronomical catalogues.

The raw data can be downloaded with the [`get_catalog`](../scripts/get_catalog.sh) script. Post-processing is implemented in the [`parse_catalog`](../scripts/parse_catalog.py) script. Run

```bash
python -m parse_catalog.py --help
```

for more details.

This [`notebook`](../visualize_catalog.ipynb) helps visualizing the parsed data.
