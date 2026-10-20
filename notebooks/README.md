# cuDF Notebooks
## Intro
These notebooks provide examples of how to use cuDF.  These notebooks are designed to be self-contained with the `runtime` version of the [RAPIDS Docker Container](https://hub.docker.com/r/rapidsai/rapidsai/) and [RAPIDS Nightly Docker Containers](https://hub.docker.com/r/rapidsai/rapidsai-nightly) and can run on air-gapped systems.  You can quickly get this container using the install guide from the [RAPIDS.ai Getting Started page](https://rapids.ai/start.html#get-rapids)

## RAPIDS notebooks
Visit the main RAPIDS [notebooks](https://github.com/rapidsai/notebooks) repo for a listing of all notebooks across all RAPIDS libraries.

## Comparing TPC-H backends

The existing PDS-H (TPC-H-derived) benchmark definitions can be run across CPU
Polars, cuDF Polars, and the custom Triton backend with:

```bash
PYTHONPATH=python/cudf_polars python -m \
  cudf_polars.streaming.benchmarks.compare_backends \
  1,6,12 --path /path/to/pdsh/scale-1 --iterations 3
```

The command writes `cpu.jsonl`, one JSONL file per requested GPU backend, and
`summary.json` to `tpch_backend_results/` (or `--output-dir`). GPU results are
validated against the CPU result before speedups are reported. Use
`--backends cudf` or `--backends triton` to run one GPU backend.

PDS-H is the repository's derived benchmark suite and should not be
interpreted as an official TPC-H result.

The [TPC-H backend comparison notebook](./tpch_backend_comparison.ipynb)
automatically generates a small local dataset with DuckDB's `tpch` extension
when `PDSH_DATASET_PATH` is not set and no local dataset is found. For larger
runs, set `PDSH_DATASET_PATH` to an existing PDS-H dataset.
