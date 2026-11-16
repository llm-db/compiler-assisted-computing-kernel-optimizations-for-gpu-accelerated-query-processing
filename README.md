# GPU-Accelerated SQL Query Execution with Triton

This repository contains the implementation and evaluation code for a
thesis on executing SQL queries on GPUs with Triton.

## Abstract

SQL is widely used to manage and analyze structured data. As the volume of
digital information continues to grow, efficient query execution becomes
increasingly important. Graphics Processing Units (GPUs) have a highly
parallel architecture and can therefore offer significant potential for
accelerating data-intensive SQL operations.

This thesis investigates the performance of executing SQL queries on GPUs by
implementing a framework that uses Triton as its backend. The framework builds
upon the implementation of cuDF-Polars, but adopts a different backend for
executing query operations.

The two implementations are evaluated and compared across different
operators to identify their respective strengths and weaknesses. The results
show that selected Triton operators improve performance in 15 of 22 TPC-H
queries compared with their cuDF counterparts, while other operators perform
more slowly.

Based on these findings, this thesis proposes a hybrid execution model that
combines the most efficient operators from both implementations. This
approach achieves significant performance improvements in 6 TPC-H queries and
multiple minor improvements in others. The results highlight the potential of
combining different GPU execution backends to improve overall SQL query
performance.

## Requirements

The framework requires:

* An NVIDIA GPU with a compatible NVIDIA driver
* CUDA 12 or CUDA 13
* Python 3.11 or newer
* cuDF and cuDF-Polars from this repository
* Polars 1.35 or newer
* Triton
* The dependencies listed in the cuDF-Polars package

The CUDA version used by cuDF packages must match the installed CUDA
environment. For current driver and CUDA compatibility information, see the
[RAPIDS system requirements](https://docs.rapids.ai/install/#system-req).

## Installation

Install the Triton backend and the Python dependencies in a virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install triton
python -m pip install -e python/cudf_polars
```

For a complete cuDF development environment, follow the
[cuDF build and contribution guide](CONTRIBUTING.md#setting-up-your-build-environment).

## Running the backend comparison

The TPC-H-derived benchmark definitions can be run across CPU Polars, cuDF
Polars, and the custom Triton backend:

```bash
PYTHONPATH=python/cudf_polars python -m \
  cudf_polars.streaming.benchmarks.compare_backends \
  1,6,12 --path /path/to/pdsh/scale-1 --iterations 3
```

The command writes one JSONL file per backend and a `summary.json` file to
`tpch_backend_results/`. Use `--backends cudf` or `--backends triton` to run a
single GPU backend. GPU results are validated against the CPU result before
speedups are reported.

PDS-H is a TPC-H-derived benchmark suite and should not be interpreted as an
official TPC-H result. For more information about the benchmark notebook and
dataset configuration, see [notebooks/README.md](notebooks/README.md).

## Project structure

* `python/cudf_polars/cudf_polars/triton/` contains the Triton data structures,
  conversions, and query operators.
* `python/cudf_polars/cudf_polars/dsl/` integrates backend selection into the
  cuDF-Polars intermediate representation.
* `notebooks/` contains notebooks and benchmark documentation used for the
  evaluation.

## License

This project is based on [cuDF](https://github.com/rapidsai/cudf) and is
licensed under the [Apache 2.0 License](LICENSE).
