# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compare CPU Polars, cuDF Polars, and the Triton backend on PDS-H queries."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cudf_polars.streaming.benchmarks import pdsh


def _parse_queries(value: str) -> str:
    """Validate the query selector using the PDS-H parser."""
    parser = pdsh.build_parser(num_queries=22)
    parser.parse_args([value, "--frontend", "polars-cpu"])
    return value


def _run(
    *,
    backend: str | None,
    frontend: str,
    query: str,
    dataset_path: Path,
    output_path: Path,
    iterations: int,
    suffix: str,
    validation: str | None,
    executor: str,
    backend_executor: str | None,
    warmup: bool,
) -> dict[str, Any]:
    """Run one backend in a fresh process and return its JSON result."""
    command = [
        sys.executable,
        "-m",
        "cudf_polars.streaming.benchmarks.pdsh",
        query,
        "--path",
        str(dataset_path),
        "--suffix",
        suffix,
        "--frontend",
        frontend,
        "--iterations",
        str(iterations + int(warmup)),
        "--output",
        str(output_path),
        "--no-summarize",
        "--no-print-results",
        "--no-print-plans",
        "--executor",
        backend_executor or executor,
    ]
    if validation is not None:
        command.extend(["--validate-against", validation])

    output_path.unlink(missing_ok=True)
    environment = os.environ.copy()
    if backend is None:
        environment.pop("CUDF_POLARS__BACKEND", None)
    else:
        environment["CUDF_POLARS__BACKEND"] = backend

    completed = subprocess.run(
        command,
        check=False,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend or 'cpu'} benchmark failed with exit code "
            f"{completed.returncode}; see the benchmark output above"
        )

    with output_path.open() as output:
        records = [json.loads(line) for line in output if line.strip()]
    if len(records) != 1:
        raise RuntimeError(
            f"Expected one result record in {output_path}, found {len(records)}"
        )
    return records[0]


def _timings(
    record: dict[str, Any], *, skip_warmup: bool = False
) -> dict[str, list[float]]:
    """Return successful timings grouped by query."""
    timings: dict[str, list[float]] = {}
    for query, records in record["records"].items():
        values = [
            float(item["duration"])
            for item in records[int(skip_warmup) :]
            if item.get("status") == "success"
        ]
        if values:
            timings[query] = values
    return timings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run identical PDS-H queries with CPU Polars, cuDF Polars, and Triton. "
            "GPU results are validated against CPU Polars."
        )
    )
    parser.add_argument(
        "query", type=_parse_queries, help="Query number, range, list, or all."
    )
    parser.add_argument(
        "--path", type=Path, required=True, help="PDS-H parquet dataset directory."
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Timed iterations per query."
    )
    parser.add_argument("--suffix", default=".parquet", help="Input file suffix.")
    parser.add_argument(
        "--backends",
        default="cudf,triton",
        help="Comma-separated GPU backends to run (default: cudf,triton).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tpch_backend_results"),
        help="Directory for per-backend JSONL and summary JSON.",
    )
    parser.add_argument(
        "--executor",
        choices=["in-memory", "streaming"],
        default="in-memory",
        help="GPU executor for the cuDF backend (default: in-memory).",
    )
    parser.add_argument(
        "--triton-executor",
        choices=["in-memory", "streaming"],
        default=None,
        help="Override the executor for the Triton backend.",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one untimed-equivalent warm-up iteration per backend.",
    )
    return parser


def main() -> None:
    """Run the comparison and write a machine-readable summary."""
    args = _build_parser().parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    backends = [
        backend.strip() for backend in args.backends.split(",") if backend.strip()
    ]
    invalid = set(backends) - {"cudf", "triton"}
    if invalid:
        raise ValueError(f"Unsupported backends: {sorted(invalid)}")
    if not backends:
        raise ValueError("--backends must contain at least one backend")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    cpu_output = args.output_dir / "cpu.jsonl"
    results["cpu"] = _run(
        backend=None,
        frontend="polars-cpu",
        query=args.query,
        dataset_path=args.path,
        output_path=cpu_output,
        iterations=args.iterations,
        suffix=args.suffix,
        validation=None,
        executor="in-memory",
        backend_executor=None,
        warmup=args.warmup,
    )
    cpu_timings = _timings(results["cpu"], skip_warmup=args.warmup)

    for backend in backends:
        results[backend] = _run(
            backend=backend,
            frontend="in-memory",
            query=args.query,
            dataset_path=args.path,
            output_path=args.output_dir / f"{backend}.jsonl",
            iterations=args.iterations,
            suffix=args.suffix,
            validation="polars-cpu",
            executor=args.executor,
            backend_executor=(
                args.triton_executor if backend == "triton" else None
            ),
            warmup=args.warmup,
        )

    summary: dict[str, Any] = {"timings_seconds": {}, "speedup_vs_cpu": {}}
    for backend, record in results.items():
        timings = _timings(record, skip_warmup=args.warmup)
        summary["timings_seconds"][backend] = {
            query: {"samples": values, "mean": sum(values) / len(values)}
            for query, values in timings.items()
        }
        if backend != "cpu":
            summary["speedup_vs_cpu"][backend] = {
                query: (sum(cpu_timings[query]) / len(cpu_timings[query]))
                / (sum(values) / len(values))
                for query, values in timings.items()
                if query in cpu_timings
            }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    for backend, query_timings in summary["timings_seconds"].items():
        for query, timing in query_timings.items():
            speedup = summary["speedup_vs_cpu"].get(backend, {}).get(query)
            suffix = f", speedup_vs_cpu={speedup:.3f}x" if speedup is not None else ""
            print(f"{backend} q{query}: {timing['mean']:.6f}s{suffix}")
    print(f"Wrote comparison to {summary_path}")


if __name__ == "__main__":
    main()
