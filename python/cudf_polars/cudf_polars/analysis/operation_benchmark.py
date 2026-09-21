# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused timing utilities for backend operation analysis.

The utilities deliberately accept already-prepared inputs and an operation
callable. Preparation, conversion, and result inspection therefore stay
outside the measured region.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import torch
from rmm.pylibrmm.stream import Stream

__all__ = [
    "FilterBenchmarkData",
    "JoinBenchmarkData",
    "TimingResult",
    "generate_filter_data",
    "generate_join_data",
    "time_cpu",
    "time_cuda",
]


@dataclasses.dataclass(frozen=True)
class FilterBenchmarkData:
    """Prepared host inputs for a filter benchmark."""

    frame: pl.DataFrame
    mask: pl.Series


@dataclasses.dataclass(frozen=True)
class JoinBenchmarkData:
    """Prepared host inputs for an inner join benchmark."""

    left: pl.DataFrame
    right: pl.DataFrame
    key: str


def generate_filter_data(
    num_rows: int,
    *,
    num_columns: int = 8,
    selectivity: float = 0.5,
    seed: int = 42,
) -> FilterBenchmarkData:
    """Generate deterministic host inputs for a filter benchmark."""
    if num_rows <= 0 or num_columns <= 0:
        raise ValueError("num_rows and num_columns must be positive")
    if not 0.0 <= selectivity <= 1.0:
        raise ValueError("selectivity must be between zero and one")

    rng = np.random.default_rng(seed)
    frame = pl.DataFrame(
        {
            f"c{index}": rng.standard_normal(num_rows).astype(np.float32)
            for index in range(num_columns)
        }
    )
    mask = pl.Series(
        "mask", rng.random(num_rows) < selectivity, dtype=pl.Boolean
    )
    return FilterBenchmarkData(frame, mask)


def generate_join_data(
    num_rows: int,
    *,
    num_columns: int = 4,
    right_rows: int | None = None,
    seed: int = 42,
) -> JoinBenchmarkData:
    """Generate deterministic host inputs for an inner join benchmark.

    The right side defaults to one quarter of the left side. Keys repeat on
    both sides, producing a non-empty but bounded join result.
    """
    if num_rows <= 0 or num_columns <= 0:
        raise ValueError("num_rows and num_columns must be positive")
    right_rows = max(1, num_rows // 4) if right_rows is None else right_rows
    if right_rows <= 0:
        raise ValueError("right_rows must be positive")

    rng = np.random.default_rng(seed)
    key = "key"
    left = pl.DataFrame(
        {
            key: np.arange(num_rows, dtype=np.int64) % right_rows,
            **{
                f"l{index}": rng.standard_normal(num_rows).astype(np.float32)
                for index in range(num_columns)
            },
        }
    )
    right = pl.DataFrame(
        {
            key: np.arange(right_rows, dtype=np.int64),
            **{
                f"r{index}": rng.standard_normal(right_rows).astype(np.float32)
                for index in range(num_columns)
            },
        }
    )
    return JoinBenchmarkData(left, right, key)


@dataclasses.dataclass(frozen=True)
class TimingResult:
    """Timing samples for one operation.

    ``milliseconds`` contains one sample per measured invocation. The return
    value of the operation is intentionally retained by the caller, allowing
    result validation to happen after timing.
    """

    milliseconds: tuple[float, ...]

    @property
    def mean_ms(self) -> float:
        """Return the arithmetic mean sample duration."""
        return sum(self.milliseconds) / len(self.milliseconds)

    @property
    def median_ms(self) -> float:
        """Return the median sample duration."""
        values = sorted(self.milliseconds)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2

    @property
    def min_ms(self) -> float:
        """Return the minimum sample duration."""
        return min(self.milliseconds)


def _check_iterations(warmups: int, repeats: int) -> None:
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")


def time_cpu(
    operation: Callable[[], Any], *, warmups: int = 3, repeats: int = 10
) -> TimingResult:
    """Time a synchronous CPU operation using a monotonic host clock."""
    _check_iterations(warmups, repeats)
    for _ in range(warmups):
        operation()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return TimingResult(tuple(samples))


def _torch_stream(stream: Stream) -> torch.cuda.ExternalStream:
    """Wrap an RMM stream without taking ownership of it."""
    version, stream_ptr = stream.__cuda_stream__()
    if version != 0:
        raise ValueError(f"Unsupported CUDA stream protocol version: {version}")
    return torch.cuda.ExternalStream(stream_ptr, device=torch.cuda.current_device())


def time_cuda(
    operation: Callable[[], Any],
    *,
    stream: Stream,
    warmups: int = 3,
    repeats: int = 10,
) -> TimingResult:
    """Time a CUDA operation queued on an RMM stream.

    CUDA events measure device execution only. The callable is expected to
    operate on inputs already associated with ``stream``; no conversion or
    result synchronization is performed inside the measured region.
    """
    _check_iterations(warmups, repeats)
    torch_stream = _torch_stream(stream)

    with torch.cuda.stream(torch_stream):
        for _ in range(warmups):
            result = operation()
            torch_stream.synchronize()
            del result

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(torch_stream):
            start.record(torch_stream)
            result = operation()
            end.record(torch_stream)
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result
    return TimingResult(tuple(samples))
