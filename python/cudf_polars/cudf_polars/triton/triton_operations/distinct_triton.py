


from typing import Sequence

import torch
import pylibcudf as plc
from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame


def _distinct_group_boundaries(
    keys: list[TritonColumn],
    indices: torch.Tensor,
) -> torch.Tensor:
    n = indices.numel()

    if n == 0:
        return torch.empty(
            0,
            device=indices.device,
            dtype=torch.bool,
        )

    if n == 1:
        return torch.ones(
            1,
            device=indices.device,
            dtype=torch.bool,
        )

    lhs = indices[:-1]
    rhs = indices[1:]

    same = torch.ones(
        n - 1,
        device=indices.device,
        dtype=torch.bool,
    )

    for key in keys:
        same &= _rows_equal(key, lhs, rhs)

    return torch.cat(
        (
            torch.ones(
                1,
                device=indices.device,
                dtype=torch.bool,
            ),
            ~same,
        )
    )

def _rows_equal(
    key: TritonColumn,
    lhs_indices: torch.Tensor,
    rhs_indices: torch.Tensor,
) -> torch.Tensor:
    lhs_valid = torch.index_select(
        key.validity, 0, lhs_indices
    )
    rhs_valid = torch.index_select(
        key.validity, 0, rhs_indices
    )

    both_null = ~lhs_valid & ~rhs_valid
    both_valid = lhs_valid & rhs_valid

    equal = both_null.clone()

    equal[both_valid] = (
        torch.index_select(key.obj, 0, lhs_indices[both_valid])
        == torch.index_select(key.obj, 0, rhs_indices[both_valid])
    )

    return equal

def distinct_triton_indices(
    key_columns: Sequence[TritonColumn],
    indices: torch.Tensor,
    keep: plc.stream_compaction.DuplicateKeepOption,
) -> torch.Tensor:
    """Evaluate and return a dataframe."""

    # ------------------------------------------------------------------
    # Find the first row of every distinct group.
    # ------------------------------------------------------------------

    group_start = _distinct_group_boundaries(
        key_columns,
        indices,
    )

    starts = torch.nonzero(
        group_start,
        as_tuple=True,
    )[0]

    # Group ends are exclusive.
    ends = torch.cat(
        (
            starts[1:],
            torch.tensor(
                [indices.numel()],
                device=indices.device,
                dtype=starts.dtype,
            ),
        )
    )

    # ------------------------------------------------------------------
    # Select which rows to retain.
    # ------------------------------------------------------------------

    if keep == plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST:
        selected = indices[starts]

    elif keep == plc.stream_compaction.DuplicateKeepOption.KEEP_LAST:
        selected = indices[ends - 1]

    elif keep == plc.stream_compaction.DuplicateKeepOption.KEEP_ANY:
        # Because `indices` is stable, this is equivalent to FIRST for
        # this implementation.
        selected = indices[starts]

    elif keep == plc.stream_compaction.DuplicateKeepOption.KEEP_NONE:
        group_sizes = ends - starts
        unique_groups = group_sizes == 1

        selected = indices[starts[unique_groups]]

    else:
        raise ValueError(f"Unsupported keep option: {keep}")

    # ------------------------------------------------------------------
    # Apply the selected rows to every column.
    # ------------------------------------------------------------------
    return selected
