

from typing import Sequence
import pylibcudf as plc
import torch

from cudf_polars.triton.components.column_triton import TritonColumn


def _sort_one_key(
    indices: torch.Tensor,
    key: TritonColumn,
    order: plc.types.Order,
    null_order: plc.types.NullOrder,
) -> torch.Tensor:
    """
    Stably sort `indices` according to one key.

    Nulls are ordered independently of ascending/descending value order.
    """

    # Work with the rows currently being sorted.
    # values = key.obj[indices]
    valid = key.validity[indices]

    # Split valid and null rows. Because we use stable sorting below,
    # the relative order of equal values is preserved.
    valid_indices = indices[valid]
    null_indices = indices[~valid]

    if valid_indices.numel() > 1:
        valid_values = key.obj[valid_indices]

        descending = order == plc.types.Order.DESCENDING

        value_order = torch.argsort(
            valid_values,
            descending=descending,
            stable=True,
        )
        valid_indices = valid_indices[value_order]

    # Null rows have no value to sort on. Their relative order must be
    # preserved so that earlier sort keys remain the tie breaker.
    if null_order == plc.types.NullOrder.BEFORE:
        return torch.cat((valid_indices, null_indices))
    else:
        return torch.cat((null_indices, valid_indices))

def apply_sort(
    column: TritonColumn,
    indices: torch.Tensor,
    order: plc.types.Order,
    null_order: plc.types.NullOrder,
    stream
) -> TritonColumn:
    with torch.cuda.stream(stream):
        if isinstance(column.obj, plc.Column):
            indices_col = plc.Column.from_array(indices, stream=stream)
            (sorted_column,) = plc.copying.gather(
                plc.Table([column.obj]),
                indices_col,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            ).columns()
        else:
            sorted_column = column.obj[indices]
        return TritonColumn(
            sorted_column,
            dtype=column.dtype,
            validity=column.validity.to(indices.device)[indices],
            is_sorted=plc.types.Sorted.YES,
            order=order,
            null_order=null_order,
            name=column.name,
        )

def sort_indices(
    keys: Sequence[TritonColumn],
    order: Sequence[plc.types.Order],
    null_order: Sequence[plc.types.NullOrder],
    stream,
) -> torch.Tensor:
    """
    Return row indices that lexicographically sort `keys`.

    Parameters
    ----------
    keys
        Columns to sort by, in priority order. The first key is the
        primary sort key.
    order
        Sort direction for each key.
    null_order
        Null placement for each key.

    Returns
    -------
    torch.Tensor
        Integer tensor containing the permutation of row indices.

    Notes
    -----
    Sorting is stable. Keys are applied from the least significant key
    to the most significant key.
    """
    if not keys:
        return torch.empty(0, dtype=torch.int64)

    if len(order) != len(keys):
        raise ValueError(
            f"Expected {len(keys)} sort orders, got {len(order)}"
        )

    if len(null_order) != len(keys):
        raise ValueError(
            f"Expected {len(keys)} null orders, got {len(null_order)}"
        )

    n = keys[0].size
    device = keys[0].obj.device

    for i, key in enumerate(keys):
        if key.size != n:
            raise ValueError(
                f"All sort keys must have the same length: "
                f"key 0 has length {n}, key {i} has length {key.size}"
            )

        if key.obj.device != device:
            raise ValueError("All sort keys must be on the same device")

        if key.validity.device != device:
            raise ValueError("Column validity must be on the same device")

    # Stable lexicographic sort:
    #
    #     key[0] is primary
    #     key[1] is secondary
    #     ...
    #
    # Therefore process keys in reverse order.
    with torch.cuda.stream(stream):
        indices = torch.arange(n, device=device, dtype=torch.int64)

        for key, key_order, key_null_order in reversed(
            list(zip(keys, order, null_order))
        ):
            indices = _sort_one_key(
                indices,
                key,
                key_order,
                key_null_order,
            )

    return indices
