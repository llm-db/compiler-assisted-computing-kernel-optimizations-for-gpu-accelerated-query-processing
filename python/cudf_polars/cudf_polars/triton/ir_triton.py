
from ast import BinOp
import itertools
import sys
import time
from typing import Any, Sequence

import torch
import pyarrow as pa
import polars as pl

from cudf_polars.containers import DataType
from cudf_polars.triton.triton_conversion import (
    dtype_cudf_to_triton,
    dtype_triton_to_cudf,
)
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame
from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.triton.utils_triton import broadcast_triton, concatonate_dataframes, stream_ordered_after
from cudf_polars.triton.components.base_triton import (
    TritonCol,
    TritonExpr,
    TritonNamedExpr,
)
from cudf_polars.typing import Schema, Slice as Zlice
from cudf_polars.triton.triton_operations.join_triton import join_coalesce, join_gather, join_gather_column, join_pairs
from cudf_polars.triton.components.binOp_triton import TritonBinOp
from cudf_polars.triton.components.agg_triton import TritonAgg, TritonLen
import pylibcudf as plc
import cupy as cp

from cudf_polars.triton.triton_operations.sort_triton import apply_sort, sort_indices
from cudf_polars.triton.triton_operations.distinct_triton import distinct_triton_indices


def join_triton(
    left: TritonDataFrame,
    right: TritonDataFrame,
    left_on: tuple[TritonNamedExpr, ...],
    right_on: tuple[TritonNamedExpr, ...],
    options: tuple[str, bool, Zlice | None, str, bool, str],
    context
) -> TritonDataFrame:
    """Evaluate a regular join using Triton tensors."""

    with context.stream_ordered_after(left, right) as stream:
        how, nulls_equal, zlice, suffix, coalesce, maintain_order = options
        left_keys = list(broadcast_triton(*(key.evaluate(left) for key in left_on), target_length=left.num_rows, stream=left.stream))
        right_keys = list(broadcast_triton(*(key.evaluate(right) for key in right_on), target_length=right.num_rows, stream=right.stream))
        left_indices, right_indices = join_pairs(
            left_keys,
            right_keys,
            how,
            nulls_equal,
            device=left.stream.device,
            n_left=left.num_rows,
            n_right=right.num_rows,
        )

        if how in {"Semi", "Anti"}:
            columns = [
                join_gather_column(column, left_indices).rename(column.name)
                for column in left.columns
            ]
            return TritonDataFrame(columns, stream).slice(zlice)

        left_columns, right_columns = join_gather(left, right, options, left_indices, right_indices)
        if coalesce:
            left_columns, right_columns = join_coalesce(left, right, right_on, left_on, options, left_columns, right_columns)
        ret_df = TritonDataFrame([*left_columns, *right_columns], stream).slice(zlice)
        return ret_df


def empty_triton(
    schema, *, context
) -> TritonDataFrame:
    stream = context.get_cuda_stream()
    return TritonDataFrame(
        [
            TritonColumn(
                [],
                dtype=dtype,
                name=name,
            )
            for name, dtype in schema.items()
        ],
        cudf_stream=stream,
    )
    
    

def hconcat_triton(
    *dfs: TritonDataFrame,
    schema,
    should_broadcast: bool,
    strict: bool,
    context
) -> TritonDataFrame:

    with context.stream_ordered_after(*dfs) as stream:
        # Special should_broadcast case.
        # Used to recombine decomposed expressions
        if should_broadcast:
            bcols = broadcast_triton(
                *itertools.chain.from_iterable(df.columns for df in dfs), stream=stream
            )
            by_name = {c.name: c for c in bcols}
            ordered = [by_name[name] for name in schema]
            result = TritonDataFrame(ordered, cudf_stream=stream)
        else:
            max_rows = max(df.num_rows for df in dfs)
            if strict and any(df.num_rows != max_rows for df in dfs):
                heights = [df.num_rows for df in dfs]
                msg = (
                    f"cannot concat DataFrames horizontally"
                    f" with strict=True: height mismatch {heights}"
                )
                raise ValueError(msg)
            # Horizontal concatenation extends shorter tables with nulls
            result = TritonDataFrame(
                itertools.chain.from_iterable(
                    df.columns
                    for df in (
                        df
                        if df.num_rows == max_rows
                        else TritonDataFrame.from_table(
                            cls._extend_with_nulls(
                                df.table,
                                nrows=max_rows - df.num_rows,
                                stream=stream,
                            ),
                            df.column_names,
                            df.dtypes,
                            stream=stream,
                        )
                        for df in dfs
                    )
                ),
                stream=stream,
            )

        return result
        
def hstack_triton(
    df: TritonDataFrame,
    exprs: tuple[TritonNamedExpr, ...],
    should_broadcast: bool,
) -> TritonDataFrame:
    """Evaluate and return a dataframe."""
    columns = [c.evaluate(df) for c in exprs]
    if should_broadcast:
        columns = broadcast_triton(
            *columns,
            target_length=df.num_rows if df.num_columns != 0 else None,
            stream=df.stream
        )
    else:  # pragma: no cover; streaming rewrites HStack(False)
        # Polars ensures this is true, but let's make sure nothing
        # went wrong. In this case, the parent node is
        # guaranteed to be a Select which will take care of making
        # sure that everything is the same length. The result
        # table that might have mismatching column lengths will
        # never be turned into a pylibcudf Table with all columns
        # by the Select, which is why this is safe.
        assert all(e.name.startswith("__POLARS_CSER_0x") for e in exprs)
    return df.with_columns(columns, stream=df.stream)


def distinct_triton(
    df: TritonDataFrame,
    keep: plc.stream_compaction.DuplicateKeepOption,
    rowSubset: frozenset[str] | None,
    zlice: Zlice | None,
) -> TritonDataFrame:
    """Evaluate and return a dataframe."""
    n = df.num_rows

    if n == 0:
        return df.slice(zlice)

    if rowSubset is None:
        key_columns = list(df.columns)
    else:
        key_columns = [
            df.column_map[name]
            for name in rowSubset
        ]

    keys_sorted = all(
        column.is_sorted == plc.types.Sorted.YES
        for column in key_columns
    )

    if keys_sorted:
        indices = torch.arange(
            n,
            device=df.columns[0].obj.device,
            dtype=torch.int64,
        )
    else:
        # Distinct doesn't specify output order, so sorting by the
        # distinct keys is a valid implementation strategy.
        #
        # Use ascending / null-before for all keys. Stable sorting also
        # means that equal keys retain their original row order.
        indices = sort_indices(
            key_columns,
            order=[
                plc.types.Order.ASCENDING
                for _ in key_columns
            ],
            null_order=[
                plc.types.NullOrder.BEFORE
                for _ in key_columns
            ],
            stream=df.stream
        )

    selected = distinct_triton_indices(key_columns, indices, keep)
    new_columns = [
        TritonColumn(
            old.obj[selected],
            dtype=old.dtype,
            validity=old.validity[selected],
            name=old.name,
            is_sorted=(
                plc.types.Sorted.YES
                if keys_sorted
                else plc.types.Sorted.NO
            ),
            order=old.order,
            null_order=old.null_order,
        )
        for old in df.columns
    ]


    result = TritonDataFrame(
        new_columns,
        cudf_stream=df.cudf_stream,
    )

    if keys_sorted:
        result = result.sorted_like(df)

    return result.slice(zlice)

def sort_triton(
    df: TritonDataFrame,
    by: Sequence[TritonNamedExpr],
    order: Sequence[plc.types.Order],
    null_order: Sequence[plc.types.NullOrder],
    stable: bool,  # noqa: FBT001
    zlice: Zlice | None,
) -> TritonDataFrame:
    """Evaluate and return a dataframe."""

    sort_keys = broadcast_triton(
        *(k.evaluate(df) for k in by), target_length=df.num_rows, stream=df.stream
    )
    indices = sort_indices(sort_keys, order,null_order, df.stream)
    result = TritonDataFrame(
        [apply_sort(col, indices, order, null_order, df.stream) for col in df.columns],
        df.cudf_stream
    )
    return result.slice(zlice)


def conditional_join_triton(
    left: TritonDataFrame,
    right: TritonDataFrame,
    predicate: TritonBinOp,
    options: tuple[None, bool, Zlice | None, str, bool, str],
) -> TritonDataFrame:
    """Evaluate a conditional inner join using a pairwise Torch mask."""
    _, nulls_equal, zlice, suffix, coalesce, maintain_order = options
    if not isinstance(predicate, TritonBinOp):
        raise TypeError("BinOp expected")
    matches_col = predicate.do_evaluate_dual(
        left, 
        right
    )

    # n_left = left.num_rows
    # n_right = right.num_rows
    # matches = torch.ones((n_left, n_right), device=left.columns[0].obj.device, dtype=torch.bool)

    # TODO: Not good, should be 2D all in triton
    
        # new_col = TritonColumn(
        #     torch.tensor([col.obj[r]], device=col.obj.device), 
        #     dtype=col.dtype, 
        #     validity=torch.tensor([col.validity[r]], device=col.obj.device)
        # )

    left_indices, right_indices = torch.nonzero(matches_col.obj, as_tuple=True)

    left_columns, right_columns = join_gather(left, right, options, left_indices, right_indices)
    # if coalesce:
    #     left_columns, right_columns = join_coalesce(left, right, right_on, left_on, options, left_columns, right_columns)
    df = TritonDataFrame([*left_columns, *right_columns], left.stream).slice(zlice)
    return df

    # predCol = predicate.do_evaluate_dual(left, right)
    # # values, validity = join_conditional_operand(predicate, left, right)
    # left_indices, right_indices = torch.nonzero(predCol.obj & predCol.validity, as_tuple=True)
    # left_columns = [
    #     join_gather_column(column, left_indices).rename(column.name)
    #     for column in left.columns
    # ]
    # right_columns = [
    #     join_gather_column(column, right_indices).rename(
    #         column.name
    #         if column.name not in left.column_names_set
    #         else f"{column.name}{suffix}"
    #     )
    #     for column in right.columns
    # ]
    # return TritonDataFrame([*left_columns, *right_columns], left.stream).slice(zlice)


def filter_triton(df: TritonDataFrame, mask: TritonExpr) -> TritonDataFrame:
    ev = mask.evaluate(df)
    (mask,) = broadcast_triton(ev, target_length=df.num_rows, stream=df.stream)
    return df.filter(mask)

def union_triton(*dfs: TritonDataFrame, zlice: Zlice, context) -> TritonDataFrame:
    return concatonate_dataframes(*dfs, context=context).slice(zlice)

def select_triton(df: TritonDataFrame, exprs: tuple[TritonNamedExpr, ...], should_broadcast: bool) -> TritonDataFrame:
    columns = [e.evaluate(df) for e in exprs]
    if should_broadcast:
        columns = broadcast_triton(*columns, stream=df.stream)
    return TritonDataFrame(columns, df.cudf_stream)

def reduce_triton(df: TritonDataFrame, exprs: tuple[TritonNamedExpr, ...]) -> TritonDataFrame:
    columns = broadcast_triton(*(expr.evaluate(df) for expr in exprs), stream=df.stream)
    columns = [column.rename(expr.name) for column, expr in zip(columns, exprs, strict=True)]
    return TritonDataFrame(columns, df.cudf_stream)

def rolling_triton(
    df: TritonDataFrame,
    index: TritonNamedExpr,
    keys: tuple[TritonNamedExpr, ...],
    aggs: tuple[TritonNamedExpr, ...],
    preceding: int,
    following: int,
    closed: str,
    zlice: Zlice | None,
) -> TritonDataFrame:
    index_column = index.evaluate(df)
    key_columns = [key.evaluate(df) for key in keys]
    index_values = index_column.obj.to(torch.int64)
    index_values_host = index_values.cpu().tolist()
    if not all(
        index_column.validity.cpu().tolist()
    ):
        raise RuntimeError(
            f"Index column '{index.name}' in rolling may not contain nulls"
        )
    if not keys and index_values_host != sorted(index_values_host):
        raise RuntimeError(
            f"Index column '{index.name}' in rolling is not sorted, please sort first"
        )
    key_values_host = [key.obj.cpu().tolist() for key in key_columns]
    key_validities_host = [
        key.validity.cpu().tolist() # TODO: WHY CPU????
        for key in key_columns
    ]
    group_labels = [
        tuple(
            None if validities is not None and not validities[row] else values[row]
            for values, validities in zip(
                key_values_host, key_validities_host, strict=True
            )
        )
        for row in range(len(index_values_host))
    ]
    if keys:
        for previous, current in zip(
            zip(group_labels[:-1], index_values_host[:-1]),
            zip(group_labels[1:], index_values_host[1:]),
            strict=True,
        ):
            if current[0] == previous[0] and current[1] < previous[1]:
                raise RuntimeError("Input for grouped rolling is not sorted")
    result_columns = [
        TritonColumn(
            key_column.obj,
            validity=key_column.validity,
            dtype=key_column.dtype,
            name=key.name,
        )
        for key, key_column in zip(keys, key_columns, strict=True)
    ]
    result_columns.append(
        TritonColumn(
            index_column.obj,
            validity=index_column.validity,
            dtype=index_column.dtype,
            name=index.name,
        )
    )

    values = {agg.name: [] for agg in aggs}
    validities = {agg.name: [] for agg in aggs}
    for row, value in enumerate(index_values_host):
        lower = value + preceding
        upper = value + preceding + following
        window_indices_host = []
        for candidate, candidate_value in enumerate(index_values_host):
            in_lower = (
                candidate_value >= lower
                if closed in {"both", "left"}
                else candidate_value > lower
            )
            in_upper = (
                candidate_value <= upper
                if closed in {"both", "right"}
                else candidate_value < upper
            )
            same_group = group_labels[candidate] == group_labels[row]
            if in_lower and in_upper and same_group:
                window_indices_host.append(candidate)
        window_indices = torch.tensor(
            window_indices_host, device=index_values.device, dtype=torch.long
        )
        window_columns = [
            TritonColumn(
                column.obj.index_select(0, window_indices),
                dtype=column.dtype,
                validity=column.validity.index_select(0, window_indices),
                name=column.name,
            )
            for column in df.columns
        ]
        window_df = TritonDataFrame(window_columns, df.cudf_stream)
        for agg in aggs:
            result = agg.evaluate(window_df)
            values[agg.name].append(result.obj[0])
            validities[agg.name].append(
                result.validity[0]
            )

    for agg in aggs:
        result_columns.append(
            TritonColumn(
                torch.stack(values[agg.name]).to(agg.value.dtype),
                validity=torch.stack(validities[agg.name]),
                dtype=agg.value.dtype,
                name=agg.name,
            )
        )

    return TritonDataFrame(result_columns, df.cudf_stream).slice(zlice)


def _triton_column_to_plc(column: TritonColumn, stream) -> plc.Column:
    """Convert a Triton column to a libcudf column without a host round trip."""
    if isinstance(column.obj, plc.Column):
        result = column.obj
    else:
        result = plc.Column.from_array(cp.asarray(column.obj), stream=stream)
    if column.is_all_valid:
        return result
    validity = plc.Column.from_array(cp.asarray(column.validity), stream=stream)
    mask, null_count = plc.transform.bools_to_mask(validity, stream=stream)
    return result.with_mask(mask, null_count)


def _plc_column_to_triton(
    column: plc.Column, dtype, name: str, stream
) -> TritonColumn:
    """Wrap a libcudf result while retaining its null mask as a device tensor."""
    if column.null_mask() is None:
        validity = None
    else:
        validity_column = plc.transform.mask_to_bools(
            int(column.null_mask().ptr),
            column.offset(),
            column.offset() + column.size(),
            stream=stream,
        )
        validity = torch.from_dlpack(
            plc.interop.to_dlpack(plc.Table([validity_column]), stream=stream)
        ).to(torch.bool)
    if isinstance(dtype, torch.dtype):
        values_column = column.with_mask(None, 0)
        values = torch.from_dlpack(
            plc.interop.to_dlpack(plc.Table([values_column]), stream=stream)
        )
        return TritonColumn(
            values,
            dtype=dtype,
            validity=validity,
            name=name,
        )
    return TritonColumn(column, dtype=dtype, validity=validity, name=name)


def groupby_triton(
    df: TritonDataFrame,
    keys: tuple[TritonNamedExpr, ...],
    aggs: tuple[TritonNamedExpr, ...],
    maintain_order: bool,
    zlice: Zlice | None,
) -> TritonDataFrame:
    evaluation_df = df
    if (
        df.cudfFallback is not None
        and df.cudfFallback.num_rows == df.num_rows
        and df.cudfFallback.column_names == df.column_names
    ):
        evaluation_df = TritonDataFrame(
            [
                TritonColumn(
                    column.obj,
                    dtype=dtype_cudf_to_triton(column.dtype),
                    name=column.name,
                )
                for column in df.cudfFallback.columns
            ],
            df.cudf_stream,
            num_rows=df._num_rows_override,
        )
    key_columns = broadcast_triton(
        *(key.evaluate(evaluation_df) for key in keys),
        target_length=df.num_rows,
        stream=df.stream,
    )

    # Keep the grouped operation on the device.  The previous implementation
    # materialized every key on the CPU and then evaluated each group
    # separately, which made the cost proportional to the number of rows and
    # groups in Python.
    if not maintain_order and all(
        isinstance(agg.value, (TritonAgg, TritonLen)) for agg in aggs
    ):
        try:
            converted_keys = [
                _triton_column_to_plc(column, df.cudf_stream)
                for column in key_columns
            ]
            grouper = plc.groupby.GroupBy(
                plc.Table(converted_keys),
                null_handling=plc.types.NullPolicy.INCLUDE,
                keys_are_sorted=(
                    plc.types.Sorted.YES
                    if all(
                        key.is_sorted == plc.types.Sorted.YES
                        for key in key_columns
                    )
                    else plc.types.Sorted.NO
                ),
                column_order=[key.order for key in key_columns],
                null_precedence=[key.null_order for key in key_columns],
            )
            requests = []
            for agg in aggs:
                value = agg.value
                if isinstance(value, TritonLen):
                    child = evaluation_df.columns[0]
                    aggregation = plc.aggregation.count(
                        null_handling=plc.types.NullPolicy.INCLUDE
                    )
                else:
                    child = value.children[0].evaluate(evaluation_df)
                    aggregation = value.agg_request
                requests.append(
                    plc.groupby.GroupByRequest(
                        _triton_column_to_plc(child, df.cudf_stream),
                        [aggregation],
                    )
                )
            grouped_keys, grouped_values = grouper.aggregate(
                requests, stream=df.cudf_stream
            )
            result_columns = [
                _plc_column_to_triton(
                    column, key.value.dtype, key.name, df.cudf_stream
                )
                for key, column in zip(keys, grouped_keys.columns(), strict=True)
            ]
            result_columns.extend(
                _plc_column_to_triton(
                    column,
                    agg.value.dtype,
                    agg.name,
                    df.cudf_stream,
                )
                for agg, column in zip(
                    aggs,
                    (column for table in grouped_values for column in table.columns()),
                    strict=True,
                )
            )
            return TritonDataFrame(result_columns, df.cudf_stream).slice(zlice)
        except NotImplementedError:
            # The Python implementation below handles expression and
            # aggregation combinations that libcudf cannot represent.
            pass

    key_values = [
        key.obj.cpu().tolist()
        if isinstance(key.obj, torch.Tensor)
        else key.obj.to_arrow(stream=df.cudf_stream).to_pylist()
        for key in key_columns
    ]
    key_validities = [
        None if key.validity is None else key.validity.cpu().tolist() # TODO: CPU?!!?!?!
        for key in key_columns
    ]
    labels = [
        tuple(
            None if not validity[row] else values[row]
            for values, validity in zip(key_values, key_validities, strict=True)
        )
        for row in range(df.num_rows)
    ]
    groups: dict[tuple, list[int]] = {}
    for row, label in enumerate(labels):
        groups.setdefault(label, []).append(row)
    if not maintain_order:
        groups = dict(sorted(groups.items(), key=lambda item: item[0]))

    result_columns = []

    def take_column(column: TritonColumn, indices: torch.Tensor) -> TritonColumn:
        if isinstance(column.obj, plc.Column):
            indices_col = plc.Column.from_array(indices, stream=df.cudf_stream)
            (obj,) = plc.copying.gather(
                plc.Table([column.obj]),
                indices_col,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=df.cudf_stream,
            ).columns()
            return TritonColumn(
                obj,
                dtype=column.dtype,
                validity=column.validity.to(indices.device)[indices],
                name=column.name,
            )
        return TritonColumn(
            column.obj.index_select(0, indices),
            validity=None
            if column.validity is None
            else column.validity.index_select(0, indices),
            dtype=column.dtype,
            name=column.name,
        )
    for key, key_column in zip(keys, key_columns, strict=True):
        group_rows = [rows[0] for rows in groups.values()]
        result_columns.append(
            TritonColumn(
                take_column(
                    key_column,
                    torch.tensor(
                        group_rows, device=df.stream.device, dtype=torch.long
                    ),
                ).obj,
                validity=torch.tensor(
                    [
                        True
                        if key_column.validity is None
                        else bool(key_column.validity[rows[0]].item())
                        for rows in groups.values()
                    ],
                    device=df.stream.device,
                    dtype=torch.bool,
                )
                if key_column.validity is not None
                else None,
                dtype=key_column.dtype,
                name=key.name,
            )
        )

    for agg in aggs:
        result_values = []
        result_validity = []
        result_list_dtype = agg.value.dtype
        for rows in groups.values():
            indices = torch.tensor(rows, device=df.stream.device, dtype=torch.long)
            subgroup = TritonDataFrame(
                [
                    take_column(column, indices)
                    for column in df.columns
                ],
                df.cudf_stream,
            )
            result = agg.evaluate(subgroup)
            if not isinstance(agg.value, TritonAgg) and isinstance(
                result.obj, torch.Tensor
            ):
                result_values.append(
                    plc.Column.from_arrow(
                        pa.array([result.obj.cpu().tolist()]),
                        stream=df.cudf_stream,
                    )
                )
                result_list_dtype = DataType(
                    pl.List(dtype_triton_to_cudf(agg.value.dtype).polars_type)
                )
                result_validity.append(True)
                continue
            if isinstance(result.obj, plc.Column):
                result_values.append(result.obj)
            else:
                result_values.append(result.obj[0].detach().cpu())
            result_validity.append(bool(result.validity[0].detach().cpu().item()))
        if result_values and isinstance(result_values[0], plc.Column):
            result = plc.Column.from_arrow(
                pa.concat_arrays([value.to_arrow(stream=df.cudf_stream)
                                  for value in result_values]),
                stream=df.cudf_stream,
            )
            result_columns.append(
                TritonColumn(
                    result,
                    validity=torch.tensor(
                        result_validity, device=df.stream.device, dtype=torch.bool
                    ),
                    dtype=result_list_dtype,
                    name=agg.name,
                )
            )
        else:
            result_columns.append(
                TritonColumn(
                    torch.stack(result_values).to(device=df.stream.device)
                    if result_values
                    else torch.empty(
                        0, device=df.columns[0].obj.device, dtype=agg.value.dtype
                    ),
                    validity=torch.tensor(
                        result_validity,
                        device=df.stream.device,
                        dtype=torch.bool,
                    ),
                    dtype=agg.value.dtype,
                    name=agg.name,
                )
            )

    return TritonDataFrame(result_columns, df.cudf_stream).slice(zlice)

def slice_triton(df: TritonDataFrame, offset: int, length: int) -> TritonDataFrame:
    return df.slice((offset, length))

def projection_triton(df: TritonDataFrame, schema: Schema) -> TritonDataFrame:
    # This can reorder things.
    columns = broadcast_triton(
        *(df.column_map[name] for name in schema),
        target_length=df.num_rows,
        stream=df.stream
    )
    return TritonDataFrame(columns, df.cudf_stream)

def map_function_triton(df: TritonDataFrame, schema: Schema, name: str, options: Any) -> TritonDataFrame:
    if name == "rechunk":
        # No-op in our data model
        # Don't think this appears in a plan tree from python
        return df  # pragma: no cover
    # elif name == "explode":
    #     ((to_explode,),) = options
    #     index = df.columns.index(to_explode)
    #     subset = df.column_names_set - {to_explode}
        
    # elif name == "unpivot":
    #     (
    #         indices,
    #         pivotees,
    #         variable_name,
    #         value_name,
    #     ) = options
    #     npiv = len(pivotees)
    #     selected = df.select(indices)
    #     index_columns = [
    #         Column(tiled, name=name, dtype=old.dtype)
    #         for tiled, name, old in zip(
    #             plc.reshape.tile(selected.table, npiv, stream=df.stream).columns(),
    #             indices,
    #             selected.columns,
    #             strict=True,
    #         )
    #     ]
    #     (variable_column,) = plc.filling.repeat(
    #         plc.Table(
    #             [
    #                 plc.Column.from_arrow(
    #                     pl.Series(
    #                         values=pivotees, dtype=schema[variable_name].polars_type
    #                     ),
    #                     stream=df.stream,
    #                 )
    #             ]
    #         ),
    #         df.num_rows,
    #         stream=df.stream,
    #     ).columns()
    #     value_column = plc.concatenate.concatenate(
    #         [
    #             df.column_map[pivotee]
    #             .astype(schema[value_name], stream=df.stream)
    #             .obj
    #             for pivotee in pivotees
    #         ],
    #         stream=df.stream,
    #     )
    #     return DataFrame(
    #         [
    #             *index_columns,
    #             Column(
    #                 variable_column, name=variable_name, dtype=schema[variable_name]
    #             ),
    #             Column(value_column, name=value_name, dtype=schema[value_name]),
    #         ],
    #         stream=df.stream,
    #     )
    # elif name == "row_index":
    #     col_name, offset = options
    #     dtype = schema[col_name]
    #     step = plc.Scalar.from_py(1, dtype.plc_type, stream=df.stream)
    #     init = plc.Scalar.from_py(offset, dtype.plc_type, stream=df.stream)
    #     index_col = Column(
    #         plc.filling.sequence(df.num_rows, init, step, stream=df.stream),
    #         is_sorted=plc.types.Sorted.YES,
    #         order=plc.types.Order.ASCENDING,
    #         null_order=plc.types.NullOrder.AFTER,
    #         name=col_name,
    #         dtype=dtype,
    #     )
    #     return DataFrame([index_col, *df.columns], stream=df.stream)
    else:
        raise AssertionError("Should never be reached")  # pragma: no cover


def merge_sorted_triton(*dfs: TritonDataFrame, key: str) -> TritonDataFrame:

    with stream_ordered_after(*dfs) as stream:
        left, right = dfs
        right = right.discard_columns(
            right.column_names_set - left.column_names_set
        )
        on_col_left = left.select_columns({key})[0]
        on_col_right = right.select_columns({key})[0]
        # return merge_dataframes(
        #         [right, left],
        #         key
        #         [left.column_names.index(key), right.column_names.index(key)],
        #         [on_col_left.order, on_col_right.order],
        #         [on_col_left.null_order, on_col_right.null_order],
        #         stream=stream,
        #     ),
        #     left.column_names,
        #     left.dtypes,
        #     stream=stream,
        # )
