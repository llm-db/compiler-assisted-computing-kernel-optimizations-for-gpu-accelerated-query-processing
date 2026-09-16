

from ast import BinOp
from typing import Any

import torch

from cudf_polars.triton.components.dataframe_triton import TritonDataFrame
from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.triton.utils_triton import broadcast_triton, stream_ordered_after
from cudf_polars.triton.components.base_triton import TritonExpr, TritonNamedExpr
from cudf_polars.typing import Schema, Slice as Zlice
from cudf_polars.triton.components.join_triton import join_coalesce, join_gather, join_gather_column, join_pairs
from cudf_polars.triton.components.binOp_triton import TritonBinOp


def join_triton(
    left: TritonDataFrame,
    right: TritonDataFrame,
    left_on: tuple[TritonNamedExpr, ...],
    right_on: tuple[TritonNamedExpr, ...],
    options: tuple[str, bool, Zlice | None, str, bool, str],
) -> TritonDataFrame:
    """Evaluate a regular join using Triton tensors."""
    how, nulls_equal, zlice, suffix, coalesce, maintain_order = options
    left_keys = list(broadcast_triton(*(key.evaluate(left) for key in left_on), target_length=left.num_rows))
    right_keys = list(broadcast_triton(*(key.evaluate(right) for key in right_on), target_length=right.num_rows))
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
        return TritonDataFrame(columns, left.stream).slice(zlice)

    left_columns, right_columns = join_gather(left, right, options, left_indices, right_indices)
    if coalesce:
        left_columns, right_columns = join_coalesce(left, right, right_on, left_on, options, left_columns, right_columns)
    return TritonDataFrame([*left_columns, *right_columns], left.stream).slice(zlice)


def conditional_join_triton(
    left: TritonDataFrame,
    right: TritonDataFrame,
    predicate: TritonBinOp,
    options: tuple[None, bool, Zlice | None, str, bool, str],
) -> TritonDataFrame:
    """Evaluate a conditional inner join using a pairwise Torch mask."""
    _, nulls_equal, zlice, suffix, coalesce, maintain_order = options
    print(left, right)
    if not isinstance(predicate, TritonBinOp):
        raise TypeError("BinOp expected")
    print(predicate.op)

    # n_left = left.num_rows
    # n_right = right.num_rows
    # matches = torch.ones((n_left, n_right), device=left.columns[0].obj.device, dtype=torch.bool)

    # TODO: Not good, should be 2D all in triton
    
        # new_col = TritonColumn(
        #     torch.tensor([col.obj[r]], device=col.obj.device), 
        #     dtype=col.dtype, 
        #     validity=torch.tensor([col.validity[r]], device=col.obj.device)
        # )
    matches_obj, matches_validity = predicate.do_evaluate_dual(
        left, 
        right
    )

    print(matches_obj)
    left_indices, right_indices = torch.nonzero(matches_obj, as_tuple=True)

    left_columns, right_columns = join_gather(left, right, options, left_indices, right_indices)
    # if coalesce:
    #     left_columns, right_columns = join_coalesce(left, right, right_on, left_on, options, left_columns, right_columns)
    df = TritonDataFrame([*left_columns, *right_columns], left.stream).slice(zlice)
    print(df)
    return df

    # print(options)
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
    (mask,) = broadcast_triton(ev, target_length=df.num_rows)
    df2 = df.filter(mask)
    return df2

def union_triton(df: TritonDataFrame, zlice: Zlice) -> TritonDataFrame:

    ev = mask.evaluate(df)
    (mask,) = broadcast_triton(ev, target_length=df.num_rows)
    df2 = df.filter(mask)
    return df2

def select_triton(df: TritonDataFrame, exprs: tuple[TritonNamedExpr, ...], should_broadcast: bool) -> TritonDataFrame:
    columns = [e.evaluate(df) for e in exprs]
    if should_broadcast:
        columns = broadcast_triton(*columns)
    return TritonDataFrame(columns, df.stream)

def reduce_triton(df: TritonDataFrame, exprs: tuple[TritonNamedExpr, ...]) -> TritonDataFrame:
    columns = broadcast_triton(*(expr.evaluate(df) for expr in exprs))
    columns = [column.rename(expr.name) for column, expr in zip(columns, exprs, strict=True)]
    return TritonDataFrame(columns, df.stream)

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
        window_df = TritonDataFrame(window_columns, df.stream)
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

    return TritonDataFrame(result_columns, df.stream).slice(zlice)

def groupby_triton(
    df: TritonDataFrame,
    keys: tuple[TritonNamedExpr, ...],
    aggs: tuple[TritonNamedExpr, ...],
    maintain_order: bool,
    zlice: Zlice | None,
) -> TritonDataFrame:
    key_columns = broadcast_triton(
        *(key.evaluate(df) for key in keys), target_length=df.num_rows
    )
    key_values = [key.obj.cpu().tolist() for key in key_columns]
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
    for key, key_column in zip(keys, key_columns, strict=True):
        group_rows = [rows[0] for rows in groups.values()]
        result_columns.append(
            TritonColumn(
                key_column.obj.index_select(
                    0,
                    torch.tensor(
                        group_rows, device=key_column.obj.device, dtype=torch.long
                    ),
                ),
                validity=torch.tensor(
                    [
                        True
                        if key_column.validity is None
                        else bool(key_column.validity[rows[0]].item())
                        for rows in groups.values()
                    ],
                    device=key_column.obj.device,
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
        for rows in groups.values():
            indices = torch.tensor(rows, device=df.stream.device, dtype=torch.long)
            subgroup = TritonDataFrame(
                [
                    TritonColumn(
                        column.obj.index_select(0, indices),
                        validity=None
                        if column.validity is None
                        else column.validity.index_select(0, indices),
                        dtype=column.dtype,
                        name=column.name,
                    )
                    for column in df.columns
                ],
                df.stream,
            )
            result = agg.evaluate(subgroup)
            result_values.append(result.obj[0])
            result_validity.append(
                bool(result.validity[0].item())
            )
        result_columns.append(
            TritonColumn(
                torch.stack(result_values)
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

    return TritonDataFrame(result_columns, df.stream).slice(zlice)

def slice_triton(df: TritonDataFrame, offset: int, length: int) -> TritonDataFrame:
    return df.slice((offset, length))

def projection_triton(df: TritonDataFrame, schema: Schema) -> TritonDataFrame:
    # This can reorder things.
    columns = broadcast_triton(
        *(df.column_map[name] for name in schema),
        target_length=df.num_rows
    )
    return TritonDataFrame(columns, df.stream)

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


