from contextlib import contextmanager
from typing import Generator

from polars import DataType
import torch
import triton
import triton.language as tl
from cudf_polars.triton.components.column_triton import TritonColumn
import pylibcudf as plc
import numpy as np

from cudf_polars.triton.components.base_triton import TritonColRef, TritonExpr
from cudf_polars.triton.components.binOp_triton import TritonBinOp
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame


def broadcast_triton(
    *columns: TritonColumn, target_length: int | None = None, stream
) -> list[TritonColumn]:
    """
    Broadcast a sequence of columns to a common length.

    Parameters
    ----------
    columns
        Columns to broadcast.
    target_length
        Optional length to broadcast to. If not provided, uses the
        non-unit length of existing columns.

    Returns
    -------
    List of broadcasted columns all of the same length.

    Raises
    ------
    RuntimeError
        If broadcasting is not possible.

    Notes
    -----
    In evaluation of a set of expressions, polars type-puns length-1
    columns with scalars. When we insert these into a DataFrame
    object, we need to ensure they are of equal length. This function
    takes some columns, some of which may be length-1 and ensures that
    all length-1 columns are broadcast to the length of the others.

    Broadcasting is only possible if the set of lengths of the input
    columns is a subset of ``{1, n}`` for some (fixed) ``n``. If
    ``target_length`` is provided and not all columns are length-1
    (i.e. ``n != 1``), then ``target_length`` must be equal to ``n``.
    """
    if len(columns) == 0:
        return []
    lengths: set[int] = {column.size for column in columns}
    if lengths == {1}:
        if target_length is None:
            return list(columns)
        nrows = target_length
    else:
        try:
            (nrows,) = lengths.difference([1])
        except ValueError as e:
            raise RuntimeError("Mismatching column lengths") from e
        if target_length is not None and nrows != target_length:
            raise RuntimeError(
                f"Cannot broadcast columns of length {nrows=} to {target_length=}"
            )

    ls = []
    for column in columns:
        if column.size != 1:
            ls.append(column)
        else:
            scalar = column.obj_scalar(stream)
            if(scalar == None):
                obj = torch.zeros(nrows).to(dtype=column.dtype).cuda()
                validity = torch.zeros(nrows).to(dtype=torch.bool).cuda()
            else:
                obj = torch.tensor(
                    [column.obj_scalar(stream)] * nrows,
                    dtype=column.dtype,
                    device="cuda",
                )
                validity = torch.ones(nrows, dtype=torch.bool, device="cuda")
            ls.append(TritonColumn(
                obj,
                validity=validity,
                is_sorted=plc.types.Sorted.YES,
                order=plc.types.Order.ASCENDING,
                null_order=plc.types.NullOrder.BEFORE,
                name=column.name,
                dtype=column.dtype,
            ))
                
    return ls

def concatonate_dataframes(*dfs: TritonDataFrame, context):
    if not dfs:
        raise ValueError("At least one dataframe is required")

    first = dfs[0]
    for d in dfs:
        print(d)

    # All frames must have the same schema/order.
    expected_names = first.column_names
    expected_dtypes = first.dtypes

    for df in dfs[1:]:
        if df.column_names != expected_names:
            raise ValueError("All dataframes must have identical column names")
        if df.dtypes != expected_dtypes:
            raise ValueError("All dataframes must have identical dtypes")

    # Nothing to concatenate.
    if len(dfs) == 1:
        return dfs[0]

    streams = [df.stream for df in dfs]
    print(streams)
    with stream_ordered_after(*(df.stream for df in dfs)):
        columns = []

        for i, name in enumerate(expected_names):
            input_columns = [df.columns[i] for df in dfs]

            obj = torch.cat(
                [col.obj for col in input_columns],
                dim=0,
            )

            validity = torch.cat(
                [col.validity for col in input_columns],
                dim=0,
            )

            # Concatenation is only known to preserve sortedness when
            # every input is sorted compatibly. Without explicitly
            # checking boundary values, conservatively drop the metadata.
            columns.append(
                TritonColumn(
                    obj,
                    dtype=input_columns[0].dtype,
                    validity=validity,
                    name=name,
                )
            )

    print(type(first)(columns, first.cudf_stream))
    return type(first)(columns, first.cudf_stream)

# TODO?: Use?
@contextmanager
def stream_ordered_after(
    *upstreams: torch.cuda.Stream,
) -> Generator[torch.cuda.Stream, None, None]:
    result_stream = torch.cuda.Stream()

    for upstream in upstreams:
        event = torch.cuda.Event()

        with torch.cuda.stream(upstream):
            event.record()

        result_stream.wait_event(event)

    with torch.cuda.stream(result_stream):
        yield result_stream

# _BINOPS = {
#     plc.binaryop.BinaryOperator.EQUAL,
#     plc.binaryop.BinaryOperator.NOT_EQUAL,
#     plc.binaryop.BinaryOperator.LESS,
#     plc.binaryop.BinaryOperator.LESS_EQUAL,
#     plc.binaryop.BinaryOperator.GREATER,
#     plc.binaryop.BinaryOperator.GREATER_EQUAL,
#     # TODO: Handle other binary operations as needed
# }

# def _add_cast(
#     target: DataType,
#     side: TritonColRef,
#     left_casts: dict[str, DataType],
#     right_casts: dict[str, DataType],
# ) -> None:  # pragma: no cover
#     (col,) = side.children
#     assert isinstance(col, TritonColumn)
#     casts = (
#         left_casts if side.table_ref == plc.expressions.TableReference.LEFT else right_casts
#     )
#     casts[col.name] = target

# def _align_decimal_binop_types(
#     left_expr: TritonColRef,
#     right_expr: TritonColRef,
#     left_casts: dict[str, DataType],
#     right_casts: dict[str, DataType],
# ) -> None:
#     left_type, right_type = left_expr.dtype, right_expr.dtype

#     if plc.traits.is_fixed_point(left_type.plc_type) and plc.traits.is_fixed_point(
#         right_type.plc_type
#     ):
#         target = DataType.common_decimal_dtype(left_type, right_type)

#         if (
#             left_type.id() != target.id() or left_type.scale() != target.scale()
#         ):  # pragma: no cover
#             _add_cast(target, left_expr, left_casts, right_casts)

#         if right_type.id() != target.id() or right_type.scale() != target.scale():
#             _add_cast(target, right_expr, left_casts, right_casts)

#     elif (
#         plc.traits.is_fixed_point(left_type.plc_type)
#         and plc.traits.is_floating_point(right_type.plc_type)
#     ) or (
#         plc.traits.is_fixed_point(right_type.plc_type)
#         and plc.traits.is_floating_point(left_type.plc_type)
#     ):  # pragma: no cover
#         is_decimal_left = plc.traits.is_fixed_point(left_type.plc_type)
#         decimal_expr, float_expr = (
#             (left_expr, right_expr) if is_decimal_left else (right_expr, left_expr)
#         )
#         _add_cast(decimal_expr.dtype, float_expr, left_casts, right_casts)

# def collect_decimal_binop_casts(
#     predicate: TritonExpr,
# ) -> tuple[dict[str, DataType], dict[str, DataType]]:
#     left_casts: dict[str, DataType] = {}
#     right_casts: dict[str, DataType] = {}

#     def _walk(node: TritonExpr) -> None:
#         if isinstance(node, TritonBinOp) and node.op in _BINOPS:
#             left_expr, right_expr = node.children
#             if isinstance(left_expr, TritonColRef) and isinstance(
#                 right_expr, TritonColRef
#             ):
#                 _align_decimal_binop_types(
#                     left_expr, right_expr, left_casts, right_casts
#                 )
#         for child in node.children:
#             _walk(child)

#     _walk(predicate)
#     return left_casts, right_casts