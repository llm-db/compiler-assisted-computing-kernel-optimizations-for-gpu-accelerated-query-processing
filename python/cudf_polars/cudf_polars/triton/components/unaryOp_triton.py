
from typing import Any, ClassVar

from cudf.api.types import is_scalar
import torch
import triton
import triton.language as tl
import pylibcudf as plc

from cudf_polars.triton.components.base_triton import ExecutionContext, TritonExpr
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame
from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.triton.components.literal_triton import TritonLiteral

BLOCK_SIZE = 256
class  TritonUnaryFunction(TritonExpr):
    """Class representing unary functions of an expression."""

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    _OP_SET: ClassVar[set[str]] = {
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "exp", "sqrt", "cbrt", "ceil", "floor", "abs", "bit_invert", "not", "negate"
    }
    
    _pointwise_fns = frozenset(
        {
            "fill_null",
            "fill_null_with_strategy",
            "mask_nans",
            "round",
            "set_sorted",
        }
    ).union(_OP_SET)

    def __init__(
        self, dtype: torch.dtype, name: str, options: tuple[Any, ...], *children: TritonExpr
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.children = children
        self.is_pointwise = self.name in TritonUnaryFunction._pointwise_fns

    def do_evaluate(
        self, df: TritonDataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> TritonColumn:
        with torch.cuda.stream(df.stream):
            if self.name == "mask_nans":
                column = self.children[0].evaluate(df, context=context)
                validity = ~torch.isnan(column.obj)
                validity &= column.validity
                return TritonColumn(
                    column.obj,
                    validity=validity,
                    dtype=column.dtype,
                    name=column.name,
                )
            if self.name in ("is_null", "is_not_null"):
                column = self.children[0].evaluate(df, context=context)
                result = column.validity
                if self.name == "is_null":
                    result = ~result
                return TritonColumn(result, dtype=self.dtype, name=column.name)
            if self.name == "is_in":
                values = self.children[0].evaluate(df, context=context)
                allowed = self.children[1].evaluate(df, context=context)
                if isinstance(values.obj, plc.Column) or isinstance(
                    allowed.obj, plc.Column
                ):
                    from cudf_polars.containers import Column
                    from cudf_polars.triton.triton_conversion import (
                        column_cudf_to_triton,
                        dtype_triton_to_cudf,
                    )
                    from cudf_polars.containers import DataType

                    result = plc.search.contains(
                        allowed.obj,
                        values.obj,
                        stream=df.cudf_stream,
                    )
                    return column_cudf_to_triton(
                        Column(result, dtype=dtype_triton_to_cudf(self.dtype)),
                        name=values.name,
                        stream=df.cudf_stream,
                    )
                return TritonColumn(
                    torch.isin(values.obj, allowed.obj),
                    dtype=self.dtype,
                    name=values.name,
                )
            if self.name == "not":
                column = self.children[0].evaluate(df, context=context)
                if isinstance(column.obj, plc.Column):
                    from cudf_polars.containers import Column
                    from cudf_polars.triton.triton_conversion import (
                        column_cudf_to_triton,
                        dtype_triton_to_cudf,
                    )

                    result = plc.unary.unary_operation(
                        column.obj,
                        plc.unary.UnaryOperator.NOT,
                        stream=df.cudf_stream,
                    )
                    return column_cudf_to_triton(
                        Column(result, dtype=dtype_triton_to_cudf(self.dtype)),
                        name=column.name,
                        stream=df.cudf_stream,
                    )
                return TritonColumn(
                    ~column.obj,
                    validity=column.validity,
                    dtype=self.dtype,
                    name=column.name,
                )
            if self.name == "round":
                column = self.children[0].evaluate(df, context=context)
                decimal_places = self.options[0]
                if isinstance(column.obj, plc.Column):
                    from cudf_polars.containers import Column, DataType
                    from cudf_polars.triton.triton_conversion import (
                        column_cudf_to_triton,
                    )

                    rounded = plc.round.round(
                        column.obj,
                        decimal_places=decimal_places,
                        stream=df.cudf_stream,
                    )
                    return column_cudf_to_triton(
                        Column(
                            rounded,
                            dtype=(
                                column.dtype
                                if isinstance(column.dtype, DataType)
                                else DataType(polars_dtype=column.dtype)
                            ),
                        ),
                        name=column.name,
                        stream=df.cudf_stream,
                    )
                return TritonColumn(
                    torch.round(column.obj, decimals=decimal_places),
                    validity=column.validity,
                    dtype=column.dtype,
                    name=column.name,
                )
            if self.name == "abs":
                column = self.children[0].evaluate(df, context=context)
                return TritonColumn(
                    torch.abs(column.obj),
                    validity=column.validity,
                    dtype=column.dtype,
                    name=column.name,
                )
            if self.name == "fill_null":
                column = self.children[0].evaluate(df, context=context)
                fill_value = self.children[1]
                if isinstance(fill_value, TritonLiteral):
                    arg = fill_value.value
                else:
                    evaluated = fill_value.evaluate(df, context=context)
                    arg = (
                        evaluated.obj_scalar(df.stream)
                        if evaluated.is_scalar
                        else evaluated.obj
                    )
                if not isinstance(column.obj, torch.Tensor):
                    from cudf_polars.containers import Column
                    from cudf_polars.triton.triton_conversion import (
                        column_cudf_to_triton,
                        dtype_triton_to_cudf,
                    )

                    column_dtype = dtype_triton_to_cudf(column.dtype)
                    replacement = plc.Scalar.from_py(
                        arg,
                        column_dtype.plc_type,
                        stream=df.cudf_stream,
                    )
                    replaced = plc.replace.replace_nulls(
                        column.obj,
                        replacement,
                        stream=df.cudf_stream,
                    )
                    return column_cudf_to_triton(
                        Column(replaced, dtype=column_dtype),
                        name=column.name,
                        stream=df.cudf_stream,
                    )
                # if not isinstance(arg, TritonColumn):
                #     arg = (
                #         TritonColumn(
                #             [arg],
                #             dtype=fill_value.dtype,
                #         )
                #         .astype(column.dtype, stream=df.stream)
                #         .obj.to_scalar(stream=df.stream)
                #     )
                return replace_null(column, arg, df.stream)
            raise NotImplementedError(
                f"Unimplemented unary function {self.name=}"
            )


class TritonCast(TritonExpr):
    """Class representing a cast of an expression."""

    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: torch.dtype, child: TritonExpr) -> None:
        self.dtype = dtype
        self.children = (child,)
        self.is_pointwise = True

    def do_evaluate(
        self,
        df: TritonDataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> TritonColumn:
        column = self.children[0].evaluate(df, context=context)
        if not isinstance(column.obj, torch.Tensor):
            from cudf_polars.containers import DataType
            from cudf_polars.containers.column import Column
            from cudf_polars.triton.triton_conversion import (
                column_cudf_to_triton,
                dtype_triton_to_cudf,
            )

            if not isinstance(self.dtype, torch.dtype):
                target_dtype = DataType(polars_dtype=self.dtype)
            else:
                target_dtype = dtype_triton_to_cudf(self.dtype)
            if not isinstance(target_dtype, DataType):
                raise TypeError(f"Unsupported Triton cast target: {self.dtype!r}")
            source_dtype = dtype_triton_to_cudf(column.dtype)
            if (
                source_dtype.plc_type.id() == plc.TypeId.LIST
                and target_dtype.plc_type.id() == plc.TypeId.UINT32
            ):
                counted = plc.lists.extract_list_element(
                    column.obj,
                    0,
                    stream=df.cudf_stream,
                )
                return column_cudf_to_triton(
                    Column(counted, dtype=target_dtype),
                    name=column.name,
                    stream=df.cudf_stream,
                )
            if (
                source_dtype.plc_type.id() == target_dtype.plc_type.id()
                and (
                    not plc.traits.is_fixed_point(source_dtype.plc_type)
                    or source_dtype.plc_type.scale() == target_dtype.plc_type.scale()
                )
            ):
                return TritonColumn(
                    column.obj,
                    validity=column.validity,
                    dtype=target_dtype,
                    name=column.name,
                )
            casted = plc.unary.cast(
                column.obj,
                target_dtype.plc_type,
                stream=df.cudf_stream,
            )
            return column_cudf_to_triton(
                Column(casted, dtype=target_dtype),
                name=column.name,
                stream=df.cudf_stream,
            )
        if not isinstance(self.dtype, torch.dtype):
            from cudf_polars.containers import DataType
            from cudf_polars.containers.column import Column
            from cudf_polars.triton.triton_conversion import (
                column_cudf_to_triton,
            )

            target_dtype = DataType(polars_dtype=self.dtype)
            source = plc.Column.from_array(column.obj, stream=df.cudf_stream)
            casted = plc.unary.cast(
                source,
                target_dtype.plc_type,
                stream=df.cudf_stream,
            )
            return column_cudf_to_triton(
                Column(casted, dtype=target_dtype),
                name=column.name,
                stream=df.cudf_stream,
            )
        return TritonColumn(
            column.obj.to(self.dtype),
            validity=column.validity,
            dtype=self.dtype,
            name=column.name,
        )

@triton.jit
def replace_null_kernel(
    x_ptr,
    mask_ptr,
    out_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    none_mask = tl.load(mask_ptr + offsets, mask=mask)

    # Replace invalid entries with `value`.
    x = tl.where(none_mask, x, value)

    tl.store(out_ptr + offsets, x, mask=mask)

def replace_null(col: TritonColumn, val, stream: torch.Stream) -> TritonColumn:
    ret = torch.empty(
        col.obj.shape,
        device=col.obj.device,
        dtype=col.obj.dtype,
    )

    n_elements = ret.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )

    with torch.cuda.stream(stream):
        replace_null_kernel[grid](
            col.obj,
            col.validity,
            ret,
            val,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return TritonColumn(
        column=ret,
        dtype=col.dtype,
        is_sorted=col.is_sorted,
        order=col.order,
        null_order=col.null_order,
        name=col.name
    )
