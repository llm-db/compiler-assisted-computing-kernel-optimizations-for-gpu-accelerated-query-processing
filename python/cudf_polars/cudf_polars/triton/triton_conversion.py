
import contextlib
import threading
import time
from typing import List

import torch
import cupy as cp

from cudf_polars.dsl.expressions.base import Col, ColRef, Expr, NamedExpr
from cudf_polars.dsl.expressions.literal import Literal, LiteralColumn
from cudf_polars.triton.components.base_triton import TritonCol, TritonColRef, TritonExpr, TritonNamedExpr
from cudf_polars.triton.components.literal_triton import TritonLiteral, TritonLiteralColumn
from cudf_polars.dsl.expressions.binaryop import BinOp
from cudf_polars.containers.dataframe import DataFrame
from cudf_polars.dsl.expressions.aggregation import Agg
from cudf_polars.dsl.expressions.unary import Cast, Len, UnaryFunction
from cudf_polars.dsl.expressions.boolean import BooleanFunction
import pylibcudf as plc
import polars as pl

from pylibcudf import transform
from cudf_polars.containers import DataType

from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame
from cudf_polars.containers.column import Column
from cudf_polars.triton.components.binOp_triton import TritonBinOp
from cudf_polars.triton.components.unaryOp_triton import TritonCast, TritonUnaryFunction
from cudf_polars.triton.components.agg_triton import TritonAgg, TritonLen
from cudf_polars.triton.components.ternary_triton import TritonTernary
from cudf_polars.dsl.expressions.ternary import Ternary


class ConversionTimer:
    """Accumulate host time spent converting Triton and cuDF dataframes."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._active = 0
        self._started: float | None = None

    def start(self, timestamp: float) -> None:
        if self._active == 0:
            self._started = timestamp
        self._active += 1

    def stop(self, timestamp: float) -> None:
        self._active -= 1
        if self._active == 0:
            assert self._started is not None
            self.elapsed += timestamp - self._started
            self._started = None


_active_conversion_timer: ConversionTimer | None = None
_conversion_timer_lock = threading.Lock()


@contextlib.contextmanager
def conversion_timer():
    """Measure dataframe conversion time in the current execution context."""
    global _active_conversion_timer
    timer = ConversionTimer()
    with _conversion_timer_lock:
        previous = _active_conversion_timer
        _active_conversion_timer = timer
    try:
        yield timer
    finally:
        with _conversion_timer_lock:
            _active_conversion_timer = previous


@contextlib.contextmanager
def _record_conversion():
    start = time.perf_counter()
    with _conversion_timer_lock:
        timer = _active_conversion_timer
        if timer is not None:
            timer.start(start)
    try:
        yield
    finally:
        end = time.perf_counter()
        with _conversion_timer_lock:
            if timer is not None:
                timer.stop(end)


class ConversionManager:
    def __init__(self):
        self.ref_map = {}

    def add_conversion(self, cudf_version, triton_version):
        if(cudf_version in self.ref_map):
            raise NotImplementedError("THIS SHOULD NOT HAPPEN?")
        self.ref_map[cudf_version] = triton_version

    def try_get_conversion(self, cudf_version):
        if cudf_version in self.ref_map:
            return self.ref_map[cudf_version]
        return None

def expr_cudf_to_triton(expr: Expr, con: ConversionManager, stream) -> TritonExpr:
    c = con.try_get_conversion(expr)
    if c != None:
        return c
    triton_expr = _expr_cudf_to_triton(expr, con, stream)
    con.add_conversion(expr, triton_expr)
    return triton_expr

def _expr_cudf_to_triton(expr: Expr, con: ConversionManager, stream) -> TritonExpr:
    
    ty = dtype_cudf_to_triton(expr.dtype)
    if isinstance(expr, Col):
        return TritonCol(ty, expr.name)
    if isinstance(expr, Literal):
        return TritonLiteral(ty, expr.value)
    if isinstance(expr, BinOp):
        return TritonBinOp(ty, expr.op, expr_cudf_to_triton(expr.children[0], con, stream), expr_cudf_to_triton(expr.children[1], con, stream))
    if isinstance(expr, LiteralColumn):
        col2 = Column(plc.Column.from_arrow(expr.value, stream=stream), dtype=expr.dtype)
        return TritonLiteralColumn(
            ty,
            column_cudf_to_triton(
                col=col2,
                name=expr.value.name,
                stream=stream,
            )
        )
    if isinstance(expr, Len):
        return TritonLen(ty)
    if isinstance(expr, Cast):
        return TritonCast(ty, expr_cudf_to_triton(expr.children[0], con, stream))
    if isinstance(expr, Agg):
        children = [expr_cudf_to_triton(e, con, stream) for e in expr.children]
        return TritonAgg(
            ty,
            expr.name,
            expr.options,
            expr.context,
            *children,
        )
    if isinstance(expr, Ternary):
        return TritonTernary(
            ty,
            *(expr_cudf_to_triton(child, con, stream) for child in expr.children),
        )
    if isinstance(expr, UnaryFunction):
        children = [expr_cudf_to_triton(e, con, stream) for e in expr.children]
        return TritonUnaryFunction( 
            ty,           
            expr.name,    
            expr.options, 
            *children     
        )
    if isinstance(expr, BooleanFunction):
        if expr.name is BooleanFunction.Name.IsIn:
            return TritonUnaryFunction(
                ty,
                "is_in",
                expr.options,
                *(expr_cudf_to_triton(e, con, stream) for e in expr.children),
            )
        if expr.name is BooleanFunction.Name.Not:
            return TritonUnaryFunction(
                ty,
                "not",
                expr.options,
                expr_cudf_to_triton(expr.children[0], con, stream),
            )
        if expr.name not in (
            BooleanFunction.Name.IsNull,
            BooleanFunction.Name.IsNotNull,
        ):
            raise NotImplementedError(expr.name)
        child = expr_cudf_to_triton(expr.children[0], con, stream)
        return TritonUnaryFunction(
            ty,
            "is_null" if expr.name is BooleanFunction.Name.IsNull else "is_not_null",
            expr.options,
            child,
        )
    if isinstance(expr, ColRef):
        c = con.try_get_conversion(expr.children[0])
        if c == None:
            c = expr_cudf_to_triton(expr.children[0], con, stream)
        return TritonColRef(ty, expr.index, expr.table_ref, c)

    raise NotImplementedError(type(expr))

def named_expr_cudf_to_triton(expr: NamedExpr, con: ConversionManager, stream):
    return TritonNamedExpr(
        name=expr.name,
        value=expr_cudf_to_triton(expr.value, con, stream)
    )

def dataframes_cudf_to_triton(*dfs: DataFrame) -> List[TritonDataFrame]:
    return [dataframe_cudf_to_triton(df) for df in dfs]

def schema_cudf_to_triton(schema):
    d = dict()
    for s in schema:
        d[s] = dtype_cudf_to_triton(schema[s])
    return d


def dataframe_cudf_to_triton(df: DataFrame) -> TritonDataFrame:
    with _record_conversion():
        if not isinstance(df, DataFrame):
            raise ValueError(df)

        cached = getattr(df, "_triton", None)
        if cached is not None:
            return cached

        triton_cols = [
            column_cudf_to_triton(
                col,
                name=col.name,
                stream=df.stream,
            )
            for col in df.columns
        ]
        result = TritonDataFrame(
            triton_cols,
            cudf_stream=df.stream,
            num_rows=df._num_rows_override,
        )
        result.cudfFallback = df
        df._triton = result
        return result


def dataframe_triton_to_cudf(df: TritonDataFrame) -> DataFrame:
    """Convert to a cudf DataFrame."""
    with _record_conversion():
        if df.cudfFallback is not None:
            return df.cudfFallback

        stream = df.cudf_stream
        gpu_cols = []
        for col in df.columns:
            if isinstance(col.obj, plc.Column):
                plc_col = col.obj
            else:
                plc_col = plc.Column.from_array(col.obj, stream=stream)

            if col.validity is not None and not isinstance(col.obj, plc.Column):
                validity_col = plc.Column.from_array(
                    cp.asarray(col.validity), stream=stream
                )
                validity_mask, null_count = transform.bools_to_mask(
                    validity_col, stream=stream
                )
                plc_col = plc_col.with_mask(validity_mask, null_count)

            gpu_col = Column(
                column=plc_col,
                name=col.name,
                dtype=dtype_triton_to_cudf(col.dtype),
            )
            gpu_cols.append(gpu_col)

        result = DataFrame(gpu_cols, stream=stream, num_rows=df._num_rows_override)
        result._triton = df
        df.cudfFallback = result
        return result

def column_cudf_to_triton(
    col: Column,
    *,
    name: str | None = None,
    stream,
) -> TritonColumn:
    if not isinstance(col, Column):
        raise ValueError(col)
    values_only = col.obj.with_mask(None, 0)
    dtype = dtype_cudf_to_triton(col.dtype)

    # DLPack only describes strided tensors. Nested and other non-primitive
    # libcudf columns are retained as opaque payloads so unrelated Triton
    # operations can still consume the rest of the dataframe.
    if not isinstance(dtype, torch.dtype):
        validity = None
        if col.obj.null_mask() is not None:
            validity_col = transform.mask_to_bools(
                int(col.obj.null_mask().ptr),
                col.obj.offset(),
                col.obj.offset() + col.obj.size(),
                stream=stream,
            )
            validity_capsule = plc.interop.to_dlpack(
                plc.Table([validity_col]),
                stream=stream,
            )
            validity = torch.from_dlpack(validity_capsule).to(torch.bool)
        return TritonColumn(
            column=values_only,
            validity=validity,
            name=name,
            dtype=dtype,
        )

    values_capsule = plc.interop.to_dlpack(
        plc.Table([values_only]),
        stream=stream,
    )
    tensor = torch.from_dlpack(values_capsule).cuda()

    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if dtype == torch.bool:
        tensor = tensor.bool()
    validity = None

    # if '''col.obj.null_count() > 0 and '''col.obj.null_mask() is not None:
    if col.obj.null_mask() is not None:
        validity_col = transform.mask_to_bools(
            int(col.obj.null_mask().ptr),
            col.obj.offset(),
            col.obj.offset() + col.obj.size(),
            stream=stream,
        )

        validity_capsule = plc.interop.to_dlpack(
            plc.Table([validity_col]),
            stream=stream,
        )
        validity = torch.from_dlpack(validity_capsule).to(torch.bool)


    return TritonColumn(
        column=tensor,
        validity=validity,
        name=name,
        dtype=dtype,
    )



def dtype_cudf_to_triton(ty: DataType):
    """Convert a cudf-polars dtype to a Triton dtype or opaque logical dtype."""
    dtype = ty.polars_type

    if isinstance(dtype, pl.Boolean):
        return torch.bool
    elif isinstance(dtype, pl.Int8):
        return torch.int8
    elif isinstance(dtype, pl.Int16):
        return torch.int16
    elif isinstance(dtype, pl.Int32):
        return torch.int32
    elif isinstance(dtype, pl.Int64):
        return torch.int64
    elif isinstance(dtype, pl.UInt8):
        return torch.uint8
    elif isinstance(dtype, pl.UInt16):
        return torch.int32
    elif isinstance(dtype, pl.UInt32):
        return torch.uint32
    elif isinstance(dtype, pl.UInt64):
        return torch.uint64
    elif isinstance(dtype, pl.Float32):
        return torch.float32
    elif isinstance(dtype, pl.Float64):
        return torch.float64
    elif isinstance(
        dtype,
        (
            pl.Date,
            pl.Time,
            pl.Datetime,
            pl.Duration,
            pl.String,
            pl.Decimal,
            pl.Null,
            pl.List,
            pl.Struct,
        ),
    ):
        return dtype
    else:
        # Preserve future or extension dtypes as opaque values rather than
        # requiring every dtype to be understood by the tensor backend.
        return dtype


def dtype_triton_to_cudf(ty) -> DataType:
    """Convert a PyTorch/Triton dtype to a cuDF-Polars DataType."""

    if isinstance(ty, DataType):
        return ty

    if ty == torch.bool:
        dtype = pl.Boolean()
    elif ty == torch.int8:
        dtype = pl.Int8()
    elif ty == torch.int16:
        dtype = pl.Int16()
    elif ty == torch.int32:
        dtype = pl.Int32()
    elif ty == torch.int64:
        dtype = pl.Int64()
    elif ty == torch.uint8:
        dtype = pl.UInt8()
    elif ty == torch.uint32:
        dtype = pl.UInt32()
    elif ty == torch.uint64:
        dtype = pl.UInt64()
    elif ty == torch.float32:
        dtype = pl.Float32()
    elif ty == torch.float64:
        dtype = pl.Float64()
    else:
        return DataType(polars_dtype=ty)

    return DataType(
        polars_dtype=dtype,
    )
