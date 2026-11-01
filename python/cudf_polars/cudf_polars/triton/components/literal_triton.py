from typing import Any, Hashable, NoReturn

import torch

from cudf_polars.triton.components.base_triton import ExecutionContext, TritonExpr
import pylibcudf as plc
import polars as pl

from cudf_polars.triton.components.dataframe_triton import TritonDataFrame
from cudf_polars.triton.components.column_triton import TritonColumn

__all__ = ["TritonLiteral"]

class TritonLiteral(TritonExpr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: Any  # Python scalar

    def __init__(self, dtype: torch.dtype, value: Any) -> None:
        if value is None and dtype == None:
            # TypeId.EMPTY not supported by libcudf
            # cuDF Python also maps EMPTY to INT8
            dtype = torch.int8
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def do_evaluate(
        self, df: TritonDataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> TritonColumn:
        """Evaluate this expression given a dataframe for context."""
        if self.value is None:
            return TritonColumn(
                torch.zeros(1, device="cuda", dtype=self.dtype),
                dtype=self.dtype,
                validity=torch.zeros(1, device="cuda", dtype=torch.bool),
            )
        if not isinstance(self.dtype, torch.dtype):
            from cudf_polars.containers import DataType

            dtype = DataType(polars_dtype=self.dtype)
            scalar = plc.Scalar.from_py(
                self.value,
                dtype.plc_type,
                stream=df.cudf_stream,
            )
            return TritonColumn(
                plc.Column.from_scalar(scalar, 1, stream=df.cudf_stream),
                dtype=self.dtype,
            )
        return TritonColumn(
            torch.tensor([self.value], device="cuda", dtype=self.dtype),
            dtype=self.dtype,
        )

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover

    def get_hashable(self) -> Hashable:
        """Get the hash of the literal."""
        return (type(self), self.dtype.plc_type, id(self.value))

    def astype(self, dtype: torch.dtype):
        """Cast self to dtype."""
        if dtype.id() == plc.TypeId.STRUCT:
            raise NotImplementedError("Struct literals are not supported")
        if self.value is None:
            return TritonLiteral(dtype, self.value)
        else:
            # Use polars to cast instead of pylibcudf
            # since there are just Python scalars
            casted = pl.Series(values=[self.value], dtype=self.dtype.polars_type).cast(
                dtype.polars_type
            )[0]
            return TritonLiteral(dtype, casted)

class TritonLiteralColumn(TritonExpr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: pl.Series

    def __init__(self, dtype: torch.dtype, value: TritonColumn) -> None:
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def get_hashable(self) -> Hashable:
        """Compute a hash of the column."""
        # This is stricter than necessary, but we only need this hash
        # for identity in groupby replacements so it's OK. And this
        # way we avoid doing potentially expensive compute.
        return (type(self), self.dtype, id(self.value))

    def do_evaluate(
        self, df: TritonDataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> TritonColumn:
        """Evaluate this expression given a dataframe for context."""
        return self.value

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover