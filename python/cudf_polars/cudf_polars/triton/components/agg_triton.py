from typing import Any

import torch

from cudf_polars.triton.components.base_triton import ExecutionContext, TritonExpr
from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame

__all__ = ["TritonAgg"]


class TritonAgg(TritonExpr):
    """Class representing an aggregation of an expression."""

    __slots__ = ("context", "name", "options")
    _non_child = ("dtype", "name", "options", "context")

    def __init__(
        self,
        dtype: torch.dtype,
        name: str,
        options: Any,
        context: ExecutionContext,
        *children: TritonExpr,
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.context = context
        self.children = children
        self.is_pointwise = False

    def do_evaluate(
        self,
        df: TritonDataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> TritonColumn:
        """Evaluate this aggregation on the child tensor."""
        if context is not ExecutionContext.FRAME:
            raise NotImplementedError(f"Agg in context {context}")

        child = self.children[0].evaluate(df, context=context)
        values = child.obj
        valid = child.validity

        if valid is not None:
            valid_values = values[valid]
        else:
            valid_values = values

        if self.name == "count":
            result = torch.tensor(
                [values.numel() if self.options else valid_values.numel()],
                device=values.device,
                dtype=self.dtype,
            )
            return TritonColumn(result, dtype=self.dtype)

        if self.name == "sum":
            result = torch.zeros(1, device=values.device, dtype=self.dtype)
            if valid_values.numel() > 0:
                result[0] = valid_values.sum().to(self.dtype)
            return TritonColumn(result, dtype=self.dtype)

        if self.name == "first" or self.name == "last":
            index = 0 if self.name == "first" else -1
            if values.numel() == 0:
                return TritonColumn(
                    torch.zeros(1, device=values.device, dtype=self.dtype),
                    dtype=self.dtype,
                    validity=torch.zeros(1, device=values.device, dtype=torch.bool),
                )
            result = values[index].reshape(1).to(self.dtype)
            result_validity = (
                None
                if valid is None
                else valid[index].reshape(1)
            )
            return TritonColumn(result, dtype=self.dtype, validity=result_validity)

        if valid_values.numel() == 0:
            return TritonColumn(
                torch.zeros(1, device=values.device, dtype=self.dtype),
                dtype=self.dtype,
                validity=torch.zeros(1, device=values.device, dtype=torch.bool),
            )

        if self.name == "min":
            result = valid_values.min().reshape(1)
        elif self.name == "max":
            result = valid_values.max().reshape(1)
        elif self.name == "mean":
            result = valid_values.to(torch.float64).mean().reshape(1)
        elif self.name == "median":
            result = valid_values.to(torch.float64).median().reshape(1)
        elif self.name == "std":
            if valid_values.numel() <= self.options:
                return TritonColumn(
                    torch.zeros(1, device=values.device, dtype=self.dtype),
                    dtype=self.dtype,
                    validity=torch.zeros(1, device=values.device, dtype=torch.bool),
                )
            result = valid_values.to(torch.float64).std(
                correction=self.options
            ).reshape(1)
        elif self.name == "var":
            if valid_values.numel() <= self.options:
                return TritonColumn(
                    torch.zeros(1, device=values.device, dtype=self.dtype),
                    dtype=self.dtype,
                    validity=torch.zeros(1, device=values.device, dtype=torch.bool),
                )
            result = valid_values.to(torch.float64).var(
                correction=self.options
            ).reshape(1)
        elif self.name == "quantile":
            quantile = self.children[1].value
            result = torch.quantile(
                valid_values.to(torch.float64),
                quantile,
                interpolation=self.options,
            ).reshape(1)
        elif self.name == "n_unique":
            result = torch.unique(valid_values).numel()
            if self.options and valid is not None:
                result += int((~valid).any().item())
            result = torch.tensor([result], device=values.device, dtype=self.dtype)
        else:
            raise NotImplementedError(f"Unimplemented aggregation {self.name=}")

        return TritonColumn(result.to(self.dtype), dtype=self.dtype)


class TritonLen(TritonExpr):
    """Expression returning the number of rows in the input dataframe."""

    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.children = ()
        self.is_pointwise = False

    def do_evaluate(
        self,
        df: TritonDataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> TritonColumn:
        return TritonColumn(
            torch.tensor([df.num_rows], device=df.stream.device, dtype=self.dtype),
            dtype=self.dtype,
        )