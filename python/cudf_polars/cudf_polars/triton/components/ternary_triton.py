import torch

from cudf_polars.triton.components.base_triton import ExecutionContext, TritonExpr
from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame

__all__ = ["TritonTernary"]


class TritonTernary(TritonExpr):
    """Class representing a conditional expression."""

    __slots__ = ()
    _non_child = ("dtype",)

    def __init__(
        self,
        dtype: torch.dtype,
        when: TritonExpr,
        then: TritonExpr,
        otherwise: TritonExpr,
    ) -> None:
        self.dtype = dtype
        self.children = (when, then, otherwise)
        self.is_pointwise = True

    def do_evaluate(
        self,
        df: TritonDataFrame,
        *,
        context: ExecutionContext = ExecutionContext.FRAME,
    ) -> TritonColumn:
        """Evaluate this conditional expression."""
        when, then, otherwise = (
            child.evaluate(df, context=context) for child in self.children
        )

        if when.is_scalar:
            if not bool(when.validity[0].item()):
                size = max(then.size, otherwise.size)
                return TritonColumn(
                    torch.zeros(size, device=when.obj.device, dtype=self.dtype),
                    dtype=self.dtype,
                    validity=torch.zeros(
                        size, device=when.obj.device, dtype=torch.bool
                    ),
                )

            pick = then if bool(when.obj[0].item()) else otherwise
            size = max(then.size, otherwise.size)
            values = pick.obj if pick.size == size else pick.obj.expand(size)
            validity = pick.validity
            if validity.size != size:
                validity = validity.expand(size)
            return TritonColumn(
                values.to(self.dtype), dtype=self.dtype, validity=validity
            )

        size = max(when.size, then.size, otherwise.size)

        def expand(values: torch.Tensor, length: int) -> torch.Tensor:
            return values if values.numel() == length else values.expand(length)

        condition = expand(when.obj, size).bool()
        then_values = expand(then.obj, size)
        otherwise_values = expand(otherwise.obj, size)
        values = torch.where(condition, then_values, otherwise_values).to(self.dtype)

        validity = None
        when_validity = expand(when.validity, size).bool()
        then_validity = expand(then.validity, size).bool()
        otherwise_validity = expand(otherwise.validity, size).bool()
        validity = when_validity & torch.where(
            condition, then_validity, otherwise_validity
        )

        return TritonColumn(values, dtype=self.dtype, validity=validity)