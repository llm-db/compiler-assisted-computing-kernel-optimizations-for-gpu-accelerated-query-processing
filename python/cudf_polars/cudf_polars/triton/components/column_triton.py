from collections.abc import Iterable
from typing import Any

import numpy as np
import pylibcudf as plc
import torch


class TritonColumn:
    obj: torch.Tensor
    validity: torch.Tensor
    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    is_scalar: bool

    # Optional name, only ever set by evaluation of NamedExpr nodes
    # The internal evaluation should not care about the name.
    name: str | None
    dtype: torch.dtype

    def __init__(
        self,
        column: Iterable[Any] | torch.Tensor,
        dtype: torch.dtype,
        *,
        validity: Iterable[Any] | torch.Tensor | None = None,
        is_sorted: plc.types.Sorted = plc.types.Sorted.NO,
        order: plc.types.Order = plc.types.Order.ASCENDING,
        null_order: plc.types.NullOrder = plc.types.NullOrder.BEFORE,
        name: str | None = None,
    ):
        # Converts any non-tensor iterable to a tensor without copying if already a tensor
        if not isinstance(column, torch.Tensor):
            self.obj = torch.as_tensor(column)
        else:
            self.obj = column

        if validity is None:
            self.validity = torch.ones_like(self.obj, device=column.device, dtype=torch.bool)
        elif isinstance(validity, torch.Tensor):
            self.validity = validity
        else:
            self.validity = torch.as_tensor(validity)

        self.is_scalar = self.size == 1
        self.name = name
        self.dtype = dtype
        self.set_sorted(is_sorted=is_sorted, order=order, null_order=null_order)

    @property
    def size(self) -> int:
        """Return the size of the column."""
        return len(self.obj)

    def obj_scalar(self):
        """
        A copy of the column object as a pylibcudf Scalar.

        Parameters
        ----------

        Returns
        -------
        pylibcudf Scalar object.

        Raises
        ------
        ValueError
            If the column is not length-1.
        """
        if not self.is_scalar:
            raise ValueError(f"Cannot convert a column of length {self.size} to scalar")
        return self.obj[0]


    def set_sorted(
        self,
        *,
        is_sorted: plc.types.Sorted,
        order: plc.types.Order,
        null_order: plc.types.NullOrder,
    ):
        """
        Modify sortedness metadata in place.

        Parameters
        ----------
        is_sorted
            Is the column sorted
        order
            The order if sorted
        null_order
            Where nulls sort, if sorted

        Returns
        -------
        Self with metadata set.
        """
        if self.size <= 1:
            is_sorted = plc.types.Sorted.YES
        self.is_sorted = is_sorted
        self.order = order
        self.null_order = null_order
        return self

    def rename(self, name: str | None, /):
        """
        Return a shallow copy with a new name.

        Parameters
        ----------
        name
            New name

        Returns
        -------
        Shallow copy of self with new name set.
        """
        new = self.copy()
        new.name = name
        return new

    
    def copy(self):
        """
        A shallow copy of the column.

        Returns
        -------
        New column sharing data with self.
        """
        return type(self)(
            self.obj,
            validity=self.validity,
            is_sorted=self.is_sorted,
            order=self.order,
            null_order=self.null_order,
            name=self.name,
            dtype=self.dtype,
        )

class NamedColumnTriton(TritonColumn):
    name: str