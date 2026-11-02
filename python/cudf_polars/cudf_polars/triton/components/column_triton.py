from collections.abc import Iterable
from typing import Any

import numpy as np
from polars import DataType
import pylibcudf as plc
import torch


class TritonColumn:
    # Tensor-backed columns are used by Triton operators. Other payloads (for
    # example libcudf list columns) are carried through without interpretation.
    obj: Any
    validity: torch.Tensor
    is_all_valid: bool
    is_sorted: plc.types.Sorted
    order: plc.types.Order
    null_order: plc.types.NullOrder
    is_scalar: bool

    # Optional name, only ever set by evaluation of NamedExpr nodes
    # The internal evaluation should not care about the name.
    name: str | None
    dtype: DataType | torch.dtype

    def __init__(
        self,
        column: Iterable[Any] | torch.Tensor | Any,
        dtype: DataType | torch.dtype,
        *,
        validity: Iterable[Any] | torch.Tensor | None = None,
        is_sorted: plc.types.Sorted = plc.types.Sorted.NO,
        order: plc.types.Order = plc.types.Order.ASCENDING,
        null_order: plc.types.NullOrder = plc.types.NullOrder.BEFORE,
        name: str | None = None,
    ):
        if isinstance(column, torch.Tensor):
            self.obj = column
        elif isinstance(column, Iterable) and not hasattr(column, "size"):
            self.obj = torch.as_tensor(column)
        else:
            # Keep non-tensor column representations (such as nested libcudf
            # columns) opaque until an operator explicitly supports them.
            self.obj = column

        if validity is None:
            self.is_all_valid = True
            if isinstance(self.obj, torch.Tensor):
                self.validity = torch.ones_like(
                    self.obj, device=self.obj.device, dtype=torch.bool
                )
            else:
                self.validity = torch.ones(
                    self.size, dtype=torch.bool
                )
        elif isinstance(validity, torch.Tensor):
            self.is_all_valid = False
            self.validity = validity
        else:
            self.is_all_valid = False
            self.validity = torch.as_tensor(validity)

        self.is_scalar = self.size == 1
        self.name = name
        self.dtype = dtype
        self.set_sorted(is_sorted=is_sorted, order=order, null_order=null_order)

    @property
    def size(self) -> int:
        """Return the size of the column."""
        size = getattr(self.obj, "size", None)
        if callable(size):
            size = size()
        if size is None and isinstance(self.obj, plc.Column):
            size = self.obj.size()
        if isinstance(size, torch.Size):
            return size[0]
        if size is not None:
            return size
        return len(self.obj)

    def obj_scalar(self, stream):
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
        # TODO: Handle stream
        if not self.is_scalar:
            raise ValueError(f"Cannot convert a column of length {self.size} to scalar")
        if not bool(self.validity[0].item()):
            return None
        if isinstance(self.obj, torch.Tensor):
            return self.obj[0].item()
        return self.obj.to_arrow(stream=stream).to_pylist()[0]


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