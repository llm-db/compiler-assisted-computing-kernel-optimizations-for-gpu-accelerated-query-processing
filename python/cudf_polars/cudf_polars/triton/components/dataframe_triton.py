from collections.abc import Iterable
from functools import cached_property
from typing import Self, Set, cast

from cudf_polars.containers.dataframe import DataFrame
from cudf_polars.triton.components.column_triton import TritonColumn, NamedColumnTriton
import pylibcudf as plc
import numpy as np
import torch

from cudf_polars.utils import conversion
from cudf_polars.typing import Slice
from cudf_polars.triton.components.boolOp_triton import boolean_operation
from cudf_polars.triton.triton_operations.filter_triton_operation import masked_gather

class TritonDataFrame:
    cudfFallback: DataFrame

    column_map: dict[str, TritonColumn]
    columns: list[NamedColumnTriton]
    _num_rows_override: int | None
    stream: torch.Stream

    def __init__(
        self, columns: Iterable[TritonColumn], stream: torch.Stream, num_rows: int | None = None
    ) -> None:
        columns = list(columns)
        self.stream = stream
        if any(c.name is None for c in columns):
            raise ValueError("All columns must have a name")
        self.columns = [cast(NamedColumnTriton, c) for c in columns]
        self.dtypes = [c.dtype for c in self.columns]
        self.column_map = {c.name: c for c in self.columns}
        # print("!!!!!!!!!!!!!!!!!")
        # for c in self.columns:
        #     print(c.obj)

        # self.table = np.asarray([c.obj for c in self.columns], dtype=object)
        self._num_rows_override = num_rows

    # @classmethod
    # def from_cudf(cls, df: DataFrame): #, stream: Stream) -> Self:
    #     """
    #     Create from a cudf dataframe.

    #     Parameters
    #     ----------
    #     df
    #         Cudf dataframe to convert
    #     Returns
    #     -------
    #     New dataframe representing the input.
    #     """
    #     arrow_columns = [
    #         col.obj.to_arrow(stream=df.stream) for col in df.columns
    #     ]
        
    #     # Reconstruct the PyArrow Table
    #     import pyarrow as pa
    #     arrow_table = pa.Table.from_arrays(
    #         arrow_columns, 
    #         names=[c.name for c in df.columns]
    #     )

    #     # Extract NumPy arrays for TritonDataFrame
    #     triton_cols = [
    #         # ColumnTriton(column=torch.tensor(col.to_numpy()), name=name, dtype=c.dtype)
    #         ColumnTriton(column=col, name=name, dtype=c.dtype)
    #         for col, name, c in zip(arrow_table.columns, df.column_map.keys(), df.columns)
    #     ]
        
    #     return cls(triton_cols, num_rows=df._num_rows_override)
    


    @property
    def num_rows(self) -> int:
        if self._num_rows_override is not None:
            return self._num_rows_override

        if not self.columns:
            return 0

        return len(self.columns[0].obj)
    
    def filter(self, mask: TritonColumn):
        """
        Return a filtered dataframe given a boolean mask.
        """
        # if(mask.validity != None):
        #     raise NotImplementedError("Did not expect validity values for masks")
        assert mask.obj.dtype == torch.bool

        mask_tensor = mask.obj
        mask_tensor = boolean_operation(mask_tensor, mask.validity, "and", stream=self.stream)
        mask_indices = torch.nonzero(mask_tensor).flatten()

        new_columns = []
        for c in self.columns:
            column_obj_tensr = masked_gather(c.obj, mask_indices, self.stream)
            column_validity_tensor = None
            column_validity_tensor = masked_gather(c.validity, mask_indices, self.stream)

            new_columns.append(
                TritonColumn(
                    column=column_obj_tensr,
                    validity=column_validity_tensor,
                    dtype=c.dtype,
                    name=c.name,
                )
            )

        return type(self)(new_columns, self.stream)
        # table = apply_boolean_mask_triton(
        #     [c.obj for c in self.columns],
        #     mask.obj,
        # )

        # validities = None
        # if any(column.validity is not None for column in self.columns):
        #     validities = apply_boolean_mask_triton(
        #         [
        #             column.validity
        #             if column.validity is not None
        #             else torch.ones_like(column.obj, dtype=torch.bool)
        #             for column in self.columns
        #         ],
        #         mask.obj,
        #     )

        # columns = [
        #     ColumnTriton(
        #         column=column,
        #         validity=validity,
        #         dtype=original.dtype,
        #         name=original.name,
        #     )
        #     for column, original, validity in zip(
        #         table,
        #         self.columns,
        #         validities if validities is not None else [None] * len(self.columns),
        #     )
        # ]

        # return type(self)(columns)

    def slice(self, zlice: Slice | None):
        """
        Slice a dataframe.

        Parameters
        ----------
        zlice
            optional, tuple of start and length, negative values of start
            treated as for python indexing. If not provided, returns self.

        Returns
        -------
        New dataframe (if zlice is not None) otherwise self (if it is)
        """
        if zlice is None:
            return self

        start, end = conversion.from_polars_slice(zlice, num_rows=self.num_rows)
        data_slice = slice(start, end if end is not None else None)
        sliced_columns = [
            TritonColumn(
                column=col.obj[data_slice],
                validity=None if col.validity is None else col.validity[data_slice],
                dtype=col.dtype,
                is_sorted=col.is_sorted,
                order=col.order,
                null_order=col.null_order,
                name=col.name,
            )
            for col in self.columns
        ]
        print("HIIII")
        for c in sliced_columns:
            print(c.obj)

        return type(self)(sliced_columns, self.stream)

    @cached_property
    def column_names_set(self) -> frozenset[str]:
        """Return the column names as a set."""
        return frozenset(self.column_map)

    @cached_property
    def column_names(self) -> list[str]:
        """Return a list of the column names."""
        return list(self.column_map)

    @cached_property
    def num_columns(self) -> int:
        """Number of columns."""
        return len(self.column_map)

    def discard_columns(self, names: Set[str]) -> Self:
        """Drop columns by name."""
        return type(self)(
            (column for column in self.columns if column.name not in names),
            stream=self.stream,
        )
    
    def select_columns(self, names: Set[str]) -> list[TritonColumn]:
        """Select columns by name."""
        return [c for c in self.columns if c.name in names]

    @staticmethod
    def _format_table(
        headers: list[str],
        rows: list[list],
        *,
        indent: str = "",
        max_rows: int | None = None,
    ) -> list[str]:
        """Format a bounded ASCII table."""

        total_rows = len(rows)

        if max_rows is not None and total_rows > max_rows:
            # Keep the beginning and end visible.
            head = max_rows // 2
            tail = max_rows - head

            omitted = total_rows - head - tail

            rows = (
                rows[:head]
                + [[f"... ({omitted} rows omitted)"] + [""] * (len(headers) - 1)]
                + rows[-tail:]
            )

        rows = [
            row[: len(headers)] + [""] * max(0, len(headers) - len(row))
            for row in rows
        ]

        widths = [
            max([len(str(header))] + [len(str(row[i])) for row in rows])
            for i, header in enumerate(headers)
        ]

        def format_row(row: list) -> str:
            return (
                "| "
                + " | ".join(
                    str(value).ljust(widths[i])
                    for i, value in enumerate(row)
                )
                + " |"
            )

        separator = (
            "+"
            + "+".join("-" * (width + 2) for width in widths)
            + "+"
        )
        return [
            indent + separator,
            indent + format_row(headers),
            indent + separator,
            *(indent + format_row(row) for row in rows),
            indent + separator,
        ]
    
    def __repr__(self) -> str:
        """Return a human-readable representation of the dataframe."""

        lines: list[str] = []

        # ------------------------------------------------------------------
        # Column summary
        # ------------------------------------------------------------------
        summary_headers = [
            "column",
            "dtype",
            "size",
            "validity",
            "sorted",
            "order",
            "null_order",
            "scalar",
        ]

        summary_rows = []

        for col in self.columns:
            summary_rows.append(
                [
                    col.name,
                    str(col.dtype),
                    col.size,
                    "yes" if col.validity is not None else "no",
                    str(col.is_sorted),
                    str(col.order),
                    str(col.null_order),
                    "yes" if col.is_scalar else "no",
                ]
            )

        lines.append("Column summary:")
        lines.extend(
            self._format_table(
                summary_headers,
                summary_rows,
                indent="  ",
            )
        )
        lines.append(self.table_to_str())
        
        return "\n".join(lines)

    def table_to_str(self):
        # ------------------------------------------------------------------
        # Values
        #
        # Each dataframe column is represented by one row:
        #
        # column | row 0 | row 1 | row 2 | ...
        # ------------------------------------------------------------------
        lines: list[str] = []
        max_rows = min(self.num_rows, 20)

        value_headers = ["column"] + [str(i) for i in range(max_rows)]
        value_rows = []

        for col in self.columns:
            values = col.obj[:max_rows]

            # Convert tensors to Python values. This works for CUDA tensors
            # as well, although it necessarily synchronizes/copies them.
            values = values.detach().cpu().tolist()

            if col.validity is not None:
                validity = col.validity[:max_rows].detach().cpu().tolist()
                values = [
                    value if valid else None
                    for value, valid in zip(values, validity)
                ]

            value_rows.append([col.name, *values])

        lines.append("")
        lines.append(
            f"Values (showing first {max_rows} of {self.num_rows} rows):"
        )
        lines.extend(
            self._format_table(
                value_headers,
                value_rows,
                indent="  ",
            )
        )

        return "\n".join(lines)
        