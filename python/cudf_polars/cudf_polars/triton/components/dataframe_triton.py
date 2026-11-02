from collections.abc import Iterable
from functools import cached_property
from typing import Self, Set, cast

from cudf_polars.containers.dataframe import DataFrame
from cudf_polars.triton.components.column_triton import TritonColumn, NamedColumnTriton
import pylibcudf as plc
import numpy as np
import torch
import cupy as cp
from rmm.pylibrmm.stream import Stream
from cudf_polars.utils import conversion
from cudf_polars.typing import Slice
from cudf_polars.triton.components.boolOp_triton import boolean_operation

class TritonDataFrame:
    cudfFallback: DataFrame | None

    column_map: dict[str, TritonColumn]
    columns: list[NamedColumnTriton]
    _num_rows_override: int | None
    stream: torch.Stream
    cudf_stream: Stream

    def __init__(
        self, columns: Iterable[TritonColumn], cudf_stream: Stream, num_rows: int | None = None
    ) -> None:
        if not isinstance(cudf_stream, Stream):
            raise ValueError("Should be cudf stream")
        columns = list(columns)

        self.cudf_stream = cudf_stream
        version, stream_ptr = cudf_stream.__cuda_stream__()
        assert version == 0

        self.stream = torch.cuda.ExternalStream(
            stream_ptr,
            device=torch.cuda.current_device(),
        )
        if any(c.name is None for c in columns):
            raise ValueError("All columns must have a name")
        self.columns = [cast(NamedColumnTriton, c) for c in columns]
        self.dtypes = [c.dtype for c in self.columns]
        self.column_map = {c.name: c for c in self.columns}
        self._num_rows_override = num_rows
        self.cudfFallback = None

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

        return self.columns[0].size
    
    def filter(self, mask: TritonColumn):
        """
        Return a filtered dataframe given a boolean mask.
        """
        # if(mask.validity != None):
        #     raise NotImplementedError("Did not expect validity values for masks")
        assert mask.obj.dtype == torch.bool

        with torch.cuda.stream(self.stream):
            mask_tensor = boolean_operation(
                mask.obj, mask.validity, "and", stream=self.stream
            )
            mask_column = plc.Column.from_array(
                cp.asarray(mask_tensor), stream=self.cudf_stream
            )
            input_columns = []
            for c in self.columns:
                if isinstance(c.obj, plc.Column):
                    column_obj = c.obj
                else:
                    column_obj = plc.Column.from_array(
                        cp.asarray(c.obj), stream=self.cudf_stream
                    )
                if not c.is_all_valid:
                    validity_column = plc.Column.from_array(
                        cp.asarray(c.validity), stream=self.cudf_stream
                    )
                    validity_mask, null_count = plc.transform.bools_to_mask(
                        validity_column, stream=self.cudf_stream
                    )
                    column_obj = column_obj.with_mask(validity_mask, null_count)
                input_columns.append(column_obj)

            filtered_table = plc.stream_compaction.apply_boolean_mask(
                plc.Table(input_columns),
                mask_column,
                stream=self.cudf_stream,
            )
            new_columns = []
            for c, column_obj in zip(
                self.columns, filtered_table.columns(), strict=True
            ):
                if column_obj.null_mask() is None:
                    new_columns.append(
                        TritonColumn(column_obj, dtype=c.dtype, name=c.name)
                    )
                else:
                    from cudf_polars.containers import Column
                    from cudf_polars.triton.triton_conversion import (
                        column_cudf_to_triton,
                        dtype_triton_to_cudf,
                    )

                    new_columns.append(
                        column_cudf_to_triton(
                            Column(
                                column_obj,
                                dtype=(
                                    c.dtype
                                    if hasattr(c.dtype, "plc_type")
                                    else dtype_triton_to_cudf(c.dtype)
                                ),
                            ),
                            name=c.name,
                            stream=self.cudf_stream,
                        )
                    )
            return type(self)(new_columns, self.cudf_stream)

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
                column=(
                    plc.copying.slice(
                        plc.Table([col.obj]),
                        [start, (end if end is not None else col.size)],
                        stream=self.cudf_stream,
                    )[0].columns()[0]
                    if isinstance(col.obj, plc.Column)
                    else col.obj[data_slice]
                ),
                validity=None if col.validity is None else col.validity[data_slice],
                dtype=col.dtype,
                is_sorted=col.is_sorted,
                order=col.order,
                null_order=col.null_order,
                name=col.name,
            )
            for col in self.columns
        ]

        return type(self)(sliced_columns, self.cudf_stream)

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

    def with_columns(
            self,
            columns: Iterable[TritonColumn],
            *,
            replace_only: bool = False,
            stream: torch.Stream,
        ) -> Self:
            """
            Return a new dataframe with extra columns.
    
            Parameters
            ----------
            columns
                Columns to add
            replace_only
                If true, then only replacements are allowed (matching by name).
            stream
                CUDA stream used for device memory operations and kernel launches.
                The caller is responsible for ensuring that
    
                1. The data in ``columns`` is valid on ``stream``.
                2. No additional operations occur on ``self.stream`` with the
                   original data in ``self``.
    
            Returns
            -------
            New dataframe
    
            Notes
            -----
            If column names overlap, newer names replace older ones, and
            appear in the same order as the original frame.
            """
            new = {c.name: c for c in columns}
            if replace_only and not self.column_names_set.issuperset(new.keys()):
                raise ValueError("Cannot replace with non-existing names")
            return type(self)((self.column_map | new).values(), self.cudf_stream)

    def discard_columns(self, names: Set[str]) -> Self:
        """Drop columns by name."""
        return type(self)(
            (column for column in self.columns if column.name not in names),
            self.cudf_stream,
        )

    def sorted_like(
        self, like, /, *, subset: Set[str] | None = None
    ) -> Self:
        """
        Return a shallow copy with sortedness copied from like.

        Parameters
        ----------
        like
            The dataframe to copy from
        subset
            Optional subset of columns from which to copy data.

        Returns
        -------
        Shallow copy of self with metadata set.

        Raises
        ------
        ValueError
            If there is a name mismatch between self and like.
        """
        if like.column_names != self.column_names:
            raise ValueError("Can only copy from identically named frame")
        subset = self.column_names_set if subset is None else subset
        return type(self)(
            (
                c.sorted_like(other) if c.name in subset else c
                for c, other in zip(self.columns, like.columns, strict=True)
            ),
            stream=self.cudf_stream,
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
            if isinstance(col.obj, torch.Tensor):
                # Convert tensors to Python values. This works for CUDA
                # tensors as well, although it necessarily synchronizes.
                values = col.obj[:max_rows].detach().cpu().tolist()
            else:
                values = col.obj.to_arrow(stream=self.cudf_stream).to_pylist()[
                    :max_rows
                ]

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
        