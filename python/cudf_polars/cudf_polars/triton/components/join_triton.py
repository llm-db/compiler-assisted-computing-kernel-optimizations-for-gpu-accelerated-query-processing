

from typing import Callable

import torch

from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.typing import Slice as Zlice
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame
from cudf_polars.triton.components.base_triton import TritonColRef, TritonExpr, TritonNamedExpr
from cudf_polars.triton.components.literal_triton import TritonLiteral


def join_gather_column(
    column: TritonColumn, indices: torch.Tensor, *, nullify: torch.Tensor | None = None
) -> TritonColumn:
    """Gather a column, optionally marking selected rows null."""
    safe_indices = indices.clamp_min(0)
    values = column.obj.index_select(0, safe_indices)
    validity = column.validity.index_select(0, safe_indices).clone()
    
    if nullify is not None:
        validity = validity & nullify
    return TritonColumn(values, dtype=column.dtype, validity=validity, name=column.name)

# def join_conditional_operand(
#     node: TritonExpr,
#     left: TritonDataFrame,
#     right: TritonDataFrame,
# ) -> TritonColumn:
#     """Evaluate one conditional-join expression over every row pair."""
#     n_left, n_right = left.num_rows, right.num_rows
#     device = left.stream.device
#     if isinstance(node, TritonColRef):
#         (column,) = node.children
#         assert isinstance(column, TritonColumn)
#         frame = (
#             left
#             if node.table_ref.name == "LEFT"
#             else right
#         )
#         source = frame.column_map[column.name]
#         values = (
#             source.obj[:, None].expand(n_left, n_right)
#             if frame is left
#             else source.obj[None, :].expand(n_left, n_right)
#         )
#         valid = (
#             torch.ones(values.shape, device=device, dtype=torch.bool)
#             if source.validity is None
#             else (
#                 source.validity[:, None].expand(n_left, n_right)
#                 if frame is left
#                 else source.validity[None, :].expand(n_left, n_right)
#             )
#         )
#         return values, valid
#     if isinstance(node, TritonLiteral):
#         return (
#             torch.full((n_left, n_right), node.value, device=device, dtype=dtype_cudf_to_triton(node.dtype)),
#             torch.ones((n_left, n_right), device=device, dtype=torch.bool),
#         )
#     if isinstance(node, cudf_expr.BinOp):
#         left_value, left_valid = _conditional_operand(node.children[0], left, right)
#         right_value, right_valid = _conditional_operand(node.children[1], left, right)
#         op = node.op.name
#         if op == "EQUAL":
#             value = left_value == right_value
#         elif op == "NOT_EQUAL":
#             value = left_value != right_value
#         elif op == "LESS":
#             value = left_value < right_value
#         elif op == "LESS_EQUAL":
#             value = left_value <= right_value
#         elif op == "GREATER":
#             value = left_value > right_value
#         elif op == "GREATER_EQUAL":
#             value = left_value >= right_value
#         elif op in {"ADD", "SUB", "MUL", "DIV", "TRUE_DIV", "FLOOR_DIV", "PYMOD"}:
#             value = {
#                 "ADD": left_value + right_value,
#                 "SUB": left_value - right_value,
#                 "MUL": left_value * right_value,
#                 "DIV": left_value / right_value,
#                 "TRUE_DIV": left_value / right_value,
#                 "FLOOR_DIV": torch.floor_divide(left_value, right_value),
#                 "PYMOD": torch.remainder(left_value, right_value),
#             }[op]
#         elif op in {"BITWISE_AND", "NULL_LOGICAL_AND", "LOGICAL_AND"}:
#             value = left_value & right_value
#         elif op in {"BITWISE_OR", "NULL_LOGICAL_OR", "LOGICAL_OR"}:
#             value = left_value | right_value
#         else:
#             raise NotImplementedError(f"Conditional Triton operator {op}")
#         return value, left_valid & right_valid
#     if isinstance(node, cudf_expr.Cast):
#         (child,) = node.children
#         values, valid = _conditional_operand(child, left, right)
#         return values.to(dtype_cudf_to_triton(node.dtype)), valid
#     raise NotImplementedError(f"Conditional Triton expression {type(node).__name__}")

def equal_match(left_key: TritonColumn, right_key: TritonColumn, nulls_equal: bool,) -> torch.Tensor: 
    equal = left_key.obj[:, None] == right_key.obj[None, :] 
    left_valid = left_key.validity 
    right_valid = right_key.validity 
    both_valid = left_valid[:, None] & right_valid[None, :] 

    if nulls_equal: 
        return ( (equal & both_valid) | (~left_valid[:, None] & ~right_valid[None, :]) ) 
    return equal & both_valid

def generate_dataframe_matches(
    left_keys: TritonDataFrame,
    right_keys: TritonDataFrame,
    nulls_equal: bool,
    match_fn: Callable[[TritonDataFrame, TritonDataFrame], torch.Tensor],
    *,
    device: torch.device
) -> torch.Tensor:
    n_left = left_keys.num_rows
    n_right = right_keys.num_rows
    matches = torch.ones((n_left, n_right), device=device, dtype=torch.bool)
    for left_key, right_key in zip(left_keys, right_keys, strict=True):
        matches &= match_fn(left_key, right_key, nulls_equal)
    return matches

def generate_column_matches(
    left_keys: list[TritonColumn],
    right_keys: list[TritonColumn],
    nulls_equal: bool,
    match_fn: Callable[[TritonColumn, TritonColumn], torch.Tensor],
    *,
    device: torch.device,
    n_left: int,
    n_right: int,
) -> torch.Tensor:
    matches = torch.ones((n_left, n_right), device=device, dtype=torch.bool)
    for left_key, right_key in zip(left_keys, right_keys, strict=True):
        matches &= match_fn(left_key, right_key, nulls_equal)
    return matches



def join_pairs(
    left_keys: list[TritonColumn],
    right_keys: list[TritonColumn],
    how: str,
    nulls_equal: bool,
    *,
    device: torch.device,
    n_left: int,
    n_right: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left/right row pairs for a join."""
    if how == "Cross":
        return (
            torch.arange(n_left, device=device).repeat_interleave(n_right),
            torch.arange(n_right, device=device).repeat(n_left),
        )

    matches = generate_column_matches(
        left_keys,
        right_keys,
        nulls_equal,
        match_fn=equal_match,
        device=device,
        n_left=n_left,
        n_right=n_right
    )

    matched_left, matched_right = torch.nonzero(matches, as_tuple=True)
    if how == "Inner":
        return matched_left, matched_right
    if how == "Semi":
        return torch.nonzero(matches.any(dim=1), as_tuple=True)[0], torch.empty(
            0, device=device, dtype=torch.long
        )
    if how == "Anti":
        return torch.nonzero(~matches.any(dim=1), as_tuple=True)[0], torch.empty(
            0, device=device, dtype=torch.long
        )

    if how == "Left":
        unmatched = torch.nonzero(~matches.any(dim=1), as_tuple=True)[0]
        return torch.cat((matched_left, unmatched)), torch.cat(
            (matched_right, torch.full_like(unmatched, -1))
        )
    if how == "Right":
        unmatched = torch.nonzero(~matches.any(dim=0), as_tuple=True)[0]
        return torch.cat((matched_left, torch.full_like(unmatched, -1))), torch.cat(
            (matched_right, unmatched)
        )
    if how == "Full":
        unmatched_left = torch.nonzero(~matches.any(dim=1), as_tuple=True)[0]
        unmatched_right = torch.nonzero(~matches.any(dim=0), as_tuple=True)[0]
        return (
            torch.cat((matched_left, unmatched_left, torch.full_like(unmatched_right, -1))),
            torch.cat((matched_right, torch.full_like(unmatched_left, -1), unmatched_right)),
        )
    raise AssertionError(f"Unsupported join type: {how}")

def join_coalesce(
    left: TritonDataFrame,
    right: TritonDataFrame,
    right_on: tuple[TritonNamedExpr, ...], 
    left_on: tuple[TritonNamedExpr, ...],
    options: tuple[str, bool, Zlice | None, str, bool, str],
    left_columns: TritonColumn,
    right_columns: TritonColumn
) -> tuple[TritonColumn, TritonColumn]:
    how, _, _, suffix, _, _ = options
    key_names = {key.name for key in right_on}
    left_key_names = {key.name for key in left_on}
    if how == "Full":
        right_by_name = {column.name: column for column in right_columns}
        for left_column in left_columns:
            if left_column.name not in left_key_names:
                continue
            right_column = right_by_name.get(left_column.name) or right_by_name.get(
                f"{left_column.name}{suffix}"
            )
            if right_column is None:
                continue
            left_valid = (
                torch.ones(left_column.size, device=left.stream.device, dtype=torch.bool)
                if left_column.validity is None
                else left_column.validity
            )
            right_valid = (
                torch.ones(right_column.size, device=left.stream.device, dtype=torch.bool)
                if right_column.validity is None
                else right_column.validity
            )
            left_column.obj = torch.where(left_valid, left_column.obj, right_column.obj)
            left_column.validity = left_valid | right_valid
    elif how == "Right":
        right_by_name = {column.name: column for column in right_columns}
        for left_column in left_columns:
            if left_column.name not in left_key_names:
                continue
            right_column = right_by_name.get(left_column.name) or right_by_name.get(
                f"{left_column.name}{suffix}"
            )
            if right_column is None:
                continue
            left_valid = (
                torch.ones(left_column.size, device=left.stream.device, dtype=torch.bool)
                if left_column.validity is None
                else left_column.validity
            )
            right_valid = (
                torch.ones(right_column.size, device=left.stream.device, dtype=torch.bool)
                if right_column.validity is None
                else right_column.validity
            )
            left_column.obj = torch.where(left_valid, left_column.obj, right_column.obj)
            left_column.validity = left_valid | right_valid
    right_columns = [
        column
        for column in right_columns
        if column.name not in key_names
        and column.name not in {f"{name}{suffix}" for name in key_names}
    ]
    if how == "Right":
        key_names = {key.name for key in left_on}
        left_columns = [
            *[column for column in left_columns if column.name not in key_names],
            *[column for column in left_columns if column.name in key_names],
        ]

    return left_columns, right_columns

def join_gather(
    left: TritonDataFrame,
    right: TritonDataFrame,
    options: tuple[str, bool, Zlice | None, str, bool, str],
    left_indices: tuple[torch.Tensor, torch.Tensor],
    right_indices: tuple[torch.Tensor, torch.Tensor]
) -> tuple[TritonColumn, TritonColumn]:
    
    _, _, _, suffix, _, maintain_order = options

    if maintain_order != "none" and left_indices.numel() > 1:
        primary, secondary = (
            (left_indices, right_indices)
            if maintain_order.startswith("left")
            else (right_indices, left_indices)
        )
        primary = primary.masked_fill(primary < 0, torch.iinfo(primary.dtype).max)
        secondary = secondary.masked_fill(secondary < 0, torch.iinfo(secondary.dtype).max)
        order = torch.argsort(secondary, stable=True)
        order = order[torch.argsort(primary[order], stable=True)]
        left_indices, right_indices = left_indices[order], right_indices[order]

    left_present = left_indices >= 0
    right_present = right_indices >= 0
    left_columns = [
        join_gather_column(column, left_indices, nullify=left_present).rename(column.name)
        for column in left.columns
    ]
    right_columns = [
        join_gather_column(
            column, right_indices, nullify=right_present
        ).rename(column.name if column.name not in left.column_names_set else f"{column.name}{suffix}")
        for column in right.columns
    ]
    return left_columns, right_columns
