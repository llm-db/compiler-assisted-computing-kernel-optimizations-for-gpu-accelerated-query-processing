import torch
import triton
import triton.language as tl

from cudf_polars.triton.components.dataframe_triton import TritonDataFrame
from cudf_polars.triton.components.column_triton import TritonColumn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from python.cudf_polars.cudf_polars.dsl.ir import IRExecutionContext



def merge_dataframes(
    dataframes_to_merge: list[TritonDataFrame],
    key: str,
    stream: torch.Stream | None = None,
) -> TritonDataFrame:
    key_cols = [df.column_names.index(key) for df in dataframes_to_merge]
    selected_cols = [df.select_columns({key})[0] for df in dataframes_to_merge]
    column_order = [c.order for c in selected_cols]
    null_precedence = [c.null_order for c in selected_cols]




@triton.jit
def _merge_indices_kernel(
    a_ptr,
    a_valid_ptr,
    b_ptr,
    b_valid_ptr,
    out_src_ptr,
    out_idx_ptr,
    n_a,
    n_b,
    descending: tl.constexpr,
    nulls_after: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    out_mask = offs < n_a + n_b

    # We assign each output position independently.
    #
    # For simplicity, use a binary-search based merge:
    # find how many elements from B belong before A[i].
    #
    # This implementation is intended for numeric key types.

    # ------------------------------------------------------------
    # Find candidate A index.
    # ------------------------------------------------------------

    # For an output position p, we need to find the split:
    #
    #     i elements from A
    #     p-i elements from B
    #
    # such that the two sequences are correctly merged.
    #
    # Do a binary search independently for every output position.
    # ------------------------------------------------------------

    lo = tl.maximum(0, offs - n_b)
    hi = tl.minimum(offs, n_a)

    for _ in range(32):
        mid = (lo + hi) // 2
        j = offs - mid

        a_mask = mid < n_a
        b_mask = j < n_b

        a_val = tl.load(a_ptr + mid, mask=a_mask, other=0)
        b_val = tl.load(b_ptr + j, mask=b_mask, other=0)

        if a_valid_ptr is not None:
            a_valid = tl.load(a_valid_ptr + mid, mask=a_mask, other=0)
        else:
            a_valid = tl.full([BLOCK], 1, tl.int1)

        if b_valid_ptr is not None:
            b_valid = tl.load(b_valid_ptr + j, mask=b_mask, other=0)
        else:
            b_valid = tl.full([BLOCK], 1, tl.int1)

        # Compare A[mid] and B[j].
        #
        # We want to determine whether A[mid] belongs before B[j].
        a_is_null = ~a_valid
        b_is_null = ~b_valid

        if nulls_after:
            a_before_b = tl.where(
                a_is_null & ~b_is_null,
                False,
                tl.where(
                    ~a_is_null & b_is_null,
                    True,
                    a_val < b_val if not descending else a_val > b_val,
                ),
            )
        else:
            a_before_b = tl.where(
                a_is_null & ~b_is_null,
                True,
                tl.where(
                    ~a_is_null & b_is_null,
                    False,
                    a_val < b_val if not descending else a_val > b_val,
                ),
            )

        # We need a[mid] <= b[j] for the split to be valid.
        if descending:
            a_le_b = tl.where(
                a_is_null & ~b_is_null,
                nulls_after,
                tl.where(
                    ~a_is_null & b_is_null,
                    not nulls_after,
                    a_val >= b_val,
                ),
            )
        else:
            a_le_b = tl.where(
                a_is_null & ~b_is_null,
                nulls_after,
                tl.where(
                    ~a_is_null & b_is_null,
                    not nulls_after,
                    a_val <= b_val,
                ),
            )

        # Boundary checks.
        left_ok = mid == 0

        if descending:
            # B[j-1] <= A[mid]
            prev_j = j - 1
            prev_b_mask = (j > 0) & (prev_j < n_b)

            prev_b_val = tl.load(
                b_ptr + prev_j,
                mask=prev_b_mask,
                other=0,
            )

            if b_valid_ptr is not None:
                prev_b_valid = tl.load(
                    b_valid_ptr + prev_j,
                    mask=prev_b_mask,
                    other=0,
                )
            else:
                prev_b_valid = tl.full([BLOCK], 1, tl.int1)

            b_before_a = tl.where(
                prev_b_valid & ~a_valid,
                not nulls_after,
                tl.where(
                    ~prev_b_valid & a_valid,
                    nulls_after,
                    prev_b_val >= a_val,
                ),
            )
        else:
            prev_j = j - 1
            prev_b_mask = (j > 0) & (prev_j < n_b)

            prev_b_val = tl.load(
                b_ptr + prev_j,
                mask=prev_b_mask,
                other=0,
            )

            if b_valid_ptr is not None:
                prev_b_valid = tl.load(
                    b_valid_ptr + prev_j,
                    mask=prev_b_mask,
                    other=0,
                )
            else:
                prev_b_valid = tl.full([BLOCK], 1, tl.int1)

            b_before_a = tl.where(
                prev_b_valid & ~a_valid,
                not nulls_after,
                tl.where(
                    ~prev_b_valid & a_valid,
                    nulls_after,
                    prev_b_val <= a_val,
                ),
            )

        right_ok = mid == n_a

        # Binary search.
        move_right = (~right_ok) & (~a_le_b)
        move_left = (~left_ok) & (~b_before_a)

        lo = tl.where(move_right, mid + 1, lo)
        hi = tl.where(move_left, mid - 1, hi)

    i = lo
    j = offs - i

    # `i` elements came from A, so if i < current output position,
    # the output element comes from B.
    from_a = i > offs - 1

    # More directly: determine whether this output position is the
    # next A or B element.
    #
    # Compute the final partition around p.
    a_idx = tl.minimum(i, n_a - 1)
    b_idx = tl.minimum(j, n_b - 1)

    a_valid = tl.load(
        a_valid_ptr + a_idx,
        mask=a_idx < n_a,
        other=1,
    ) if a_valid_ptr is not None else tl.full([BLOCK], 1, tl.int1)

    b_valid = tl.load(
        b_valid_ptr + b_idx,
        mask=b_idx < n_b,
        other=1,
    ) if b_valid_ptr is not None else tl.full([BLOCK], 1, tl.int1)

    a_val = tl.load(
        a_ptr + a_idx,
        mask=a_idx < n_a,
        other=0,
    )

    b_val = tl.load(
        b_ptr + b_idx,
        mask=b_idx < n_b,
        other=0,
    )

    if nulls_after:
        a_before_b = tl.where(
            ~a_valid & b_valid,
            False,
            tl.where(
                a_valid & ~b_valid,
                True,
                a_val <= b_val if not descending else a_val >= b_val,
            ),
        )
    else:
        a_before_b = tl.where(
            ~a_valid & b_valid,
            True,
            tl.where(
                a_valid & ~b_valid,
                False,
                a_val <= b_val if not descending else a_val >= b_val,
            ),
        )

    # If we've exhausted one input, take from the other.
    take_a = tl.where(
        j >= n_b,
        True,
        tl.where(
            i >= n_a,
            False,
            a_before_b,
        ),
    )

    out_src = tl.where(take_a, 0, 1)
    out_idx = tl.where(take_a, i, j)

    tl.store(out_src_ptr + offs, out_src, mask=out_mask)
    tl.store(out_idx_ptr + offs, out_idx, mask=out_mask)

def merge_two_dataframes(
    left: TritonDataFrame,
    right: TritonDataFrame,
    key: str,
    context: IRExecutionContext,
) -> TritonDataFrame:

    left_key = left.column_map[key]
    right_key = right.column_map[key]

    n_left = left_key.obj.numel()
    n_right = right_key.obj.numel()
    n_out = n_left + n_right

    src = torch.empty(
        n_out,
        dtype=torch.int32,
        device=left_key.obj.device,
    )

    idx = torch.empty(
        n_out,
        dtype=torch.int64,
        device=left_key.obj.device,
    )

    grid = (triton.cdiv(n_out, 256),)

    with context.stream_ordered_after(left.cudf_stream, right.cudf_stream) as stream:
        _merge_indices_kernel[grid](
            left_key.obj,
            left_key.validity,
            right_key.obj,
            right_key.validity,
            src,
            idx,
            n_left,
            n_right,
            descending=left_key.order == plc.types.Order.DESCENDING,
            nulls_after=left_key.null_order == plc.types.NullOrder.AFTER,
            BLOCK=256,
        )

        # Gather all columns using src + idx.
        output_columns = []

        for name in left.column_names:
            lcol = left.column_map[name]
            rcol = right.column_map[name]

            lmask = src == 0
            rmask = ~lmask

            out = torch.empty(
                n_out,
                dtype=lcol.obj.dtype,
                device=lcol.obj.device,
            )

            # `idx` is an index into either left or right.
            #
            # Avoid invalid indexing by clamping.
            lidx = idx.clamp(max=max(n_left - 1, 0))
            ridx = idx.clamp(max=max(n_right - 1, 0))

            out[lmask] = lcol.obj[lidx[lmask]]
            out[rmask] = rcol.obj[ridx[rmask]]

            output_columns.append(
                TritonColumn(
                    out,
                    lcol.dtype,
                    name=name,
                )
            )

        return TritonDataFrame(
            output_columns,
            stream=stream,
            num_rows=n_out,
        )
    
    
    