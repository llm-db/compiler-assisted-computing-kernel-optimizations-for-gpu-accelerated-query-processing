

import torch
import triton
import numpy as np
import triton.language as tl

from cudf_polars.triton.components.column_triton import NamedColumnTriton

BLOCK_SIZE = 256



# @triton.jit
# def _filter_column_kernel(
#     input_ptr,
#     mask_ptr,
#     cumsum_ptr,
#     output_ptr,
#     n_rows,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     valid_bounds = offsets < n_rows

#     # Load data, mask, and pre-computed prefix sum
#     mask_val = tl.load(mask_ptr + offsets, mask=valid_bounds, other=False)
#     data_val = tl.load(input_ptr + offsets, mask=valid_bounds, other=0)
#     cumsum_val = tl.load(cumsum_ptr + offsets, mask=valid_bounds, other=0)

#     # Calculate 0-indexed destination output offsets
#     write_idx = cumsum_val - 1
#     store_mask = valid_bounds & mask_val

#     # Scatter valid elements to the contiguous output buffer
#     tl.store(output_ptr + write_idx, data_val, mask=store_mask)


@triton.jit
def gather_masked_kernel(
    x_ptr,
    indices_ptr,
    out_ptr,
    n_selected,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    m = offs < n_selected

    indices = tl.load(indices_ptr + offs, mask=m)
    values = tl.load(x_ptr + indices, mask=m)

    tl.store(out_ptr + offs, values, mask=m)

def masked_gather(
        x: torch.Tensor, 
        indices: torch.Tensor,
        stream: torch.Stream
    ):

    n_selected = indices.numel()

    grid = (triton.cdiv(n_selected, BLOCK_SIZE),)

    ret = torch.empty(
        (n_selected,),
        dtype=x.dtype,
        device=x.device,
    )

    with torch.cuda.stream(stream):
        gather_masked_kernel[grid](
            x,
            indices,
            ret,
            n_selected,
            BLOCK=BLOCK_SIZE,
        )

    return ret