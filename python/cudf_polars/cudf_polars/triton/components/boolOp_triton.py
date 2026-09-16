
import torch
import triton

import triton.language as tl

def is_integer_not_bool(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }

@triton.jit
def bool_operation_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    IS_X_SCALAR: tl.constexpr,
    IS_Y_SCALAR: tl.constexpr,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x: if scalar, read from single memory location (x_ptr); otherwise, offset
    if IS_X_SCALAR:
        x = tl.load(x_ptr)
    else:
        x = tl.load(x_ptr + offsets, mask=mask)

    # Load y: if scalar, read from single memory location (y_ptr); otherwise, offset
    if IS_Y_SCALAR:
        y = tl.load(y_ptr)
    else:
        y = tl.load(y_ptr + offsets, mask=mask)

    # Comparison / Logical operations
    if OP == 0:
        out = x & y
    elif OP == 1:
        out = x | y
    elif OP == 2:
        out = ~(x | y)
    elif OP == 3:
        out = ~(x & y)
    elif OP == 4:
        out = ~x

    tl.store(out_ptr + offsets, out, mask=mask)

def boolean_operation(
    x: torch.Tensor,
    y: torch.Tensor,
    op: str,
    outty: torch.dtype = torch.bool,
    *,
    stream: torch.Stream
):
    op_map = {
        "and": 0,
        "or": 1,

        "nor": 2,
        "nand": 3,

        "inv": 4
    }

    x_is_scalar = not isinstance(x, torch.Tensor)
    y_is_scalar = not isinstance(y, torch.Tensor)

    assert not (x_is_scalar and y_is_scalar), \
        "At least one input must be a torch.Tensor"

    if not x_is_scalar:
        x_is_scalar = x.numel() == 1
    if not y_is_scalar:
        y_is_scalar |= y.numel() == 1

    # Determine device from whichever input is a tensor
    if not isinstance(x, torch.Tensor):
        device = y.device
    else:
        device = x.device

    # Convert Python scalars to 1D GPU tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor([x], device=device, dtype=y.dtype)
    elif x.numel() == 1 and x.dim() == 0:
        x = x.reshape(1)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor([y], device=device, dtype=x.dtype)
    elif y.numel() == 1 and y.dim() == 0:
        y = y.reshape(1)

    assert op in op_map
    assert x.is_cuda
    assert y.is_cuda

    # Output shape matches the tensor that is not a scalar
    out_shape = y.shape if x_is_scalar else x.shape
    if outty == None:
        outty = torch.bool
    out = torch.empty(
        out_shape,
        device=device,
        dtype=outty
    )
    n_elements = out.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch.cuda.stream(stream):
        bool_operation_kernel[grid](
            x,
            y,
            out,
            n_elements,
            IS_X_SCALAR=x_is_scalar,
            IS_Y_SCALAR=y_is_scalar,
            OP=op_map[op],
            BLOCK_SIZE=1024,
        )

    return out