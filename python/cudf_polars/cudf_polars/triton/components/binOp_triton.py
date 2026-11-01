import torch
import triton

from cudf_polars.triton.components.base_triton import ExecutionContext, TritonColRef, TritonExpr
from cudf_polars.triton.components.column_triton import TritonColumn
from cudf_polars.triton.components.dataframe_triton import TritonDataFrame

import triton.language as tl
import pylibcudf as plc


from typing import ClassVar

from cudf_polars.triton.components.boolOp_triton import boolean_operation, is_integer_not_bool


class TritonBinOp(TritonExpr):
    __slots__ = ("op",)
    _non_child = ("dtype", "op")

    def __init__(
        self,
        dtype: torch.dtype,
        op: plc.binaryop.BinaryOperator,
        left: TritonExpr,
        right: TritonExpr,
    ) -> None:
        self.dtype = dtype
        if dtype == torch.bool:# plc.traits.is_boolean(self.dtype.plc_type):
            # For boolean output types, bitand and bitor implement
            # boolean logic, so translate. bitxor also does, but the
            # default behaviour is correct.
            op = TritonBinOp._BOOL_KLEENE_MAPPING.get(op, op)
        self.op = op
        self.children = (left, right)
        self.is_pointwise = True
        # if not plc.binaryop.is_supported_operation(
        #     self.dtype.plc_type, left.dtype.plc_type, right.dtype.plc_type, op
        # ):
        #     raise NotImplementedError(
        #         f"Operation {op.name} not supported "
        #         f"for types {left.dtype.id().name} and {right.dtype.id().name} "
        #         f"with output type {self.dtype.id().name}"
        #     )

    _BOOL_KLEENE_MAPPING: ClassVar[
        dict[plc.binaryop.BinaryOperator, plc.binaryop.BinaryOperator]
    ] = {
        plc.binaryop.BinaryOperator.BITWISE_AND: plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
        plc.binaryop.BinaryOperator.BITWISE_OR: plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
        plc.binaryop.BinaryOperator.LOGICAL_AND: plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
        plc.binaryop.BinaryOperator.LOGICAL_OR: plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
    }

    # _MAPPING: ClassVar[
    #     dict[polars._expr_nodes.Operator, plc.binaryop.BinaryOperator]
    # ] = {
    #     pl_expr.Operator.Eq: plc.binaryop.BinaryOperator.EQUAL,
    #     pl_expr.Operator.EqValidity: plc.binaryop.BinaryOperator.NULL_EQUALS,
    #     pl_expr.Operator.NotEq: plc.binaryop.BinaryOperator.NOT_EQUAL,
    #     pl_expr.Operator.NotEqValidity: plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
    #     pl_expr.Operator.Lt: plc.binaryop.BinaryOperator.LESS,
    #     pl_expr.Operator.LtEq: plc.binaryop.BinaryOperator.LESS_EQUAL,
    #     pl_expr.Operator.Gt: plc.binaryop.BinaryOperator.GREATER,
    #     pl_expr.Operator.GtEq: plc.binaryop.BinaryOperator.GREATER_EQUAL,
    #     pl_expr.Operator.Plus: plc.binaryop.BinaryOperator.ADD,
    #     pl_expr.Operator.Minus: plc.binaryop.BinaryOperator.SUB,
    #     pl_expr.Operator.Multiply: plc.binaryop.BinaryOperator.MUL,
    #     pl_expr.Operator.Divide: plc.binaryop.BinaryOperator.DIV,
    #     pl_expr.Operator.TrueDivide: plc.binaryop.BinaryOperator.TRUE_DIV,
    #     pl_expr.Operator.FloorDivide: plc.binaryop.BinaryOperator.FLOOR_DIV,
    #     pl_expr.Operator.Modulus: plc.binaryop.BinaryOperator.PYMOD,
    #     pl_expr.Operator.And: plc.binaryop.BinaryOperator.BITWISE_AND,
    #     pl_expr.Operator.Or: plc.binaryop.BinaryOperator.BITWISE_OR,
    #     pl_expr.Operator.Xor: plc.binaryop.BinaryOperator.BITWISE_XOR,
    #     pl_expr.Operator.LogicalAnd: plc.binaryop.BinaryOperator.LOGICAL_AND,
    #     pl_expr.Operator.LogicalOr: plc.binaryop.BinaryOperator.LOGICAL_OR,
    # }

    # def do_evaluate_dual(
    #     self, left_df: TritonDataFrame, right_df: TritonDataFrame, *, context: ExecutionContext = ExecutionContext.FRAME    
    # ) -> TritonColumn:
    _JOIN_DUAL_OP = {
        plc.binaryop.BinaryOperator.EQUAL,
        plc.binaryop.BinaryOperator.NOT_EQUAL,
        plc.binaryop.BinaryOperator.LESS,
        plc.binaryop.BinaryOperator.LESS_EQUAL,
        plc.binaryop.BinaryOperator.GREATER,
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
        # TODO: Handle other binary operations as needed
    }

    def do_evaluate_dual(
            self, left_df: TritonDataFrame, right_df: TritonColumn, *, context: ExecutionContext = ExecutionContext.FRAME    
    ) -> TritonColumn:
        # TODO? Combine streams?
        def find_col_ref(exp: TritonExpr):
            res = None
            for c in exp.children:
                t = find_col_ref(c)
                if t != None:
                    if res != None and res.table_ref != t.table_ref:
                        raise ValueError("Expected only one triton data frame!")
                    res = t

            if isinstance(exp, TritonColRef):
                if res != None and res.table_ref != exp.table_ref:
                    raise ValueError("Expected only one triton data frame!")
                return exp
            return res
        
        def child_call(exp):
            if isinstance(exp, TritonBinOp):
                return exp.do_evaluate_dual(left_df, right_df, context=context)
            tbl = find_col_ref(exp)

            if tbl == None or tbl.table_ref == plc.expressions.TableReference.LEFT:
                return exp.do_evaluate(left_df, context=context)
            else:
                return exp.do_evaluate(right_df, context=context)
            
            
        left, right = (child_call(child) for child in self.children)
        if self.op in self._JOIN_DUAL_OP:
            n_left = left_df.num_rows
            n_right = right_df.num_rows
            matches_obj = torch.ones((n_left, n_right), device=left.obj.device, dtype=self.dtype)
            matches_validity = torch.ones((n_left, n_right), device=left.obj.device, dtype=self.dtype)
            for r in range(n_left): # TODO: Obj scalar may be null
                new_col = TritonColumn(
                    torch.tensor([left.obj[r]], device=left.obj.device), 
                    dtype=left.dtype, 
                    validity=torch.tensor([left.validity[r]], device=left.obj.device)
                )
                t = self._apply_operation(left_df, new_col, right, context=context)
                matches_obj[r,:] = t.obj & t.validity
                matches_validity[r,:] = t.validity
        
            return TritonColumn(matches_obj, self.dtype, validity=matches_validity)
        
        return self._apply_operation(left_df, left, right, context=context)

    def __repr__(self):
        return f"<TritonBinOp(op={self.op}, outty={self.dtype})>"

    def do_evaluate(
        self, df: TritonDataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> TritonColumn:
        """Evaluate this expression given a dataframe for context."""
        left, right = (child.evaluate(df, context=context) for child in self.children)
        return self._apply_operation(df, left, right, context=context)

    def _apply_operation(
        self, df: TritonDataFrame, left: TritonColumn, right: TritonColumn, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> TritonColumn:
        if isinstance(left.obj, plc.Column) or isinstance(right.obj, plc.Column):
            from cudf_polars.containers import Column
            from cudf_polars.triton.triton_conversion import (
                column_cudf_to_triton,
                dtype_triton_to_cudf,
            )

            def as_libcudf_operand(
                column: TritonColumn,
            ) -> plc.Column | plc.Scalar:
                if isinstance(column.obj, plc.Column):
                    if column.is_scalar:
                        return plc.Scalar.from_py(
                            column.obj_scalar(df.cudf_stream),
                            dtype_triton_to_cudf(column.dtype).plc_type,
                            stream=df.cudf_stream,
                        )
                    return column.obj
                value = column.obj_scalar(df.cudf_stream)
                return plc.Scalar.from_py(
                    value,
                    dtype_triton_to_cudf(column.dtype).plc_type,
                    stream=df.cudf_stream,
                )

            output_dtype = dtype_triton_to_cudf(self.dtype)
            libcudf_op = {
                plc.binaryop.BinaryOperator.ADD: plc.binaryop.BinaryOperator.ADD,
                plc.binaryop.BinaryOperator.SUB: plc.binaryop.BinaryOperator.SUB,
                plc.binaryop.BinaryOperator.MUL: plc.binaryop.BinaryOperator.MUL,
                plc.binaryop.BinaryOperator.DIV: plc.binaryop.BinaryOperator.DIV,
                plc.binaryop.BinaryOperator.LESS: plc.binaryop.BinaryOperator.LESS,
                plc.binaryop.BinaryOperator.GREATER: plc.binaryop.BinaryOperator.GREATER,
                plc.binaryop.BinaryOperator.GREATER_EQUAL: plc.binaryop.BinaryOperator.GREATER_EQUAL,
                plc.binaryop.BinaryOperator.LESS_EQUAL: plc.binaryop.BinaryOperator.LESS_EQUAL,
                plc.binaryop.BinaryOperator.EQUAL: plc.binaryop.BinaryOperator.EQUAL,
            }.get(self.op)
            if libcudf_op is not None:
                result = plc.binaryop.binary_operation(
                    as_libcudf_operand(left),
                    as_libcudf_operand(right),
                    libcudf_op,
                    output_dtype.plc_type,
                    stream=df.cudf_stream,
                )
                return column_cudf_to_triton(
                    Column(result, dtype=output_dtype),
                    name=None,
                    stream=df.cudf_stream,
                )

        lop: torch.Tensor | plc.Scalar = left.obj
        rop: torch.Tensor | plc.Scalar = right.obj
        if left.size != right.size:
            if left.is_scalar:
                lop = left.obj_scalar(df.stream)
            elif right.is_scalar:
                rop = right.obj_scalar(df.stream)
            else:
                print("NO SCALAR DETECTED!")

        if is_integer_not_bool(self.dtype) and self.op in {
            plc.binaryop.BinaryOperator.FLOOR_DIV,
            plc.binaryop.BinaryOperator.PYMOD,
        }:
            raise NotImplementedError("NOT IMPL")
            if (
                right.obj.size() == 1
                and right.obj.to_scalar(stream=df.stream).to_py(stream=df.stream) == 0
            ):
                return TritonColumn(
                    plc.Column.all_null_like(
                        left.obj, left.obj.size(), stream=df.stream
                    ),
                    dtype=self.dtype,
                )

            if right.obj.size() > 1:
                rop = plc.replace.find_and_replace_all(
                    right.obj,
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(
                            0, dtype=self.dtype.plc_type, stream=df.stream
                        ),
                        1,
                        stream=df.stream,
                    ),
                    plc.Column.from_scalar(
                        plc.Scalar.from_py(
                            None, dtype=self.dtype.plc_type, stream=df.stream
                        ),
                        1,
                        stream=df.stream,
                    ),
                    stream=df.stream,
                )

        # Binary Operations
        ret_obj = None
        if left.validity.numel() == 1:
            out_shape = right.validity.shape
        else:
            out_shape = left.validity.shape
        left_validity = left.validity.expand(out_shape)
        right_validity = right.validity.expand(out_shape)
        ret_validty = left_validity & right_validity
        
        if self.op == plc.binaryop.BinaryOperator.LESS:
            ret_obj = compare(lop, rop, "less", self.dtype, df.stream)
        elif self.op == plc.binaryop.BinaryOperator.GREATER:
            ret_obj = compare(lop, rop, "greater", self.dtype, df.stream)
        elif self.op == plc.binaryop.BinaryOperator.GREATER_EQUAL:
            ret_obj = compare(lop, rop, "geq", self.dtype, df.stream)
        elif self.op == plc.binaryop.BinaryOperator.LESS_EQUAL:
            ret_obj = compare(lop, rop, "leq", self.dtype, df.stream)
        elif self.op == plc.binaryop.BinaryOperator.EQUAL:
            ret_obj = compare(lop, rop, "equal", self.dtype, df.stream)

        elif self.op == plc.binaryop.BinaryOperator.NULL_LOGICAL_AND:
            ret_obj = boolean_operation(lop, rop, "and", self.dtype, stream=df.stream)

            r_inv = boolean_operation(rop, rop, "inv", stream=df.stream)
            l_inv = boolean_operation(lop, lop, "inv", stream=df.stream)
            r_val = boolean_operation(left_validity, r_inv, "and", stream=df.stream)
            l_val = boolean_operation(right_validity, l_inv, "and", stream=df.stream)
            t_val = boolean_operation(r_val, l_val, "or", stream=df.stream)
            ret_validty = boolean_operation(ret_validty, t_val, "or", stream=df.stream) # FALSExNULL = FALSE

        elif self.op == plc.binaryop.BinaryOperator.NULL_LOGICAL_OR:
            ret_obj = boolean_operation(lop, rop, "or", self.dtype, stream=df.stream)

            left_val = boolean_operation(lop, left.validity, "and", stream=df.stream)
            right_val = boolean_operation(rop, right.validity, "and", stream=df.stream)
            t_val = boolean_operation(left_val, right_val, "or", stream=df.stream)
            ret_validty = boolean_operation(ret_validty, t_val, "or", stream=df.stream) # TRUExNULL = TRUE

        elif self.op == plc.binaryop.BinaryOperator.ADD:
            ret_obj = compare(lop, rop, "add", self.dtype, df.stream)
        elif self.op == plc.binaryop.BinaryOperator.SUB:
            ret_obj = compare(lop, rop, "sub", self.dtype, df.stream)
        elif self.op == plc.binaryop.BinaryOperator.MUL:
            ret_obj = compare(lop, rop, "mul", self.dtype, df.stream)
        elif self.op == plc.binaryop.BinaryOperator.DIV:
            ret_obj = compare(lop, rop, "div", self.dtype, df.stream)
            divisor = (
                rop
                if isinstance(rop, torch.Tensor)
                else torch.as_tensor(rop, device=ret_obj.device)
            )
            ret_validty = (
                torch.ones(ret_obj.shape, device=ret_obj.device, dtype=torch.bool)
                if ret_validty is None
                else ret_validty
            ) & (divisor != 0)
        else:
            raise NotImplementedError(f"Operator not implemented yet {self.op}!")
        
        

        return TritonColumn(ret_obj, validity=ret_validty, dtype=self.dtype)


@triton.jit
def compare_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    x_is_scalar: tl.constexpr,
    y_is_scalar: tl.constexpr,
    x_scalar,
    y_scalar,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load x
    if x_is_scalar:
        x = x_scalar
    else:
        x = tl.load(x_ptr + offsets, mask=mask)

    # Load y
    if y_is_scalar:
        y = y_scalar
    else:
        y = tl.load(y_ptr + offsets, mask=mask)

    # Comparison
    if OP == 0:
        out = x < y
    elif OP == 1:
        out = x > y
    elif OP == 2:
        out = x == y
    elif OP == 3:
        out = x <= y
    elif OP == 4:
        out = x >= y
    elif OP == 5:
        out = x + y
    elif OP == 6:
        out = x - y
    elif OP == 7:
        out = x * y
    elif OP == 8:
        out = x / y

    tl.store(out_ptr + offsets, out, mask=mask)


def compare(
        x: torch.Tensor | tl.constexpr, 
        y: torch.Tensor | tl.constexpr, 
        op: str,
        outty: torch.dtype,
        stream: torch.Stream
    ):
    op_map = {
        "less": 0,
        "greater": 1,
        "equal": 2,
        "leq": 3,
        "geq": 4,
        "add": 5,
        "sub": 6,
        "mul": 7,
        "div": 8
    }
    
    assert op in op_map

    # Determine whether inputs are scalars
    x_is_scalar = not isinstance(x, torch.Tensor)
    y_is_scalar = not isinstance(y, torch.Tensor)

    # At least one input must be a tensor
    assert not (x_is_scalar and y_is_scalar), \
        "At least one input must be a torch.Tensor"

    # Determine device
    if x_is_scalar:
        device = y.device
    else:
        device = x.device

    # Convert Python scalar to tensor on the correct device/dtype
    if x_is_scalar:
        x = torch.tensor(x, device=device, dtype=y.dtype)

    if y_is_scalar:
        y = torch.tensor(y, device=device, dtype=x.dtype)

    assert x.is_cuda
    assert y.is_cuda

    # Shapes must either match or one must be a scalar
    if x.numel() != 1 and y.numel() != 1:
        assert x.shape == y.shape

    # Output shape is the non-scalar shape
    if x.numel() == 1:
        out_shape = y.shape
    else:
        out_shape = x.shape

    out = torch.empty(
        out_shape,
        device=device,
        dtype=outty,
    )

    n_elements = out.numel()

    # A 1-element tensor is treated as a scalar
    x_is_scalar = x.numel() == 1
    y_is_scalar = y.numel() == 1

    # Extract scalar values
    if x_is_scalar:
        x_scalar = x.reshape(()).item()
    else:
        x_scalar = 0

    if y_is_scalar:
        y_scalar = y.reshape(()).item()
    else:
        y_scalar = 0

    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )

    with torch.cuda.stream(stream):
        compare_kernel[grid](
            x,
            y,
            out,
            n_elements,

            x_is_scalar=x_is_scalar,
            y_is_scalar=y_is_scalar,

            x_scalar=x_scalar,
            y_scalar=y_scalar,

            OP=op_map[op],
            BLOCK_SIZE=1024,
        )

    return out