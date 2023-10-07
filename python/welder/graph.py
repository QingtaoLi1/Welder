import functools
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tvm
from tvm import te, relay

from .lang import translate_ir_to_tvm
from .shape_inference import get_analyzer_by_te
from .te_utils import *


class Edge:
    def __init__(self, src_node: 'Node', dst_node: 'Node', src_id: int, dst_id: int):
        self.src_node = src_node
        self.dst_node = dst_node
        self.src_id = src_id
        self.dst_id = dst_id

class Node:
    def __init__(self, inputs: List[Union[Tuple['Node', int], 'Node', None]], name: str):
        self.name = name
        self._out_edges = []
        self._in_edges = []
        self._shapes = []
        self._dtypes = []
        self._tag = {}

        for i, node in enumerate(inputs):
            if node is None:
                inputs[i] = PlaceHolderNode()

        for dst_id, n in enumerate(inputs):
            if isinstance(n, Node):
                n = (n, 0)
            assert(len(n) == 2)
            src_node, src_id = n[0], n[1]
            edge = Edge(src_node, self, src_id, dst_id)
            self._in_edges.append(edge)
            src_node._out_edges.append(edge)

    @property
    def inputs(self) -> List[Edge]:
        return self._in_edges

    @property
    def outputs(self) -> List[Edge]:
        return self._out_edges

    def set_inputs(self, i: int, edge: Edge):
        assert i < len(self._in_edges)
        self._in_edges[i] = edge

    def set_outputs(self, i: int, edge: Edge):
        assert i < len(self._out_edges)
        self._out_edges[i] = edge

    def get_shape(self, id: int = 0) -> List[int]:
        return self._shapes[id]

    def set_shape(self, shape: List[int], id=0, overwrite=False) -> None:
        if len(self._shapes) <= id:
            self._shapes.extend([None for _ in range(id - len(self._shapes) + 1)])
        elif self._shapes[id] is not None and not overwrite:
            assert self._shapes[id] == list(map(int, shape)), (self._shapes, list(map(int, shape)))
        self._shapes[id] = list(map(int, shape))

    def get_dtype(self, id=0) -> tvm.DataType:
        return self._dtypes[id]

    def set_dtype(self, dtype: tvm.DataType, id=0) -> None:
        assert isinstance(dtype, tvm.DataType), type(dtype)
        if dtype == tvm.DataType("bool"):
            dtype = tvm.DataType("int8")
        if len(self._dtypes) <= id:
            self._dtypes.extend([None for _ in range(id - len(self._dtypes) + 1)])
        elif self._dtypes[id] is not None:
            assert self._dtypes[id] == dtype, (self._dtypes, dtype)
        self._dtypes[id] = dtype

    def is_placeholder(self):
        return False

    def is_output(self):
        return False

    def add_tag(self, k: str, v: Any = True) -> None:
        self._tag[k] = v

    def get_tag(self, k: str) -> Any:
        if k not in self._tag:
            return None
        return self._tag[k]

    def num_outputs(self) -> int:
        if len(self.outputs) == 0:
            return 0
        return max([e.src_id for e in self.outputs]) + 1

    def get_ir(self) -> str:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "<Node, " + self.name + ">"

class PlaceHolderNode(Node):
    def __init__(self, name=""):
        super().__init__([], "PlaceHolder " + name)

    def is_placeholder(self):
        return True

    def get_ir(self) -> str:
        return "placeholder"

class OutputNode(Node):
    def __init__(self, node, id=0):
        super().__init__([(node, id)], "Output ")
        self.set_shape(node.get_shape(id))
        self.set_dtype(node.get_dtype(id))

    def is_output(self):
        return True

    def get_ir(self) -> str:
        return "output"

class IRNode(Node):
    def __init__(self, inputs, compute: Union[str, relay.TensorType, relay.TupleType, List[te.Tensor], None], name="Compute") -> None:
        super().__init__(inputs, name)
        if compute is None:
            self.args = None
            return
        elif isinstance(compute, relay.TensorType):
            self.args = compute
            self.set_shape(compute.shape)
            self.set_dtype(tvm.DataType(compute.dtype))
            return
        elif isinstance(compute, relay.TupleType):
            self.args = compute
            for idx, type in enumerate(compute.fields):
                self.set_shape(type.shape, idx)
                self.set_dtype(tvm.DataType(type.dtype), idx)
            return
        elif isinstance(compute, str):
            input_args, output_args = translate_ir_to_tvm(compute)
        else:
            input_args, output_args = [], []
            for arg in compute:
                if isinstance(arg.op, te.PlaceholderOp):
                    input_args.append(arg)
                else:
                    output_args.append(arg)

        self.ana = get_analyzer_by_te(input_args + output_args)
        self.args = input_args + output_args

        # set input shapes and dtypes
        for edge, arg in zip(self.inputs, self.args):
            edge.src_node.set_shape(arg.shape, edge.src_id)
            edge.src_node.set_dtype(tvm.DataType(arg.dtype), edge.src_id)
        for output_id, arg in enumerate(output_args):
            self.set_shape(arg.shape, output_id)
            self.set_dtype(tvm.DataType(arg.dtype), output_id)

        # process all axis search space
        self.compute_ops = get_compute_ops(self.args)
        reduce_ops, _ = seperate_reduce_ops(self.compute_ops)
        if len(reduce_ops) > 0:
            self.reduce_op = reduce_ops[0]
            self.raxis = {str(axis.var.name): int(axis.dom.extent) for axis in self.reduce_op.reduce_axis}
            if len(self.raxis) != len(self.reduce_op.reduce_axis):
                raise Exception("Reduce axis should have unique names.")
        else:
            self.reduce_op = None
            self.raxis = {}

        self.schedule_stages = []
        for tensor in self.args:
            if isinstance(tensor.op, te.ComputeOp):
                self.schedule_stages.append(tensor.op)

    @functools.lru_cache()
    def get_space_dim(self):
        dim_size = []
        for axis in self.schedule_stages[0].axis:
            dim_size.append(int(axis.dom.extent))
        return dim_size

    def propogate(self, tile, rstep={}, targets=None):
        shape = {stage.name: [tvm.arith.ConstIntBound(0, val - 1) for val in tile] for stage in self.schedule_stages}
        return self.ana.infer(shape, rstep, targets)

    def propogate_inputs(self, tile, rstep={}) -> List[List[int]]:
        read_idx_offset = len(self.inputs)
        targets = [t.name for t in self.args[:read_idx_offset]]
        shapes = self.propogate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(self.args[:read_idx_offset]):
            # should not exceed original shape
            trimmed_shape = list(map(min, zip(shapes[arg.name], self.inputs[i].src_node.get_shape())))
            results.append(trimmed_shape)
        return results

    def propogate_outputs(self, tile, rstep={}):
        read_idx_offset = len(self.inputs)
        targets = [t.name for t in self.args[read_idx_offset:]]
        shapes = self.propogate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(self.args[read_idx_offset:]):
            # should not exceed original shape
            trimmed_shape = list(map(min, zip(shapes[arg.name], self.get_shape(i))))
            results.append(trimmed_shape)
        return results

    def propogate_reduction_inputs(self, shape, rstep={}) -> Dict[str, List[int]]:
        if self.reduce_op is None:
            return {}
        targets = [t.name for t in self.reduce_op.input_tensors]
        results = self.propogate(shape, rstep, targets)
        return results

    def get_reduce_inputs_dtype(self):
        if self.reduce_op is None:
            return {}
        return {t.name: tvm.DataType(t.dtype) for t in self.reduce_op.input_tensors}

    def footprint(self, shape, rstep, stride_map={}) -> int:
        result = 0
        shapes = self.propogate(shape, rstep)

        def is_broadcast_pattern(tensor, op):
            return isinstance(tensor.op, tvm.te.PlaceholderOp) \
                and len(shapes[op.name]) > len(shapes[tensor.name]) \
                and np.prod(shapes[op.name]) > np.prod(shapes[tensor.name])

        def is_after_reduce_stage(op):
            if not self.reduce_op: return False
            reduce_dependent_ops = getattr(self, "reduce_dependent_ops", None)
            if reduce_dependent_ops is None:
                reduce_dependent_ops = set()
                pre_order_traverse([self.reduce_op.output(0)], lambda t: reduce_dependent_ops.add(t.op))
                self.reduce_dependent_ops = reduce_dependent_ops
            return op not in reduce_dependent_ops

        # compute cached stages
        cached_tensor = []
        for op in self.compute_ops:
            for tensor in op.input_tensors:
                cache = tensor.name not in cached_tensor and (is_broadcast_pattern(tensor, op) or op is self.reduce_op)
                if not cache: continue
                cached_tensor.append(tensor.name)
                if is_after_reduce_stage(op):
                    continue # cache after reduce op can often reuse buffer in reduce stage
                if tensor in self.args:
                    input_id = self.args.index(tensor)
                    src_node = self.inputs[input_id].src_node
                    if not src_node.is_placeholder():
                        continue # allocated by previous node, not counted here
                if tensor.name in stride_map:
                    num_elem = stride_map[tensor.name].compute_elements_from_shape(shapes[tensor.name])
                else:
                    num_elem = np.prod(shapes[tensor.name])
                buffer_len = num_elem * int((tvm.DataType(tensor.dtype).bits + 7) // 8)
                buffer_len = (buffer_len + 31) // 32 * 32
                result += buffer_len
        return result, cached_tensor

    @functools.lru_cache()
    def infer_tensorcore_axis(self) -> Tuple[int]:
        # axis is fixed for one expression, so only inference and cached
        assert self.get_tag("tensorCoreConfig")
        C_ax_m, C_ax_n = self.get_tag("tensorCoreConfig")
        wmma_m, wmma_n, wmma_k = [16, 16, 16] # just for testing, any number is ok
        CL_shape = [1] * len(self.get_space_dim())
        CL_shape[C_ax_m] = wmma_m
        CL_shape[C_ax_n] = wmma_n
        shapes = self.propogate_reduction_inputs(CL_shape, {x : 1 for x in self.raxis})
        A_deps, B_deps = shapes.values()
        A_ax_m = A_deps.index(wmma_m)
        B_ax_n = B_deps.index(wmma_n)

        CL_shape = [1] * len(self.get_space_dim())
        shapes = self.propogate_reduction_inputs(CL_shape, {x : wmma_k for x in self.raxis})
        A_deps, B_deps = shapes.values()
        A_ax_k = A_deps.index(wmma_k)
        B_ax_k = B_deps.index(wmma_k)
        tc_axis = (A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n)
        return tc_axis

    def block_infer(self, tile_map, block_expr) -> List[tvm.tir.PrimExpr]:
        space_expr = []
        for ax_len, tile_len in zip(reversed(self.get_shape()), reversed(tile_map[self])):
            num_block = (ax_len + tile_len - 1) // tile_len
            space_expr.append(block_expr % num_block * tile_len)
            block_expr = block_expr // num_block
        output_exprs = {stage.name : list(reversed(space_expr)) for stage in self.schedule_stages}
        input_exprs = self.ana.get_input_exprs(output_exprs)
        result = []
        for i in range(len(self.inputs)):
            inode = self.inputs[i].src_node
            if isinstance(inode, PlaceHolderNode):
                result.append(None)
                continue
            block_expr = 0
            for expr, ax_len, tile_len in zip(input_exprs[self.args[i].name], inode.get_shape(), tile_map[inode]):
                num_block = (ax_len + tile_len - 1) // tile_len
                block_expr = block_expr * num_block + te.max(expr // tile_len, 0)
            result.append(block_expr)
        return result

    def clone(self, inputs) -> 'IRNode':
        new_node = IRNode(inputs, self.args, self.name)
        for k, v in self._tag.items():
            new_node.add_tag(k, v)
        return new_node

    def get_ir(self) -> str:
        return tvm.ir.save_json(self.args)

def topo_order(list_of_nodes) -> List[Node]:
    input_ready_count = {node : len(node.inputs) for node in list_of_nodes}
    ready = list(filter(lambda node : input_ready_count[node] == 0, list_of_nodes))
    output_list = []
    while len(ready) > 0:
        node = ready.pop(0)
        output_list.append(node)
        for edge in node.outputs:
            dst_node = edge.dst_node
            if dst_node not in input_ready_count:
                input_ready_count[dst_node] = len(dst_node.inputs)
                list_of_nodes.append(dst_node)
            input_ready_count[dst_node] -= 1
            assert(input_ready_count[dst_node] >= 0)
            if input_ready_count[dst_node] == 0:
                ready.append(dst_node)
    assert(len(list_of_nodes) == len(output_list))
    return output_list

def find_topo_sort_priority(output_node_list) -> List[Node]:
    import sys
    sys.setrecursionlimit(10000)
    def topo_sort_get_layer(node, topo_layer):
        if node in topo_layer:
            return
        topo_layer[node] = 0
        for edge in node.inputs:
            topo_sort_get_layer(edge.src_node, topo_layer)
            topo_layer[node] = max(topo_layer[node], topo_layer[edge.src_node] + 1)
    topo_layer = {}
    for node in output_node_list:
        topo_sort_get_layer(node, topo_layer)

    def topo_sort_dfs(node, visited, topo_order):
        if node in visited:
            return
        visited.add(node)
        ordered_input_nodes = sorted([edge.src_node for edge in node.inputs], key=lambda n:topo_layer[n], reverse=True)
        for n in ordered_input_nodes:
            topo_sort_dfs(n, visited, topo_order)
        topo_order.append(node)
    visited = set()
    topo_order = []
    for node in output_node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def find_topo_sort(output_node_list) -> List[Node]:
    def topo_sort_dfs(node, visited, topo_order):
        if node in visited:
            return
        visited.add(node)
        for edge in node.inputs:
            topo_sort_dfs(edge.src_node, visited, topo_order)
        topo_order.append(node)
    visited = set()
    topo_order = []
    for node in output_node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
