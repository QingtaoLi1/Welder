import tvm
from tvm import relay, ir
from welder.graph import IRNode, OutputNode, Node
from welder.te_utils import normalize_tensor_names
from welder.engine import Engine, MultiProcTunner
from ..integration import add_source

def tune_node(ordered_nodes, names):
    nodes = []
    for node in ordered_nodes:
        if node.name in names:
            nodes.append(node)
    from welder.arch import cuda
    tunner = MultiProcTunner(ordered_nodes, cuda(), device=0, topk=20)
    best = tunner.tune(nodes)

@ir.transform.module_pass(opt_level=0)
class WelderTunePass(relay.ExprMutator):
    def __init__(self, arch, save_perf_log=None):
        super().__init__()
        self.arch = arch
        self.save_perf_log_ = save_perf_log

    def transform_module(self, mod, ctx):
        extractor = TileGraphExtractor(self.arch.target)
        extractor.visit(mod["main"])

        ordered_nodes = extractor.ordered_nodes
        node_map = extractor.node_map
        tunner = MultiProcTunner(ordered_nodes, arch=self.arch, device="cuda:0", topk=20)
        engine = Engine(tunner)
        # tunner.load_cache("a.pkl")
        fusion_groups = engine.run(ordered_nodes)
        # tunner.dump_cache("a.pkl")
        if self.save_perf_log_:
            from ...engine import save_results
            save_results(fusion_groups, self.save_perf_log_)

        # apply fusion groups
        tuple_index_redirect = {}
        name_map = {node.name : (node_map[node] if node in node_map else None) for node in ordered_nodes}
        for group in fusion_groups:
            if group.cpresult is None:
                continue
            args = []
            params = []
            function_body = []
            ret_type = []
            for idx, (node_name, input_idx) in enumerate(group.cpresult.input_desc):
                shape = name_map[node_name].args[input_idx].checked_type.shape
                dtype = name_map[node_name].args[input_idx].checked_type.dtype
                params.append(relay.Var(f"p{idx}", relay.TensorType(shape, dtype)))
                args.append(self.visit(name_map[node_name].args[input_idx]))
            for node_name, output_idx in group.cpresult.output_desc:
                type = name_map[node_name].checked_type
                if isinstance(type, relay.TensorType):
                    shape = type.shape
                    dtype = type.dtype
                elif isinstance(type, relay.TupleType):
                    shape = type.fields[output_idx].shape
                    dtype = type.fields[output_idx].dtype
                else:
                    assert(0)
                function_body.append(relay.zeros(shape, dtype))
                ret_type.append(relay.TensorType(shape, dtype))
            if len(function_body) == 1:
                function_body = function_body[0]
                ret_type = ret_type[0]
            elif len(function_body) > 1:
                function_body = relay.Tuple(function_body)
                ret_type = relay.TupleType(ret_type)
            else:
                assert(0)
            function = relay.Function(params, function_body, ret_type)
            is_reshape_only = all([node.get_tag("memcpy") for node in group.nodes])
            if is_reshape_only:
                function = function.with_attr({"relay.reshape_only": 1, "Primitive": 1})
                var = function
            else:
                global_symbol = "tvmgen_welder_" + str("_".join([node.name for node in group.nodes]))
                func = function.with_attr({"Compiler": "welder", "global_symbol": global_symbol, "Primitive": 1, "Inline": 1})
                var = ir.GlobalVar(global_symbol)
                mod.update_func(var, func)
                add_source(global_symbol, group.cpresult.origin)
            call = relay.Call(var, args)

            if len(group.cpresult.output_desc) == 1:
                node_name, output_idx = group.cpresult.output_desc[0]
                assert output_idx == 0
                self.memo_map[name_map[node_name]] = call
            else:
                for i, (node_name, output_idx) in enumerate(group.cpresult.output_desc):
                    original_call = name_map[node_name]
                    if isinstance(original_call.op.body, relay.Tuple):
                        self.memo_map[original_call] = call
                        if original_call not in tuple_index_redirect:
                            tuple_index_redirect[original_call] = {}
                        tuple_index_redirect[original_call][output_idx] = i
                    else:
                        self.memo_map[original_call] = relay.TupleGetItem(call, i)
        mod.update_func(mod.get_global_var("main"), self.visit(mod["main"]))
        mod = relay.transform.InferType()(mod)
        return mod

class TileGraphExtractor(relay.ExprVisitor):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.ordered_nodes = []
        self.node_map = {}

    class NameExtractor(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.op_names = []

        def visit_call(self, call):
            super().visit_call(call)
            name = call.op.name.replace(".", "_")
            self.op_names.append(name)

    def visit_call(self, call):
        node_inputs = [self.visit(arg) for arg in call.args]

        if isinstance(call.op, relay.Function):
            options = {}
            if call.op.attrs and "Composite" in call.op.attrs:
                args = None
                op_name = "Opaque" + "_" + str(len(self.ordered_nodes))
                options["skip"] = True
            else:
                out = tvm._ffi.get_global_func("relay.backend.LowerToTE")(call.op)
                args = list(out.inputs) + list(out.outputs)
                args = normalize_tensor_names(args)
                f = self.NameExtractor()
                f.visit(call.op)
                op_name = "_".join(f.op_names) + "_" + str(len(self.ordered_nodes))
            if call.op.attrs and "tensorCoreConfig" in call.op.attrs:
                options["tensorCoreConfig"] = [int(x) for x in call.op.attrs["tensorCoreConfig"]]
            node = IRNode(node_inputs, args, op_name)
            if call.op.attrs and "relay.reshape_only" in call.op.attrs and call.op.attrs["relay.reshape_only"]:
                options["memcpy"] = True
            for k, v in options.items(): node.add_tag(k, v)
            self.ordered_nodes.append(node)
            self.node_map[node] = call
        elif isinstance(call.op, ir.expr.GlobalVar):
            args = call.checked_type
            op_name = "Opaque" + "_" + str(len(self.ordered_nodes))
            node = IRNode(node_inputs, args, op_name)
            node.add_tag("skip", True)
            self.ordered_nodes.append(node)
            self.node_map[node] = call
        else:
            raise NotImplementedError()
        return node

    def visit_function(self, fn):
        super().visit_function(fn)
        if isinstance(fn.body, relay.Tuple):
            body = fn.body
        elif isinstance(fn.body, relay.Call):
            body = [fn.body]
        else:
            raise NotImplementedError()
        for op in body:
            out = self.visit(op)
            if isinstance(out, Node):
                out = (out, 0)
            node = OutputNode(*out)
            self.ordered_nodes.append(node)

    def visit_tuple_getitem(self, t):
        return (self.visit(t.tuple_value), t.index)
