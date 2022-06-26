import numpy as np
import onnx
import onnxruntime as ort
import ctypes
import os
import torch
import argparse

def get_max_diff(tensor_list_a, tensor_list_b):
    assert len(tensor_list_a) > 0
    total_diff = [0]
    for a, b in zip(tensor_list_a, tensor_list_b):
        assert a.shape == b.shape
        diff = np.abs(a-b)
        diff /= np.abs(b).clip(1) # handle large floating numbers
        diff = np.max(diff)
        total_diff.append(diff)
    total_diff = max(total_diff)
    return total_diff

def ref_output(onnx_model_path):
    np.random.seed(0)
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABL_ALL
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {}
    inputs = []
    for value in ort_session.get_inputs():
        tensor = np.random.normal(size=value.shape).astype(np.float32)
        ort_inputs[value.name] = tensor
        inputs.append(tensor)
    outputs = ort_session.get_outputs()
    outputs_name = [item.name for item in outputs]
    outputs = ort_session.run(outputs_name, ort_inputs)
    return inputs, outputs

def test_output(prefix, inputs, outputs):
    lib_path = os.path.join(prefix, "build/libnnfusion_naive_rt.so")
    lib = ctypes.CDLL(lib_path)
    cur_dir = os.path.abspath(".")
    os.chdir(prefix)
    lib.cuda_init()
    arrs = []
    for tensor in inputs:
        arrs.append(torch.tensor(tensor, device=0))
    for tensor in outputs:
        dtype = torch.__getattribute__(str(tensor.dtype))
        arrs.append(torch.empty(tensor.shape, dtype=dtype, device=0))
    args = [ctypes.c_void_p(tensor.data_ptr()) for tensor in arrs]
    lib.kernel_entry(*args)
    torch.cuda.synchronize()

    os.chdir(cur_dir)

    output_arr = [tensor.cpu().numpy() for tensor in arrs[len(inputs):]]
    return output_arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    args = parser.parse_args()

    prefix = args.prefix
    inputs, outputs_ref = ref_output(os.path.join(prefix, "model.onnx"))

    outputs = test_output(os.path.join(prefix, "nnfusion_rt/cuda_codegen/"), inputs, outputs_ref)

    max_diff = get_max_diff(outputs, outputs_ref)
    print("Output diff : ", max_diff)

