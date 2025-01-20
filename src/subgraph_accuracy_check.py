import csv
import re
import os
import numpy as np
import openvino as ov


def load_model(core, model_path, device):
    try:
        return core.compile_model(model_path, device)
    except RuntimeError as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None


def perform_inference(compiled_model, input_data):
    try:
        infer_request = compiled_model.create_infer_request()
        for i, input_tensor in enumerate(input_data):
            infer_request.set_input_tensor(i, input_tensor)

        infer_request.start_async()
        infer_request.wait()

        return [infer_request.get_output_tensor(i).data for i in range(len(compiled_model.outputs))]
    except RuntimeError as e:
        print(f"Error during inference: {str(e)}")
        return None


def compare_results(cpu_results, npu_results, accuracy, use_tol=True):
    all_passed = True
    matched_count = 0
    total_elements = sum(cpu.size for cpu in cpu_results)

    for cpu, npu in zip(cpu_results, npu_results):
        if use_tol:
            matches = np.isclose(cpu, npu, atol=accuracy)
        else:
            matches = np.round(cpu, accuracy) == np.round(npu, accuracy)

        matched_count += np.sum(matches)
        if not np.all(matches):
            all_passed = False

    match_percentage = int((matched_count / total_elements) * 100)
    return all_passed, match_percentage


def extract_number(filename):
    match = re.search(r'_(\d+)\.xml$', filename)
    return int(match.group(1)) if match else float('inf')


def format_tensor_elements(tensor):
    return ','.join(f"{x:.4f}" for x in tensor.flatten()[:5])


def accuracy_check(subgraph_file, core, model_path_cpu, model_path_npu, tol, dp):
    compiled_model_cpu = load_model(core, model_path_cpu, "CPU")
    compiled_model_npu = load_model(core, model_path_npu, "NPU")
    if not compiled_model_cpu or not compiled_model_npu:
        print(f"Skipping {subgraph_file} due to model loading error.")
        return [subgraph_file] + ["Model loading failed"]

    print(f"Processing {subgraph_file}...")

    input_info = compiled_model_npu.inputs
    np.random.seed(911)
    input_data = [ov.Tensor(array=np.random.rand(*input.shape).astype(input.element_type.to_dtype()), shared_memory=True) for input in input_info]

    cpu_result = perform_inference(compiled_model_cpu, input_data)
    if cpu_result is None:
        print(f"Error during CPU inference for {subgraph_file}")
        return [subgraph_file] + ["Inference failed"]

    npu_result = perform_inference(compiled_model_npu, input_data)
    if npu_result is None:
        print(f"Error during NPU inference for {subgraph_file}")
        return [subgraph_file] + ["Inference failed"]

    results_label = []
    for tol_val in tol:
        passed, match_percentage = compare_results(cpu_result, npu_result, tol_val, True)
        results_label.append(f"{'passed' if passed else 'failed'} {match_percentage}")

    for dp_val in dp:
        passed, match_percentage = compare_results(cpu_result, npu_result, dp_val, False)
        results_label.append(f"{'passed' if passed else 'failed'} {match_percentage}")

    first_5_cpu = [format_tensor_elements(tensor) for tensor in cpu_result[:5]]
    first_5_npu = [format_tensor_elements(tensor) for tensor in npu_result[:5]]

    return [subgraph_file] + results_label + [first_5_cpu, first_5_npu]


def write_result(results, output_csv_filepath, tol, dp):
    with open(output_csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = tol + [f"{dp_val}dp" for dp_val in dp]
        writer.writerow(["Subgraph"] + [f"{accuracy} result" for accuracy in header] + ["First 5 CPU Tensors", "First 5 NPU Tensors"])
        writer.writerows(results)

    print(f"Results saved to {output_csv_filepath}")