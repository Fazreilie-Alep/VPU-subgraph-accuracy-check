import os
import numpy as np
import openvino as ov
import csv
import re

def load_model(core, model_path, device):
    try:
        compiled_model = core.compile_model(model_path, device)
        return compiled_model
    except RuntimeError as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        raise

def perform_inference(compiled_model, input_data):
    try:
        infer_request = compiled_model.create_infer_request()

        # Set the input tensors
        for i, input_tensor in enumerate(input_data):
            infer_request.set_input_tensor(i, input_tensor)

        infer_request.start_async()
        infer_request.wait()

         # Handle multiple output tensors
        output_tensors = []
        for i in range(len(compiled_model.outputs)):
            output_tensor = infer_request.get_output_tensor(i)
            output_tensors.append(output_tensor.data)

        return output_tensors

    except RuntimeError as e:
        print(f"Error during inference: {str(e)}")
        raise

def compare_results(cpu_results, npu_results, accuracy, use_tol=True):
    # Assuming cpu_results and npu_results are lists of numpy arrays (one per output tensor)
    all_passed = True
    
    for i, (cpu, npu) in enumerate(zip(cpu_results, npu_results)):
        if use_tol:
            if not np.allclose(cpu, npu, atol=accuracy):
                all_passed = False
                print(f"Output {i} differs:")
            else:
                print(f"Output {i} matches.")
        else:
            cpu_rounded = np.round(cpu, accuracy)
            npu_rounded = np.round(npu, accuracy)
            if not np.array_equal(cpu_rounded, npu_rounded):
                all_passed = False
                print(f"Output {i} differs:")
            else:
                print(f"Output {i} matches.")
        
        print(f"CPU result:\n{cpu_rounded if not use_tol else cpu}")
        print(f"NPU result:\n{npu_rounded if not use_tol else npu}")
        
    return all_passed


def extract_number(filename):
    match = re.search(r'_(\d+)\.xml$', filename)
    return int(match.group(1)) if match else float('inf')


def main(subgraph_folder_cpu, subgraph_folder_npu, output_csv, tol, dp):
    core = ov.Core()
    subgraph_files_cpu = [f for f in os.listdir(subgraph_folder_cpu) if f.endswith('.xml')]
    subgraph_files_npu = [f for f in os.listdir(subgraph_folder_npu) if f.endswith('.xml')]
    subgraph_files = list(set(subgraph_files_cpu) & set(subgraph_files_npu))
    
    results = []
    
    for subgraph_file in subgraph_files:
        model_path_cpu = os.path.join(subgraph_folder_cpu, subgraph_file)
        model_path_npu = os.path.join(subgraph_folder_npu, subgraph_file)
        
        # Load models for CPU and NPU
        try:
            compiled_model_cpu = load_model(core, model_path_cpu, "CPU")
            compiled_model_npu = load_model(core, model_path_npu, "NPU")
        except RuntimeError as e:
            print(f"Skipping {subgraph_file} due to model loading error.")
            continue
        
        print(f"Processing {subgraph_file}...")
        
        # Get input information
        input_info = compiled_model_npu.inputs  # use the static dimension NPU input
        input_data = []
        
        for input in input_info:
            input_shape = input.shape
            input_type = input.element_type.to_dtype()
            input_array = np.random.rand(*input_shape).astype(input_type)
            input_tensor = ov.Tensor(array=input_array, shared_memory=True)
            input_data.append(input_tensor)
        
        # Perform cpu inference
        try:
            cpu_result = perform_inference(compiled_model_cpu, input_data)
        except RuntimeError as e:
            print(f"Error during cpu inference for {subgraph_file}: {str(e)}")
            continue
        
        # Perform cpu inference
        try:
            npu_result = perform_inference(compiled_model_npu, input_data)
        except RuntimeError as e:
            print(f"Error during npu inference for {subgraph_file}: {str(e)}")
            continue
        
        # Compare results and Store results
        results_label = []
        for tol_val in tol:
            passed = compare_results(cpu_result, npu_result, tol_val, True)
            result_label = "passed" if passed else "failed"
            results_label.append(result_label)
            
        for dp_val in dp:
            passed = compare_results(cpu_result, npu_result, dp_val, False)
            result_label = "passed" if passed else "failed"
            results_label.append(result_label)
            
        results.append([subgraph_file] + results_label)
    
    # Sort results by subgraph filename
    results.sort(key=lambda x: extract_number(x[0]))
        
    
    # Write results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = tol + [str(dp_val)+"dp" for dp_val in dp]
        writer.writerow(["Subgraph"] + [ str(accuracy) + " result" for accuracy in header])
        writer.writerows(results)
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    subgraph_folder_cpu = "C:\\projects\\subgraph_accuracy_check\\cpu"
    subgraph_folder_npu = "C:\\projects\\subgraph_accuracy_check\\npu"
    output_csv = "C:\\Users\\falep\OneDrive - Intel Corporation\\DL Compiler\\Initiatives\\subgraph_accuracy_check_onnx_result\\result.csv"
    
    main(subgraph_folder_cpu, subgraph_folder_npu, output_csv, [1e-2, 1e-4], [2, 4])