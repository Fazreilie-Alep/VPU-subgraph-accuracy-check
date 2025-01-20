import subgraph_accuracy_check
import os
import openvino as ov
from dotenv import load_dotenv

load_dotenv()


def accuracy_check_per_subgraph(subgraph_folder_cpu, subgraph_folder_npu, output_csv, tol, dp):
    core = ov.Core()
    subgraph_files_cpu = [f for f in os.listdir(subgraph_folder_cpu) if f.endswith('.xml')]
    subgraph_files_npu = [f for f in os.listdir(subgraph_folder_npu) if f.endswith('.xml')]
    subgraph_files = list(set(subgraph_files_cpu) & set(subgraph_files_npu))
    
    results = []
    
    for subgraph_file in subgraph_files:
        model_path_cpu = os.path.join(subgraph_folder_cpu, subgraph_file)
        model_path_npu = os.path.join(subgraph_folder_npu, subgraph_file)
        results.append(subgraph_accuracy_check.accuracy_check(subgraph_file, core, model_path_cpu, model_path_npu, tol, dp))
    
    # Sort results by subgraph filename
    results.sort(key=lambda x: subgraph_accuracy_check.extract_number(x[0]))
    
    # Store results to CSV
    subgraph_accuracy_check.write_result(results=results, output_csv_filepath=output_csv, tol=tol, dp=dp)
    

if __name__ == "__main__":
    subgraph_folder_cpu = os.getenv('CPU_SUBGRAPH_FOLDER')
    subgraph_folder_npu = os.getenv('NPU_SUBGRAPH_FOLDER')
    output_csv = os.getenv('OUTPUT_CSV')
    tol = []
    dp = [4]
    
    accuracy_check_per_subgraph(subgraph_folder_cpu, subgraph_folder_npu, output_csv, tol, dp) 