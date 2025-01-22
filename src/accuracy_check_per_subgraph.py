import accuracy_check
import os
import openvino as ov
from dotenv import load_dotenv

load_dotenv()

def accuracy_check_per_subgraph(subgraph_folder_cpu, subgraph_folder_npu, subgraph_files, output_csv, tol, dp):
    core = ov.Core()
    
    results = [
        accuracy_check.accuracy_check(subgraph_file, core, os.path.join(subgraph_folder_cpu, subgraph_file), os.path.join(subgraph_folder_npu, subgraph_file), tol, dp)
        for subgraph_file in subgraph_files
    ]
    
    # Sort results by subgraph filename
    results.sort(key=lambda x: accuracy_check.extract_number(x[0]))
    
    # Store results to CSV
    accuracy_check.write_result(results=results, output_csv_filepath=output_csv, tol=tol, dp=dp)

def accuracy_check_per_subgraph_all(subgraph_folder_cpu, subgraph_folder_npu, output_csv, tol, dp):
    subgraph_files_cpu = {f for f in os.listdir(subgraph_folder_cpu) if f.endswith('.xml')}
    subgraph_files_npu = {f for f in os.listdir(subgraph_folder_npu) if f.endswith('.xml')}
    subgraph_files = subgraph_files_cpu & subgraph_files_npu
    
    accuracy_check_per_subgraph(subgraph_folder_cpu, subgraph_folder_npu, subgraph_files, output_csv, tol, dp)
    
if __name__ == "__main__":
    subgraph_folder_cpu = os.getenv('CPU_SUBGRAPH_FOLDER')
    subgraph_folder_npu = os.getenv('NPU_SUBGRAPH_FOLDER')
    tol = [0.01,0.001]
    dp = [4] 
    
    # check with respective subgraphs
    output_csv = os.getenv('OUTPUT_CSV')
    accuracy_check_per_subgraph_all(subgraph_folder_cpu, subgraph_folder_npu, output_csv, tol, dp) 
    
    # check with npu subgraphs
    # output_csv = os.getenv('OUTPUT_CSV_NPU')
    # accuracy_check_per_subgraph_all(subgraph_folder_npu, subgraph_folder_npu, output_csv, tol, dp) 
    # subgraph_files = ['OpenVINO-EP-subgraph_34.xml',
    #         'OpenVINO-EP-subgraph_23.xml',
    #         'OpenVINO-EP-subgraph_15.xml',
    #         'OpenVINO-EP-subgraph_12.xml',
    #         'OpenVINO-EP-subgraph_9.xml',
    #         'OpenVINO-EP-subgraph_8.xml',
    #         'OpenVINO-EP-subgraph_4.xml']
    # accuracy_check_per_subgraph(subgraph_folder_cpu, subgraph_folder_npu, subgraph_files, output_csv, tol, dp)