import edit_xml
import subgraph_accuracy_check
from lxml import objectify
from shutil import copyfile
from pathlib import Path
import openvino as ov
import os
import csv
from dotenv import load_dotenv

load_dotenv()

def get_node_list(modelpath):
    # Parse XML
    modelpath = Path(modelpath)  # Ensure modelpath is a Path object
    parser = objectify.makeparser(remove_comments=True)
    temp_tree = objectify.parse(str(modelpath), parser=parser)

    if temp_tree is None:
        raise Exception("Cannot parse model IR")

    # List of unwanted substrings to filter out
    excluded_names = {"OpenVINO-EP-subgraph", "sink_port", "GroupQueryAttention"}

    node_lst = []
    i=0
    # Iterate over all elements/nodes in the tree
    for node in temp_tree.iter():  # Start enumerating from 1
        if 'name' in node.attrib and "/model/layers." in node.attrib['name'] and not any(excluded in node.attrib['name'] for excluded in excluded_names):
            print(f"Node {i+1}: {node.tag}, Name Attribute: {node.attrib['name']}")
            node_lst.append(node.attrib['name'])
            i+=1

    # Return first 3 nodes
    return node_lst


def create_new_subgraph(modelpath, layername):
    # Ensure modelpath is a Path object
    modelpath = Path(modelpath)

    if modelpath.suffix != ".xml":
        raise Exception("Path is not to model IR file")

    if len(layername) == 0:
        raise Exception("Empty layer name")

    # Parse XML
    parser = objectify.makeparser(remove_comments=True)
    tree = objectify.parse(str(modelpath), parser=parser)
    if tree is None:
        raise Exception("Cannot parse model IR")

    # Cut layers off
    edit_xml.delLayers(tree, layername)

    # Save to new file
    new_file_name = modelpath.parent / (modelpath.stem + "-cut-" + layername.replace('/', '-') + modelpath.suffix)
    tree.write(str(new_file_name), pretty_print=True)

    # Copy *.bin file with the new model name
    old_bin_path = modelpath.parent / (modelpath.stem + ".bin")
    new_bin_path = modelpath.parent / (new_file_name.stem + ".bin")
    if not new_bin_path.is_file():
        print(f"Copying weights from {old_bin_path} to {new_bin_path}")
        copyfile(str(old_bin_path), str(new_bin_path))
    else:
        print(f"Can't copy weights. File already exists: {new_bin_path}")
    
    return new_file_name


def delete_subgraph_files(subgraph_path):
    # Prepare file paths
    subgraph_path = Path(subgraph_path)
    bin_file_to_delete = subgraph_path.parent / (subgraph_path.stem + ".bin")

    # Delete XML file if exists
    if subgraph_path.is_file():
        print(f"Deleting XML file: {subgraph_path}")
        subgraph_path.unlink()
    else:
        print(f"XML file does not exist: {subgraph_path}")

    # Delete BIN file if exists
    if bin_file_to_delete.is_file():
        print(f"Deleting BIN file: {bin_file_to_delete}")
        bin_file_to_delete.unlink()
    else:
        print(f"BIN file does not exist: {bin_file_to_delete}")


def accuracy_check_for_subgraph_specified(subgraph_folder_cpu, subgraph_folder_npu, subgraph_file, output_folder, tol, dp):
    core = ov.Core()
    
    model_path_cpu = os.path.join(subgraph_folder_cpu, subgraph_file)
    model_path_npu = os.path.join(subgraph_folder_npu, subgraph_file)
    
    nodes = get_node_list(modelpath=model_path_cpu)
    results = []
    for node in nodes:
        print(f"\nChecking accuracy for {node}...")

        # Create subgraphs
        new_subgraph_path_cpu = create_new_subgraph(modelpath=model_path_cpu, layername=node)
        new_subgraph_path_npu = create_new_subgraph(modelpath=model_path_npu, layername=node)

        # Accuracy check
        results.append(subgraph_accuracy_check.accuracy_check(node, core, new_subgraph_path_cpu, new_subgraph_path_npu, tol, dp))

        print(f"Deleting {node}...")
        delete_subgraph_files(new_subgraph_path_cpu)
        delete_subgraph_files(new_subgraph_path_npu)

        print("\n")
    # Store results to CSV
    subgraph_accuracy_check.write_result(results=results, output_csv_filepath=os.path.join(output_folder, "result_" + subgraph_file.replace(".xml",".csv")), tol=tol, dp=dp)
    

def accuracy_check_for_subgraph(subgraph_folder_cpu, subgraph_folder_npu, output_folder, tol, dp):
    core = ov.Core()
    subgraph_files_cpu = [f for f in os.listdir(subgraph_folder_cpu) if f.endswith('.xml')]
    subgraph_files_npu = [f for f in os.listdir(subgraph_folder_npu) if f.endswith('.xml')]
    subgraph_files = list(set(subgraph_files_cpu) & set(subgraph_files_npu))

    for subgraph_file in subgraph_files:
        print(f"Examining nodes inside {subgraph_file}...")
        accuracy_check_for_subgraph_specified(subgraph_folder_cpu, subgraph_folder_npu, subgraph_file, output_folder, tol, dp)
        
        # model_path_cpu = os.path.join(subgraph_folder_cpu, subgraph_file)
        # model_path_npu = os.path.join(subgraph_folder_npu, subgraph_file)
       
        # nodes = get_node_list(modelpath=model_path_cpu)
        # results = []
        # for node in nodes:
        #     print(f"\nChecking accuracy for {node}...")

        #     # Create subgraphs
        #     new_subgraph_path_cpu = create_new_subgraph(modelpath=model_path_cpu, layername=node)
        #     new_subgraph_path_npu = create_new_subgraph(modelpath=model_path_npu, layername=node)

        #     # Accuracy check
        #     results.append(subgraph_accuracy_check.accuracy_check(node, core, new_subgraph_path_cpu, new_subgraph_path_npu, tol, dp))
            
        #     print(f"Deleting {node}...")
        #     delete_subgraph_files(new_subgraph_path_cpu)
        #     delete_subgraph_files(new_subgraph_path_npu)
            
        #     print("\n")
        # # Store results to CSV
        # subgraph_accuracy_check.write_result(results=results, output_csv_filepath=os.path.join(output_folder, "result_" + subgraph_file.replace(".xml",".csv")), tol=tol, dp=dp)
    print("\n============================================================================================\n")
    
    
# Main execution
if __name__ == "__main__":
    subgraph_folder_cpu = os.getenv('CPU_SUBGRAPH_FOLDER')
    subgraph_folder_npu = os.getenv('NPU_SUBGRAPH_FOLDER')
    output_folder = os.getenv('OUTPUT_FOLDER')
    tol = []  # Example tolerances for accuracy check
    dp = [4]  # Example DP values
    
    
    # check all subgraphs
    accuracy_check_for_subgraph(subgraph_folder_cpu, subgraph_folder_npu, output_folder, tol, dp)
    
    # check specified subgraph
    # accuracy_check_for_subgraph_specified(subgraph_folder_cpu, subgraph_folder_npu, "OpenVINO-EP-subgraph_30.xml", output_folder, tol, dp)