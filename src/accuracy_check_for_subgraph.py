import edit_xml
import accuracy_check
from lxml import objectify
from shutil import copyfile
from pathlib import Path
import openvino as ov
import os
import csv
from dotenv import load_dotenv

load_dotenv()

def get_matched_node_list(modelpath):
    modelpath = Path(modelpath)
    parser = objectify.makeparser(remove_comments=True)
    temp_tree = objectify.parse(str(modelpath), parser=parser)

    if temp_tree is None:
        raise Exception("Cannot parse model IR")

    excluded_names = {"OpenVINO-EP-subgraph", "sink_port", "GroupQueryAttention"}

    node_lst = [
        node.attrib['name']
        for node in temp_tree.iter()
        if 'name' in node.attrib and "/model/layers." in node.attrib['name'] and not any(excluded in node.attrib['name'] for excluded in excluded_names)
    ]

    return node_lst


def get_node_list(modelpath):
    modelpath = Path(modelpath)
    parser = objectify.makeparser(remove_comments=True)
    temp_tree = objectify.parse(str(modelpath), parser=parser)

    if temp_tree is None:
        raise Exception("Cannot parse model IR")

    excluded_names = {"OpenVINO-EP-subgraph", "sink_port", "GroupQueryAttention", "Constant", "weight_scales", "weight"}

    node_lst = [
        node.attrib['name']
        for node in temp_tree.iter()
        if 'name' in node.attrib and not any(excluded in node.attrib['name'] for excluded in excluded_names)
    ]

    return node_lst


def create_new_subgraph(modelpath, layername):
    modelpath = Path(modelpath)

    if modelpath.suffix != ".xml":
        raise Exception("Path is not to model IR file")

    if not layername:
        raise Exception("Empty layer name")

    parser = objectify.makeparser(remove_comments=True)
    tree = objectify.parse(str(modelpath), parser=parser)
    if tree is None:
        raise Exception("Cannot parse model IR")

    edit_xml.delLayers(tree, layername)

    new_file_name = modelpath.parent / (modelpath.stem + "-cut-" + layername.replace('/', '-') + modelpath.suffix)
    tree.write(str(new_file_name), pretty_print=True)

    old_bin_path = modelpath.parent / (modelpath.stem + ".bin")
    new_bin_path = modelpath.parent / (new_file_name.stem + ".bin")
    if not new_bin_path.is_file():
        print(f"Copying weights from {old_bin_path} to {new_bin_path}")
        copyfile(str(old_bin_path), str(new_bin_path))
    else:
        print(f"Can't copy weights. File already exists: {new_bin_path}")
    
    return new_file_name


def delete_subgraph_files(subgraph_path):
    subgraph_path = Path(subgraph_path)
    bin_file_to_delete = subgraph_path.parent / (subgraph_path.stem + ".bin")

    if subgraph_path.is_file():
        print(f"Deleting XML file: {subgraph_path}")
        subgraph_path.unlink()
    else:
        print(f"XML file does not exist: {subgraph_path}")

    if bin_file_to_delete.is_file():
        print(f"Deleting BIN file: {bin_file_to_delete}")
        bin_file_to_delete.unlink()
    else:
        print(f"BIN file does not exist: {bin_file_to_delete}")


def accuracy_check_for_subgraph(subgraph_folder_cpu, subgraph_folder_npu, subgraph_files, output_folder, tol, dp):
    core = ov.Core()

    for subgraph_file in subgraph_files:
        print(f"processing nodes in {subgraph_file}...")
        model_path_cpu = os.path.join(subgraph_folder_cpu, subgraph_file)
        model_path_npu = os.path.join(subgraph_folder_npu, subgraph_file)
        
        results = []
        
        if model_path_cpu != model_path_npu:
            nodes = get_matched_node_list(modelpath=model_path_cpu)
            for node in nodes:
                print(node)
                
            print("\n")
            
            for node in nodes:
                print(f"\nChecking accuracy for {node}...")

                new_subgraph_path_cpu = create_new_subgraph(modelpath=model_path_cpu, layername=node)
                new_subgraph_path_npu = create_new_subgraph(modelpath=model_path_npu, layername=node)

                results.append(accuracy_check.accuracy_check(node, core, new_subgraph_path_cpu, new_subgraph_path_npu, tol, dp))

                print(f"Deleting {node}...")
                delete_subgraph_files(new_subgraph_path_cpu)
                delete_subgraph_files(new_subgraph_path_npu)

                print("\n============================================================================================\n")
        else:
            nodes = get_node_list(modelpath=model_path_npu)
            for node in nodes:
                print(node)
                
            print("\n")
            
            for node in nodes:
                print(f"\nChecking accuracy for {node}...")
                
                new_subgraph_path_npu = create_new_subgraph(modelpath=model_path_npu, layername=node)
                
                results.append(accuracy_check.accuracy_check(node, core, new_subgraph_path_npu, new_subgraph_path_npu, tol, dp))
                
                print(f"Deleting {node}...")
                delete_subgraph_files(new_subgraph_path_npu)
                
                print("\n============================================================================================\n")
        
        accuracy_check.write_result(results=results, output_csv_filepath=os.path.join(output_folder, "result_" + subgraph_file.replace(".xml",".csv")), tol=tol, dp=dp)
    
def accuracy_check_for_subgraph_all(subgraph_folder_cpu, subgraph_folder_npu, output_folder, tol, dp):
    subgraph_files_cpu = {f for f in os.listdir(subgraph_folder_cpu) if f.endswith('.xml')}
    subgraph_files_npu = {f for f in os.listdir(subgraph_folder_npu) if f.endswith('.xml')}
    subgraph_files = subgraph_files_cpu & subgraph_files_npu
    accuracy_check_for_subgraph(subgraph_folder_cpu, subgraph_folder_npu, subgraph_files, output_folder, tol, dp)
        
    
# Main execution
if __name__ == "__main__":
    subgraph_folder_cpu = os.getenv('CPU_SUBGRAPH_FOLDER')
    subgraph_folder_npu = os.getenv('NPU_SUBGRAPH_FOLDER')
    output_folder = os.getenv('OUTPUT_FOLDER')
    tol = [0.01,0.001]
    dp = [4]    
    
    # check with respective subgraphs (matched nodes)
    # accuracy_check_for_subgraph_all(subgraph_folder_cpu, subgraph_folder_npu, output_folder, tol, dp)
    # accuracy_check_for_subgraph_specified(subgraph_folder_cpu, subgraph_folder_npu, "OpenVINO-EP-subgraph_30.xml", output_folder, tol, dp)
    
    # check with npu subgraphs (all nodes)
    # accuracy_check_for_subgraph_all(subgraph_folder_npu, subgraph_folder_npu, output_folder, tol, dp)
    
    subgraph_files = [  "OpenVINO-EP-subgraph_10.xml",
                        "OpenVINO-EP-subgraph_15.xml",
                        "OpenVINO-EP-subgraph_16.xml",
                        "OpenVINO-EP-subgraph_19.xml",
                        "OpenVINO-EP-subgraph_21.xml",
                        "OpenVINO-EP-subgraph_24.xml",
                        "OpenVINO-EP-subgraph_25.xml",
                        "OpenVINO-EP-subgraph_27.xml",
                        "OpenVINO-EP-subgraph_28.xml",
                        "OpenVINO-EP-subgraph_30.xml",
                        "OpenVINO-EP-subgraph_31.xml",
                        "OpenVINO-EP-subgraph_32.xml",
                        "OpenVINO-EP-subgraph_33.xml",
                        "OpenVINO-EP-subgraph_34.xml"  ]
    accuracy_check_for_subgraph(subgraph_folder_npu, subgraph_folder_npu, subgraph_files, output_folder, tol, dp)