# This script analyzes subgraph results from CSV files in the specified directory.
# It identifies subgraphs with non-positive absolute errors and finds the subgraph with the highest absolute error.

import os
import pandas as pd

def get_csv_files(directory, exclude_files):
    """
    Get a list of CSV files in the directory, excluding the specified files.
    
    Args:
    directory (str): The directory to search for CSV files.
    exclude_files (list): The files to exclude from the search.
    
    Returns:
    list: A list of CSV file names.
    """
    return [f for f in os.listdir(directory) if f.endswith('.csv') and f not in exclude_files]

def analyze_subgraphs(directory):
    """
    Analyze subgraphs from CSV files in the specified directory.
    
    Args:
    directory (str): The directory containing the CSV files.
    """
    exclude_files = ['result_subgraphs.csv', 'subgraph_nodes_with_highest-abs-err.csv', 'subgraph_nodes_with_positive_abs_err.csv', '']  # Add more files to exclude if needed
    csv_files = get_csv_files(directory, exclude_files)
    subgraphs_with_highest_err_lst = []
    subgraphs_lst = []

    for csv_file in csv_files:
        print(f"Checking {csv_file} .....")
        file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(file_path)

        subgraph_with_highest_err = None
        highest_error = 0.0
        
        # Get subgraph max absolute error column
        sub_subgraphs, abs_errors = df['Subgraph'], df['5 highest absolute error']
        
        # Get subgraphs with positive absolute error at index 0
        print("Subgraph nodes with positive absolute error")
        for subgraph, abs_error in zip(sub_subgraphs, abs_errors):
            try:
                # Convert the absolute error to a list of floats
                abs_error_list = [float(error) for error in abs_error.split(',')] if ',' in abs_error else [float(abs_error)]
            except ValueError:
                continue  # Skip this subgraph if abs_error is not a valid float
        
            if abs_error_list[0] > 0.0:
                print([subgraph, abs_error])
                subgraphs_lst.append([csv_file.replace("result_",""), subgraph, abs_error])
                if highest_error < abs_error_list[0]:
                    highest_error = abs_error_list[0]
                    subgraph_with_highest_err = [subgraph, abs_error]
                    
        if subgraph_with_highest_err:
            subgraphs_with_highest_err_lst.append([csv_file.replace("result_","")] + subgraph_with_highest_err)

    # Print the subgraph with the highest absolute error for each CSV file
    print(f"\Subgraphs node with highest abs error")
    for subgraph_with_high_err in subgraphs_with_highest_err_lst:
        print(subgraph_with_high_err)
        
    # Save the subgraph with the highest absolute error for each CSV file to a new CSV file
    output_df = pd.DataFrame(subgraphs_with_highest_err_lst, columns=['subgraph', 'highest error node', '5 absolute error'])
    output_file_path = os.path.join(directory, 'subgraph_nodes_with_highest-abs-err.csv')
    output_df.to_csv(output_file_path, index=False)
    print(f"\nSubgraphs with highest errors saved to {output_file_path}")
    
    # Save all subgraphs with positive absolute error to a new CSV file
    subgraphs_output_df = pd.DataFrame(subgraphs_lst, columns=['subgraph', 'node', '5 absolute error'])
    subgraphs_output_file_path = os.path.join(directory, 'subgraph_nodes_with_positive_abs_err.csv')
    subgraphs_output_df.to_csv(subgraphs_output_file_path, index=False)
    print(f"\nSubgraphs with positive absolute errors saved to {subgraphs_output_file_path}")
     
            
if __name__ == "__main__":
    directory = 'C:\\projects\\subgraph_accuracy_check\\results\\npu-subgraphs'
    analyze_subgraphs(directory)