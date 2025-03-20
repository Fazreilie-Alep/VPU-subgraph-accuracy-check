# Subgraph Accuracy Comparison for NPU and CPU Inference

## Problem Statement

When inferring a deep learning (DL) model that is compiled for an AI accelerator (CPU, GPU, NPU), the output for the same input/prompt should ideally remain consistent across different hardware platforms. However, an issue was observed with the ONNX Phi-3 model, where the output on the NPU was different from the output on the CPU. This discrepancy can negatively impact the reliability of AIPC (AI-powered Computing Systems).

The root cause of this problem was traced to a subgraph within the compiled model, which generated different outputs on the NPU compared to the CPU. Consequently, the final output was affected by this inconsistency.

## Solution

To resolve this issue, this repository was created to facilitate the comparison of subgraphs from the model's inference on different hardware platforms. The goal is to identify problematic subgraphs that lead to inconsistent outputs.

The process involves comparing the output of subgraphs between the NPU and CPU. If the outputs match, it is labeled as a success. Otherwise, it is marked as a failure, helping developers pinpoint and debug problematic subgraphs.

### Key Idea
- Isolate the problematic subgraph to CPU using the NPUW plugin to investigate further.

## Methods of Comparison

There are two primary methods available in this repository for comparing subgraphs:

### 1. Comparison Per Subgraph
- This method compares multiple subgraphs at a time.
- Refer to `src/accuracy_check_per_subgraph.py` for implementation.

### 2. Comparison for Subgraph
- This method compares one subgraph and iteratively creates smaller subgraphs within it to compare.
- Refer to `src/accuracy_check_for_subgraph.py` for implementation.

### Subgraph Slicing
- In Method 2, the subgraph is sliced using the `src/edit_xml.py` script, which is part of the OV (OpenVINO) code repository.

## Directory Structure

