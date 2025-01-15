import numpy as np
import openvino as ov

# Initialize OpenVINO core
core = ov.Core()

# Load the model
model_path = "model.xml"
compiled_model = core.compile_model(model_path, "CPU")

# Get input information
input_info = compiled_model.inputs
for i, input in enumerate(input_info):
    print(f"Input {i} expects shape: {input.shape}, type: {input.element_type}")

# Example input arrays (Manual)
input_array_1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]], dtype=np.float32)
input_array_2 = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    [1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 8.0],
    [8.0, 6.0, 4.0, 2.0, 9.0, 7.0, 5.0, 3.0, 1.0]
], dtype=np.float32)


# Example input arrays (Random)
# input_array_1 = np.random.rand(1, 9).astype(np.float32)
# input_array_2 = np.random.rand(4, 9).astype(np.float32)

# Create input tensors with the correct shapes and types
input_tensor_1 = ov.Tensor(array=input_array_1, shared_memory=True)
input_tensor_2 = ov.Tensor(array=input_array_2, shared_memory=True)

# Create an inference request
infer_request = compiled_model.create_infer_request()

# Set the input tensors
infer_request.set_input_tensor(0, input_tensor_1)
infer_request.set_input_tensor(1, input_tensor_2)

# Perform inference
infer_request.start_async()
infer_request.wait()

# Get the output tensor
output = infer_request.get_output_tensor()
output_buffer = output.data

print("Inference output:", output_buffer)
print("OK")





# pseudocode
"""
1. access file
2. get data
3. do loop
4. modify data
"""