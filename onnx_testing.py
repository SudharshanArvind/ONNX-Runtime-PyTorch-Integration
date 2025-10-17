import torch
import torch.nn as nn
import time
import torch.onnx

import onnxruntime as ort
import numpy as np

print(torch.cuda.is_available())

# Define a simple feedforward network
class SimpleNet(nn.Module):
  def __init__(self):
    super(SimpleNet, self).__init__()
    # Define Layers
    self.fc1=nn.Linear(2,2) # Input to Layer 1
    self.fc2=nn.Linear(2,3) # Layer 1 to Layer 2
    self.fc3=nn.Linear(3,4) # Layer 2 to Layer 3 (output)

  def forward(self, x):
    # Forward pass through the network
    x = torch.sigmoid(self.fc1(x))
    x = torch.sigmoid(self.fc2(x))
    x = self.fc3(x) # Output layer has no activation
    return x

# Instantiate the network
model = SimpleNet()

# Example input for inference
example_input = torch.tensor([[1.0, 2.0]])

# Perform inference
output = model(example_input)

# Print the inference output
print("PyTorch Inference Output: ", output.detach().numpy())

# Specify the path for the ONNX model file
onnx_model_path = "simple_model.onnx"

# Convert the PyTorch model to ONNX
torch.onnx.export(
    model,                                # model being exported
    example_input,                        # model input (or a tuple for multiple inputs)
    onnx_model_path,                      # where to save the model (can be a file or file-like object)
    input_names=["input"],                # the model's input names
    output_names=["output"],              # the model's output names
)

print(f"Model successfully converted to ONNX: {onnx_model_path}")

# Load the ONNX model
onnx_model_path = "simple_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Prepare sample input data (same shape as the PyTorch model)
onnx_input = np.array([[1.0, 2.0]], dtype = np.float32)

# Run inference on the ONNX model
onnx_output = ort_session.run(None, {"input": onnx_input})

# Print the ONNX inference result
print("ONNX Inference Output: ", onnx_output)

time_sum = 0
n_iters = 1000
onnx_model_path = "simple_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

for i in range(n_iters):
  onnx_input = np.random.rand(1, 2).astype(np.float32)
  t0 = time.time()
  onnx_output = ort_session.run(None, {"input": onnx_input})
  time_sum += (time.time() - t0)

print(f"Total time: {time_sum/n_iters}")