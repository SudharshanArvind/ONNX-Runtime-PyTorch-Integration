# ONNX-Runtime-PyTorch-Integration
This repository demonstrates how to export a simple PyTorch neural network model to the ONNX format and perform efficient inference using ONNX Runtime.
It’s a beginner-friendly guide to understanding ONNX conversion, runtime loading, and inference benchmarking.

# 🚀 Features

* Defines and runs a basic feedforward neural network using PyTorch

* Converts the trained model to ONNX format

* Loads and performs inference using ONNX Runtime

* Benchmarks average inference time across multiple iterations


# 🧰 Requirements

Install all dependencies with:

```
pip install -r requirements.txt
```

Or manually:
```
pip install torch onnx onnxruntime numpy
```

# 📘 Usage

Clone this repository:
```
git clone https://github.com/<your-username>/onnx-runtime-tutorial.git
cd onnx-runtime-tutorial
```
Run the script:
```
python onnx_testing.py
```
Expected outputs:

* PyTorch inference results

* Confirmation that model was successfully converted to ONNX

* ONNX inference results

* Average inference time across multiple iterations

# 🧩 Example Output
```
PyTorch Inference Output:  [[0.55 0.43 0.61 0.52]]
Model successfully converted to ONNX: simple_model.onnx
ONNX Inference Output:  [array([[0.55, 0.43, 0.61, 0.52]], dtype=float32)]
Total time: 0.00012345
```

# 🧑‍💻 Author

Sudharshan Arvind

B.Tech Student

# 📜 License

This project is released under the MIT License.


# 🧩 4️⃣ Directory Structure

Here’s how your repo should look:

onnx-runtime-tutorial/

├── license

├── README.md

├── onnx_testing.py

└── requirements.txt

