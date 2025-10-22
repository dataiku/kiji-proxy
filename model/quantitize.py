import os

import onnx
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Load the PII model and export to ONNX
model_path = "/Users/hannes/Private/yaak-proxy/pii_model"
output_path = "/Users/hannes/Private/yaak-proxy/pii_onnx_model"

print("Loading PII model and exporting to ONNX...")
model = ORTModelForTokenClassification.from_pretrained(
    model_path, export=True, provider="CPUExecutionProvider"
)

# Quantize the model
print("Quantizing the model...")
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
quantizer.quantize(save_dir=output_path, quantization_config=qconfig)

# Load the quantized ONNX model and print inputs/outputs
print(f"\nLoading quantized ONNX model from {output_path}...")
onnx_model_path = os.path.join(output_path, "model_quantized.onnx")
if os.path.exists(onnx_model_path):
    model_onnx = onnx.load(onnx_model_path)
    print("Inputs:", [input.name for input in model_onnx.graph.input])
    print("Outputs:", [output.name for output in model_onnx.graph.output])
else:
    print(f"Quantized model not found at {onnx_model_path}")
    # Try to find any .onnx file in the output directory
    for file in os.listdir(output_path):
        if file.endswith(".onnx"):
            onnx_file = os.path.join(output_path, file)
            print(f"Found ONNX model: {onnx_file}")
            model_onnx = onnx.load(onnx_file)
            print("Inputs:", [input.name for input in model_onnx.graph.input])
            print("Outputs:", [output.name for output in model_onnx.graph.output])
            break
