
import torch

from models.vgg import VGG

# Load the PyTorch model

checkpoint = torch.load('logs/model_best.pth')

model = VGG()
model.load_state_dict(checkpoint['state_dict'])

# Set the input shape of the model
input_shape = (1, 3, 32, 32)  # (batch_size, channels, height, width)

# Set the output file path
output_path = "logs/vgg11.onnx"

# Export the model to ONNX format
torch.onnx.export(
    model,                    # PyTorch model to be exported
    torch.randn(input_shape), # Example input tensor
    output_path,              # Output file path
    verbose=True)             # Print the ONNX graph to console
