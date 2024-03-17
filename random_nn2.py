import os
import torch
from torch.nn import Sequential, Flatten, Linear
from random_nn import model_stealing_submission

if __name__ == "__main__":
    model = Sequential(
        Flatten(),
        Linear(3 * 32 * 32, 1028),
        Linear(1028, 512)
    )
    os.makedirs("modelstealing/models", exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        "submission.onnx",
        export_params=True,
        input_names=["x"],
    )
    submit = False
    if submit:
        model_stealing_submission("submission.onnx")
