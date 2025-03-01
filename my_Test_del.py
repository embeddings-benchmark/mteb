# from datasets import load_dataset
from __future__ import annotations

from transformers import ClapModel, ClapProcessor

model = ClapModel.from_pretrained("laion/clap-htsat-fused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

# Print the methods and attributes of the model
print("ClapModel methods:")
model_methods = [method for method in dir(model) if not method.startswith("_")]
for method in model_methods:
    print(f"- {method}")

# Print the methods and attributes of the processor
print("\nClapProcessor methods:")
processor_methods = [method for method in dir(processor) if not method.startswith("_")]
for method in processor_methods:
    print(f"- {method}")
