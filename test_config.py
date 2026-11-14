
import yaml

# Load configuration file
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Display project details
print("Project Name:", config["project"]["name"])

# Display dataset details
print("Dataset:", config["data"]["dataset_name"])

# Display emotion labels
print("Emotion Classes:", config["labels"]["classes"])

# Display model details
print("Model:", config["model"]["architecture"])

# Display training details
print("Epochs:", config["training"]["epochs"])
print("Batch Size:", config["training"]["batch_size"])

print("\nConfiguration loaded successfully!")
