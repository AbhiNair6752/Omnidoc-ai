from datasets import load_dataset

DATASET_NAME = "hf-tuner/rvl-cdip-document-classification"

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)

print(dataset)

print("\nNumber of images:")
print(len(dataset))

print("\nFeatures:")
print(dataset.features)

print("\nFirst example:")
print(dataset[0])