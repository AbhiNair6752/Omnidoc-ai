from datasets import load_dataset

dataset = load_dataset(
        "hf-tuner/rvl-cdip-document-classification"

)

print(
    dataset["train"].features["label"]
)