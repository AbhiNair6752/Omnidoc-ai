from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

DATASET_NAME = "hf-tuner/rvl-cdip-document-classification"

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])

class DocumentDataset(Dataset):

    def __init__(self, dataset, transform=None):

        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        item = self.dataset[index]

        image = item["image"]
        label = item["label"]

        if self.transform:

            image = self.transform(image)

        return image, label


document_dataset = DocumentDataset(
    dataset,
    transform=transform
)

print("Dataset size:", len(document_dataset))


image, label = document_dataset[0]

print("Image Tensor shape:", image.shape)

print("Label:", label)


loader = DataLoader(
    document_dataset,
    batch_size=32,
    shuffle=True
)

images, labels = next(iter(loader))

print("Batch image shape:", images.shape)

print("Batch label shape:", labels.shape)

print("Labels:", labels)