# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision import models, transforms, datasets
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fine-tune BERT model for sentiment analysis
class SentimentAnalysisModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Custom dataset for image classification
class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [f"{image_folder}/{img}" for img in os.listdir(image_folder)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# Convolutional Neural Network for image classification
class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageClassificationModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.cnn(x)

# Set up data transforms for image classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image classification dataset
image_dataset = CustomImageDataset(image_folder='path/to/cifar10_subset', transform=transform)
image_loader = DataLoader(image_dataset, batch_size=64, shuffle=True)

# Initialize and train image classification model
image_model = ImageClassificationModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(image_model.parameters(), lr=0.001)

for epoch in range(5):
    for images in image_loader:
        images = images.to(device)
        labels = torch.randint(0, 10, (images.size(0),)).to(device)

        optimizer.zero_grad()
        outputs = image_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Initialize and fine-tune sentiment analysis model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dummy data for illustration
text_data = ["This is a positive sentence.", "This is a negative sentence."]
labels = torch.tensor([1, 0])

# Tokenize input text
input_ids = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True)['input_ids']

# Initialize and train sentiment analysis model
sentiment_model = SentimentAnalysisModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sentiment_model.parameters(), lr=0.0001)

for epoch in range(3):
    optimizer.zero_grad()
    outputs = sentiment_model(input_ids, attention_mask=(input_ids != 0).long())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Example usage
image_path = 'path/to/your/image.jpg'
image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
image_prediction = image_model(image).argmax().item()

text_data = "This is a positive example."
input_ids = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(device)
text_prediction = sentiment_model(input_ids, attention_mask=(input_ids != 0).long()).argmax().item()

print(f"Image Prediction: Class {image_prediction}")
print(f"Text Prediction: Class {text_prediction}")
