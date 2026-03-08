import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import rse

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(
    r"E:\Volleyballey\rally_dataset",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

model = rse.build_model().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch", epoch, "loss", loss.item())