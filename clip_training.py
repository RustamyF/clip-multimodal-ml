import torch
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.custom_model import CustomModel
from src.clip_dl import CLIPDataset
from src.config import BATCH_SIZE

# Define the transformation for the images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

root_dir = "datasets"
annotations_dir = os.path.join(root_dir, "annotations")
annotation_file = os.path.join(
    annotations_dir, "annotations", "captions_train2017.json"
)
# Create the CLIP dataset
clip_dataset = CLIPDataset(
    root_dir="datasets", annotation_file=annotation_file, transform=transform
)

# Create the DataLoader
clip_dataloader = DataLoader(
    clip_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Create an instance of your model
model = CustomModel().to(device)

# Define optimizer
optimizer = torch.optim.Adam(
    [
        {"params": model.vision_encoder.parameters()},
        {"params": model.caption_encoder.parameters()},
    ],
    lr=model.lr,
)


# Dummy training and validation loops
num_epochs = 5
batch_zero = True
for epoch in range(num_epochs):
    model.train()
    for batch in clip_dataloader:
        image = batch["image"].to(device)
        text = batch["caption"]
        # images, text = batch
        loss, img_acc, cap_acc = model.common_step((image, text))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_zero:
            print(f"Epoch [{0}/{num_epochs}], Batch Loss: {loss.item()}")
            batch_zero = False

    # Print training statistics
    print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item()}")

print("Training complete.")
