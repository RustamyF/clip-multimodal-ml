import torch
import os
import torch
import subprocess
from torch.utils.data import DataLoader
from src.custom_model import CustomModel
from src.clip_dl import CocoDataset, Flickr30kDataset
from src.config import Config

coco_dataset = False
# Create the CLIP dataset
if coco_dataset:
    if not "datasets" in os.listdir():
        print("coco dataset is not downloaded! running the downloading script ....")
        subprocess.run(["python", "src/download_coco_data.py"])

    clip_dataset = CocoDataset(root_dir="datasets")
else:
    clip_dataset = Flickr30kDataset()


# Create the DataLoader
clip_dataloader = DataLoader(
    clip_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4
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
        loss, img_acc, cap_acc = model(image, text)

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
