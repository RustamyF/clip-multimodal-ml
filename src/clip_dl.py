import os
import json
from PIL import Image
from torch.utils.data import Dataset
import collections
from torchvision import transforms
from datasets import load_dataset
import torch


class CocoDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        annotations_dir = os.path.join(root_dir, "annotations")
        annotation_file = os.path.join(
            annotations_dir, "annotations", "captions_train2017.json"
        )

        self.caption_list, self.image_path_list = self.load_annotations(annotation_file)

    def load_annotations(self, annotation_file):
        with open(annotation_file, "r") as f:
            annotations = json.load(f)["annotations"]

        image_path_to_caption = collections.defaultdict(list)
        for element in annotations:
            caption = f"{element['caption'].lower().rstrip('.')}"
            image_path = os.path.join(
                self.root_dir,
                "train2017",
                "train2017",
                "%012d.jpg" % (element["image_id"]),
            )
            image_path_to_caption[image_path].append(caption)
        image_paths = list(image_path_to_caption.keys())
        caption_list, image_path_list = self.training_list(
            image_paths, image_path_to_caption
        )

        return caption_list, image_path_list

    def training_list(self, image_paths, image_path_to_caption):
        captions_per_image = 2
        caption_list = []
        image_path_list = []
        for image_path in image_paths:
            captions = image_path_to_caption[image_path][:captions_per_image]
            caption_list.extend(captions)
            image_path_list.extend([image_path] * len(captions))

        return caption_list, image_path_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        caption = self.caption_list[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}


class Flickr30kDataset(Dataset):
    def __init__(self):
        self.dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.cap_per_image = 2

    def __len__(self):
        return self.dataset.num_rows["test"] * self.cap_per_image

    def __getitem__(self, idx):
        original_idx = idx // self.cap_per_image
        # image_path = self.dataset[idx]["image_path"]
        image = self.dataset["test"][original_idx]["image"].convert("RGB")
        image = self.transform(image)

        # You might need to adjust the labels based on your task
        caption = self.dataset["test"][original_idx]["caption"][
            idx % self.cap_per_image
        ]

        return {"image": image, "caption": caption}
