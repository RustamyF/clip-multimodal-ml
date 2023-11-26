import os
import json
from PIL import Image
from torch.utils.data import Dataset
import collections


class CLIPDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
