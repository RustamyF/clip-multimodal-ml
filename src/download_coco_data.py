import os
import torch
from zipfile import ZipFile
import urllib.request
import json
import collections

root_dir = "datasets"
annotations_dir = os.path.join(root_dir, "annotations")
images_dir = os.path.join(root_dir, "train2017")
annotation_file = os.path.join(annotations_dir, "captions_train2017.json")

# Download caption annotation files
if not os.path.exists(annotations_dir):
    annotation_zip_url = (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    annotation_zip_path = os.path.join(os.path.abspath("."), "captions.zip")
    urllib.request.urlretrieve(annotation_zip_url, annotation_zip_path)
    with ZipFile(annotation_zip_path, "r") as zip_ref:
        zip_ref.extractall(annotations_dir)
    os.remove(annotation_zip_path)

# Download image files
if not os.path.exists(images_dir):
    image_zip_url = "http://images.cocodataset.org/zips/train2017.zip"
    image_zip_path = os.path.join(os.path.abspath("."), "train2017.zip")
    urllib.request.urlretrieve(image_zip_url, image_zip_path)
    with ZipFile(image_zip_path, "r") as zip_ref:
        zip_ref.extractall(images_dir)
    os.remove(image_zip_path)

print("Dataset is downloaded and extracted successfully.")
