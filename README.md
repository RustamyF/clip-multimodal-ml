# CLIP
This repository presents CLIP model training and serving. CLIP, wich stands for Contrastive Language-Image Pretraining,  is a deep learning model developed by OpenAI in 2021. It's a powerful vision-and-language model that bridges the gap between images and their textual descriptions.

![App Screenshot](assets/CLIP.png)

## CLIP Training
To run the training pipeline for CLIP, run the following command:

```python
python clip_training.py
```
by default, the training will use fliker30k dataset. For training on COCO dataset, change the `coco_dataset = False` to True in the `clip_training.py file`.
