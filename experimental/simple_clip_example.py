import torch
import clip
from PIL import Image
import torch.nn.functional as F


def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0


def image_gen(images):
    image = torch.empty(0)
    for im in images:
        image1 = preprocess(Image.open("assets/" + im)).unsqueeze(0)
        image = torch.cat((image, image1), dim=0)
    return image.to(device)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

images = ["CLIP.png", "dog.jpg", "cat.jpg", "map.jpg"]
true_text = [
    "diagram of CLIP model",
    "photo a dog sitting on grass",
    "photo of a cat",
    "photo of the world map",
]
wrong_text = true_text[::-1]


image = image_gen(images=images)

for captions in [true_text, wrong_text]:
    text = clip.tokenize(captions).to(device)

    # with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    similarity = text_features @ image_features.T
    loss = clip_loss(similarity)
    print(loss)
