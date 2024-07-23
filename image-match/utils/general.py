import torch
import torchvision.transforms.transforms as transform
from PIL import Image


def preprocess_image(
    image_path: str,
    preprocess: transform.Compose,
    device: str = "cpu",
):

    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)


# Function for calculating similarity
def calculate_image_similarity(
    model, img1_path, img2_path, preprocess: transform.Compose, device: str = "cpu"
):

    # Image Preprocessing
    img1 = preprocess_image(img1_path, preprocess, device)
    img2 = preprocess_image(img2_path, preprocess, device)

    # Getting the features
    with torch.no_grad():
        image_features1 = model.encode_image(img1)
        image_features2 = model.encode_image(img2)

    # Normalize feature vectors
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features1 @ image_features2.T).item()

    return round(similarity, 2)
