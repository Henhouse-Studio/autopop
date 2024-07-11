import torch
from torch import nn
from PIL import Image
from pathlib import PurePath
from torchvision import transforms
from models_mamba import VisionMamba
from huggingface_hub import snapshot_download
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Download the pretrained model
VIM_REPO = "hustvl/Vim-small-midclstok"
pretrained_model_dir = snapshot_download(repo_id=VIM_REPO)
MODEL_FILE = PurePath(pretrained_model_dir, "vim_s_midclstok_ft_81p6acc.pth")

# Initialize the model
model = VisionMamba(
    patch_size=16,
    stride=8,
    embed_dim=384,
    depth=24,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    final_pool_type="mean",
    if_abs_pos_embed=True,
    if_rope=False,
    if_rope_residual=False,
    bimamba_type="v2",
    if_cls_token=True,
    if_devide_out=True,
    use_middle_cls_token=True,
    num_classes=1000,
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224,
)
checkpoint = torch.load(str(MODEL_FILE), map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()
model.to("cuda")


# Define the image transformation function
def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_as_tensor = transforms.ToTensor()(image)
    normalized_tensor = transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    )(image_as_tensor)
    return normalized_tensor.unsqueeze(0).cuda()


# Define the function to calculate similarity (no norm needed here)
def calculate_image_similarity(model, image1_path, image2_path):

    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    with torch.no_grad():
        embedding1 = model(img1)
        embedding2 = model(img2)

    similarity = nn.functional.cosine_similarity(embedding1, embedding2)
    similarity = round(similarity.item() * 100, 2)

    return similarity


# Example usage
image1_path = "path1.jpg"
image2_path = "path2.jpg"
similarity = calculate_image_similarity(model, image1_path, image2_path)
print(f"Image similarity between '{image1_path}' and '{image2_path}': {similarity}%")
