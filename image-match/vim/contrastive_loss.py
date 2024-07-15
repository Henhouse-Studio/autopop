import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
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
model.to("cuda")


# Define the image transformation function
def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_as_tensor = transforms.ToTensor()(image)
    normalized_tensor = transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    )(image_as_tensor)
    return normalized_tensor


# Dataset for contrastive learning
class ContrastiveDataset(Dataset):
    def __init__(self, image_pairs: list):
        self.image_pairs = image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = preprocess_image(img1_path)
        img2 = preprocess_image(img2_path)
        return img1, img2


# Contrastive loss function (NT-Xent Loss)
class NTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):

        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):

        batch_size = z_i.shape[0]
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = (
            nn.functional.cosine_similarity(
                representations.unsqueeze(1), representations.unsqueeze(0), dim=2
            )
            / self.temperature
        )
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z_i.device)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z_i.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z_i.device)
        loss = self.criterion(logits, labels)

        return loss


# Function to calculate similarity (no norm needed here)
def calculate_image_similarity(model, img1, img2):

    with torch.no_grad():
        embedding1 = model(img1.unsqueeze(0).cuda())
        embedding2 = model(img2.unsqueeze(0).cuda())
    similarity = nn.functional.cosine_similarity(embedding1, embedding2)
    similarity = round(similarity.item() * 100, 2)

    return similarity


# Fine-tuning loop
def fine_tune_contrastive(model, dataloader, num_epochs=10, learning_rate=3e-4):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    contrastive_loss = NTXentLoss().cuda()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for img1, img2 in dataloader:
            img1, img2 = img1.cuda(), img2.cuda()
            optimizer.zero_grad()
            z_i = model(img1)
            z_j = model(img2)
            loss = contrastive_loss(z_i, z_j)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader)}")


# Example usage
image_pairs = [
    (
        "/home/gregorygo/autopop/images/rick_base.jpg",
        "/home/gregorygo/autopop/images/apple2.jpg",
    ),
]

dataset = ContrastiveDataset(image_pairs)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Fine-tune the model
fine_tune_contrastive(model, dataloader, num_epochs=10)
model.eval()

# Calculate similarity after fine-tuning
img1_path = "/home/gregorygo/autopop/images/rick_base.jpg"
img2_path = "/home/gregorygo/autopop/images/apple2.jpg"
img1 = preprocess_image(img1_path)
img2 = preprocess_image(img2_path)
similarity = calculate_image_similarity(model, img1, img2)
print(f"Image similarity between '{img1_path}' and '{img2_path}': {similarity}%")
