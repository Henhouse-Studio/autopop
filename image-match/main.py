import torch
import numpy as np
import face_recognition
from clip import clip
from utils.general import calculate_image_similarity
from utils.face_detection import get_face_encoding, display_face_locations


def main(image1_path: str, image2_path: str, verbose: bool = False):

    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # See if faces are present
    enc_1, loc_1 = get_face_encoding(image1_path)
    enc_2, loc_2 = get_face_encoding(image2_path)

    # Handle cases depending on if faces are present
    if (enc_1 is None) or (enc_2 is None):
        print("No faces detected.")
        similarity = calculate_image_similarity(
            model, image1_path, image2_path, preprocess, device
        )

    else:
        print("Faces found.")
        if verbose:
            display_face_locations(image1_path, loc_1)
            display_face_locations(image2_path, loc_2)

        similarity = np.max(1 - face_recognition.face_distance(enc_2, enc_1[0]))
        similarity = round(similarity * 100, 2)

    return similarity


# Example usage
if __name__ == "__main__":

    # Get the images
    image1_path = "images/rutte1.jpg"
    image2_path = "images/tweede_kamer.jpg"

    # Test
    similarity = main(image1_path, image2_path, verbose=True)
    print(
        f"Image similarity between '{image1_path}' and '{image2_path}': {similarity}%"
    )
