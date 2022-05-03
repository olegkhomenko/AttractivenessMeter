"""
🔥 This algorithm will give a real assessment of how attractive you are! 💖
"""
import argparse
import logging
import platform
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import clip
import torch
import torch.nn as nn
from deepface import DeepFace
from deepface.detectors import MediapipeWrapper
from PIL import Image, ImageDraw, ImageFont


# Configuration
@dataclass
class ConfigGlobal:
    BASE_PATH: Path = Path("./photos/")
    LOG_FILENAME: str = "AttractiveMeter.log"


logging.basicConfig(filename=ConfigGlobal.LOG_FILENAME, level=logging.DEBUG)
ConfigGlobal.BASE_PATH.mkdir(exist_ok=True)


# Helpers
def write_on_pil(img: Image.Image, text: str, font_size=80, color=(255, 255, 0)):
    """Function writes text on the image"""
    img = img.copy()
    _platform = platform.platform()
    if "macOS" in _platform:  # TODO: Better platform checks (Win)
        font = None
    else:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    draw = ImageDraw.Draw(img)

    x = y = 0
    for line in text.split("\n"):
        draw.text((x, y), line, color, font=font)
        y += font_size

    return img


# Classes
class PredictorCLIP(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        # Loading CLIP
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.detector = MediapipeWrapper.build_model()

        self.captions = [["handsome", "ugly"], ["beautiful", "ugly"]]

    @torch.no_grad()
    def predict_clip(self, image: Image.Image, text: List[str]):
        text = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=1, keepdim=True)

        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        logits_per_image, logits_per_text
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

    def detect_face_with_padding(self, img, padding=0.1):
        """Detects faces in an image. Returns first detected face. Returns None if no face detected."""
        try:
            w, h, _ = img.shape

            padding_coef_v = 1 + padding
            padding_coef_h = 1 + padding

            bbox = MediapipeWrapper.detect_face(self.detector, img)[0][1]

            l, t, r, b = bbox

            r += l
            b += t
            w = r - l
            h = b - t

            padding_w = int((w * padding_coef_h - w) // 2)
            padding_h = int((h * padding_coef_v - h) // 2)

            r += padding_w
            b += padding_h
            l -= padding_w
            t -= padding_h

            # FIXME: Fix bbox if it is out of image
            result = Image.fromarray(img[t:b, l:r][..., ::-1])

        except Exception as e:
            result = None

        return result

    def predict(self, img_path: str):
        """Predicts how attractive is the person on the image"""
        img = DeepFace.functions.load_image(img_path)
        pil_to_predict = self.detect_face_with_padding(img, 0.2)
        if pil_to_predict is None:
            return (None,) * 3

        gender = self.predict_clip(pil_to_predict, ["man", "woman"]).argmax()
        score = self.predict_clip(pil_to_predict, self.captions[gender])[0][0]

        # make text caption from raw number
        caption = f"{int(score * 100)}% " + (
            "attractive man" if gender == 0 else "beautiful girl"
        )

        pil_with_score = write_on_pil(
            pil_to_predict.resize((256, 256)), caption, font_size=20
        )

        return pil_with_score, score, caption


def analyze_photo(predictor_clip: PredictorCLIP, image_path: str):
    fname = str(uuid.uuid4())  # we will store all photos which were process
    fpath = Path(ConfigGlobal.BASE_PATH, fname + ".jpg").as_posix()
    img, score, caption = predictor_clip.predict(image_path)
    if all(el is not None for el in [img, score, caption]):
        img.save(fpath)
        print(caption)
    else:
        print("Sorry 😢. Cannot find any person on the photo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AttractiveMeter")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to image to be analyzed"
    )

    args = parser.parse_args()
    image_path = args.image_path

    if not Path(image_path).exists():
        sys.exit(f"File {image_path} does not exist")

    predictor_clip = PredictorCLIP()
    analyze_photo(predictor_clip=predictor_clip, image_path=image_path)
