import gradio as gr
import os
import torch

from model import create_vit_model
from timeit import default_timer as timer
from typing import Tuple, Dict

with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

vit, vit_transforms = create_vit_model(
    num_classes=len(class_names),
)

vit.load_state_dict(
    torch.load(
        f="pretrained_vit_feature_extractor_food101_20_percent.pth",
        map_location=torch.device("cpu"),
    )
)

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on an image and returns prediction and time taken"""

    start_time = timer()

    img = vit_transforms(img).unsqueeze(0)

    vit.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(vit(img), dim=1)

    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time


title = "Food Classification"

description = """A ViT feature extractor computer vision model to classify images of food into <a href="https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101.html" target="_blank">101 different classes</a>.

For full source code and details, check out my GitHub:
<a href="https://github.com/JankData/food-classification" target="_blank">https://github.com/JankData/food-classification</a>"""

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
)

demo.launch()