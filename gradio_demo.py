import pandas as pd
from predict import PredictImage
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import gradio as gr


transforms = A.Compose(
    [A.Resize(896, 896, always_apply=True), A.Normalize(), ToTensorV2()]
)


model_path = "pretrained_models/tablenet_baseline_adam_gradclipping.ckpt"

examples = [
    "examples/10.1.1.160.708_11.bmp",
    "examples/10.1.1.193.1812_24.bmp",
    "examples/10.1.1.1.2071_4.bmp",
]

predictor = PredictImage(model_path, transforms)


def predict_function(image):
    """
    Function to predict table content using PredictImage

    Args:
        Image (pil): Image to predict content

    Returns List[pd.DataFrame]: Contents of table in dataframe format
    """
    output = predictor.predict(image)
    return output[0] if output else pd.DataFrame()


gr.Interface(
    predict_function,
    gr.components.Image(type="pil"),
    gr.Dataframe(type="pandas", label="Dataframe"),
    title="Pytorch Lightning TableNet Demo",
    examples=examples,
).launch(server_name="0.0.0.0", server_port=7861)
