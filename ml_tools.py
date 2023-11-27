from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection


processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")


def get_embedding(fname):
    image1 = Image.open(fname)
    inputs = processor(images=image1, return_tensors="pt")
    return model(**inputs).image_embeds.detach().numpy()[0]
