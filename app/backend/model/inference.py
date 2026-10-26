import torch
from torchvision import transforms
from PIL import Image
import io
import os
from .vit_model import VisionTransformers

IMG_SIZE = 256
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vit_thyroid_usg.pth")

transform = transforms.Compose([
	transforms.Resize((IMG_SIZE, IMG_SIZE)),
	transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor()
])

def load_model(model_path=MODEL_PATH):
	model = VisionTransformers(img_size=256, patch_size=16, in_chans=1, n_classes=2)
	state_dict = torch.load(model_path, map_location=torch.device('cpu'))
	model.load_state_dict(state_dict)
	model.eval()
	return model

def predict_image(model, image_bytes):
	image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
	input_tensor = transform(image).unsqueeze(0)
	with torch.no_grad():
		output = model(input_tensor)
		pred = torch.argmax(output, dim=1).item()
	return int(pred)
