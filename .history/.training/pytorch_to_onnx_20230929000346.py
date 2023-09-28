import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet50_Weights

# one by one
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image = Image.open("D:\\Python_Project\\DogVsCat_Classifier\\imgs\\terrier.jpg")
image = image.convert("RGB")

transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

image = transform(image)
image = image.to(device)

net_model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
num_ftrs = net_model.fc.in_features
net_model.fc = nn.Linear(num_ftrs, 35)
net_model = torch.load('cat_classification_resnet50.pth')
net_model = net_model.eval().to(device)
image = image.to(device)

output = net_model(image.unsqueeze(0))
print(output)

with torch.no_grad():
    torch.onnx.export(
        net_model,
        image.unsqueeze(0),
        "cat_classification_resnet50.onnx",
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
import onnx

onnx_model = onnx.load("cat_classification_resnet50.onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
