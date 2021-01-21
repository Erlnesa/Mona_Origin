import torch

# Load the model
model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
model.eval()

from PIL import Image
import torch
from torchvision import transforms

input_image = Image.open(
    "C:/Users/76067/Pictures/data/+/illust_84936157_20201011_064536.jpg")  # load an image of your choice
preprocess = transforms.Compose([
    transforms.Resize(360),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a sigmoid on it.
probs = torch.sigmoid(output[0])  # Tensor of shape 6000, with confidence scores over Danbooru's top 6000 tags


import matplotlib.pyplot as plt
import json
import urllib, urllib.request

# Get class names
with urllib.request.urlopen("http://erlnesa.com/class_names_6000.json") as url:
    class_names = json.loads(url.read().decode())
# Plot image
plt.figure(figsize=(8, 6))
plt.imshow(input_image)
# plt.grid(False)
plt.axis('off')


def plot_text(thresh=0.2):
    tmp = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    txt = 'Predictions with probabilities above ' + str(thresh) + ':\n'
    for i in inds[0:len(tmp)]:
        txt += class_names[i] + ': {:.4f} \n'.format(probs[i].cpu().numpy())
    print(txt)
    plt.text(input_image.size[0] * 1.05, input_image.size[1] * 1.2, txt)



plot_text()
plt.tight_layout()
plt.show()
