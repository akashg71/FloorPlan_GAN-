# Import dependencies
import torch
import torchvision
import torchvision.transforms as tt
from torchvision.utils import save_image
from torchvision.transforms import Compose
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import PIL
from PIL import Image
import pickle 
from Model import UNet

test_dir = "./"
 # Load pretrained model
model = UNet(True)
model.load_state_dict(torch.load("./generator.pth",map_location=torch.device('cpu')))
# Rather use pickel model
# filename = 'model_pickle.sav'
# model = pickle.load(open(filename, 'rb'))
# Define function to display dataset

def denorm(img_tensor):
    return img_tensor*0.5 + 0.5


def predict(image, idx=0):
    image = image.convert("RGB")
    transform=Compose([ 
                        tt.Resize((256,256),interpolation=Image.ANTIALIAS),
                        tt.CenterCrop(256),
                        tt.ToTensor(),
                        tt.Normalize(mean=(0.5,), std=(0.5,))])

    image = transform(image)
    image = torch.unsqueeze(image, 0)
    prediction = model(image).detach()
    prediction = denorm(prediction.squeeze(0))
    fname = "/test-images-{0:0=4d}.png".format(idx)
    save_image(prediction, test_dir + fname)
    prediction = prediction.permute(1,2,0).numpy()
    # idx+=1
    return prediction

if __name__ == "__main__":
    with Image.open("./test.png") as image:
        prediction = predict(image)
        print("DONE")
        # print(prediction.shape)
