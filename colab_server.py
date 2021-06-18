import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import matplotlib.pyplot as plt
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
from flask import Flask, flash, render_template, request
# please note the import from `flask_uploads` - not `flask_reuploaded`!!
# this is done on purpose to stay compatible with `Flask-Uploads`
from flask_uploads import IMAGES, UploadSet, configure_uploads

# Make url public for colab
from flask_ngrok import run_with_ngrok


app = Flask(__name__, static_folder='generated')
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "images"
app.config["SECRET_KEY"] = os.urandom(24)
configure_uploads(app, photos)

test_dir ="./"

# Start ngrok when the app is running
run_with_ngrok(app)

def denorm(img_tensor):
    return img_tensor*0.5 + 0.5

@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        os.system('rm -rf images')
        os.system('rm -rf generated')
        os.system('mkdir images')
        os.system('mkdir generated')

        photos.save(request.files['photo'])
        flash("Photo saved successfully.", "p")
        # img = cv2.imread('images/'+str(request.files['photo'].filename))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        image = Image.open('images/'+str(request.files['photo'].filename))
        image = image.convert("RGB")
        transform=Compose([ 
                            tt.Resize((256,256),interpolation=Image.ANTIALIAS),
                            tt.CenterCrop(256),
                            tt.ToTensor(),
                            tt.Normalize(mean=(0.5,), std=(0.5,))])


        image = transform(image)
        image = torch.unsqueeze(image, 0)

        model = UNet(True)
        model.load_state_dict(torch.load("./generator.pth",map_location=torch.device('cpu')))
        # Rather use pickel model
        # filename = 'model_pickle.sav'
        # model = pickle.load(open(filename, 'rb'))
        prediction = model(image).detach()
        prediction = denorm(prediction.squeeze(0))
        fname = "/test-images.png"
        save_image(prediction, test_dir + fname)
        prediction = prediction.permute(1,2,0).numpy()
        # plt.imshow(image)
        
        gen_path_to_save = "generated/"+str(request.files['photo'].filename)
        orig_path_to_save = "generated/orig"+str(request.files['photo'].filename)
        plt.imsave(gen_path_to_save, prediction)
        plt.imsave(orig_path_to_save, denorm(image.squeeze(0)).permute(1,2,0).numpy())
        flash("Processed Successfully", "p")
        path_to_save = [orig_path_to_save, gen_path_to_save]
        return render_template('upload.html', img_path=path_to_save)
    return render_template('upload.html')

if __name__ == "__main__":
    # app.run(debug=True, use_reloader=True, threaded=True)
    app.run()