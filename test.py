import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
import numpy as np
from ssd import SSD300
from encoder import DataEncoder
from PIL import Image, ImageDraw


# Load model
net = SSD300()
#net = net.cuda() # for gpu
checkpoint = torch.load('checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
net.eval()

# Load test image

list_image = os.listdir('/media/biometric/Data1/Ranjeet/NewPytorch/Journal_ROI/Testing_Data/Iris')
for image in list_image:
    try:
        image_path = '/media/biometric/Data1/Ranjeet/NewPytorch/Journal_ROI/Testing_Data/Iris/' + image
        img = Image.open(image_path)
        img1 = img.resize((300,300))
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        img1 = transform(img1)
# Forward
        loc, conf = net(Variable(img1[None,:,:,:], volatile=True))
# Decode
    
    
        data_encoder = DataEncoder()
        boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)

        draw = ImageDraw.Draw(img)
        for box in boxes:
            box[::2] *= img.width
            box[1::2] *= img.height
            draw.rectangle(list(box), outline='blue')
            #draw.ellipse(list(box), outline='blue')
        #img.save(image_path)
        img.show()
    except:
        print("No box")

    	
