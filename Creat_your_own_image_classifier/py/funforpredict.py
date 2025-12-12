import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict 
import torchvision
import numpy as np
from PIL import Image

import utilsfun
import parser
#define fun for define image
#first fun for processing the image
args = parser.parse_args()

def preprocess_image(image_path):
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Resize the image to have the shortest side of 256 pixels
    shortest_side = 256
    new_size = (shortest_side, int(shortest_side * image.size[1] / image.size[0])) if image.size[0] < image.size[1] else (int(shortest_side * image.size[0] / image.size[1]), shortest_side)
    image.thumbnail(new_size)
    
    # Crop the center 224x224 portion of the image
    crop_size = 224
    left = (image.width - crop_size) / 2
    top = (image.height - crop_size) / 2
    right = (image.width + crop_size) / 2
    bottom = (image.height + crop_size) / 2
    image = image.crop((left, top, right, bottom))

    # Convert the PIL image to a Numpy array
    np_image = np.array(image) / 255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the color channel to be the first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert the Numpy array to a PyTorch tensor
    torch_image = torch.from_numpy(np_image).float()
    
    return torch_image
####################################################################################

#define funcation for predict data
#second fub for predict the image


##########################################################
#third fun for load the checkpoint
#model=utilsfun.nn_setup(args.arch, args.lr)


#================================================#

def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath)

    # Create a new model instance based on the architecture specified in the checkpoint
    model, criterion, optimizer = utilsfun.nn_setup(args.arch, args.lr)

    # Load the saved model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the class to index mapping
    model.class_to_idx = checkpoint['class_to_idx']

    return model



#============================================================#
#define funcation for predict data
def predict(image_path, model, topk, device, cat_to_name):

    # Set the model to evaluation mode
    model.eval()

    # Load and preprocess the image
    image_tensor = preprocess_image(image_path)
    
  
    img_tensor = image_tensor.unsqueeze(0)

    # Move the tensor to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)

    # Perform the prediction
    with torch.no_grad():
        output = model(img_tensor)

    # Get the topk predictions
    probabilities, classes = torch.topk(F.softmax(output, dim=1), topk)

    # Convert the tensors to numpy arrays
    probabilities = probabilities.cpu().numpy().squeeze()
    classes = classes.cpu().numpy().squeeze()

    # Map indices to class labels
    class_to_idx = model.class_to_idx
    idx_to_class= {v:cat_to_name[k] for k, v in model.class_to_idx.items()}
    #idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_labels = [idx_to_class[idx] for idx in classes]

    return probabilities, class_labels
   
    
    





