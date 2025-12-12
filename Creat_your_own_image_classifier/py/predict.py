import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict 
import torchvision
import parser
import json
import funforpredict
################################################################
#python predict.py --image_path 'flowers/test/19/image_06197.jpg' --save_dir checkpoint.pth
def main():
     args = parser.parse_args()
    #mapping for labels for the predictions

     with open(args.category_file_path, 'r') as f:
        cat_to_name = json.load(f)
    
    #load the weigts
     model=funforpredict.load_checkpoint(args.save_dir)
      
     funforpredict.preprocess_image(args.image_path)
     print(funforpredict.predict(args.image_path,model, args.topk, args.device, cat_to_name))
if __name__ == "__main__":
   main()          

     
    