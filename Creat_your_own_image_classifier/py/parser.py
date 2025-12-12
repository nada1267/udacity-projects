
import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict 
import torchvision
def parse_args():
    parser = argparse.ArgumentParser(
    prefix_chars='-+',  # Set the prefix character for optional arguments to '-' or '+'
    allow_abbrev=True,  # Allow abbreviations for long option names
    description='HI there program for tranning your classfier'
)
    #set the dir you want to take data from
    parser.add_argument( '--data_dir',
                        type=str, 
                        default='flowers',
                        help='Specify the location of your data (e.g., flowers)')
 
    #set architecture you want to use it for training our model
    parser.add_argument('--arch',
                        type=str,
                        default='vgg16',
                        choices=['vgg16','resnet18'],
                        help='Detemine the architecture you want to train your model !')
    # parser for setting learning rate parameter
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='set value for learning rate parameter prefer ')
    #parser for chosing runtime type
    parser.add_argument('--device',
                        type=str,
                        default='gpu',
                        choices=['gpu','cpu'],
                        help ='determine your run time:CPU/GPU (prefer to use GPU)'
                        
    )
    #parser for chosing the number of epochs
    parser.add_argument('--epochs',
                        type = int,
                        default=1,
                        help = 'Determinr the number of epochs you want to train your model with')
    #parser for set the input layer size
    parser.add_argument('--input_layers',
                        type = int,
                        default = 2048,
                        help = 'Determine the number of input layers')
    #parser for set the hidden layer
    parser.add_argument('--hidden_layers',
                        type=int,
                        default=256,
                        help='Determine the number of hidden layers')
    #parser for determine the number of output layer
    parser.add_argument('--output_layers',
                        type=int,
                        default=102,
                        help='Determine the number of output layers')
    #parser for determine the dropout rate
    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.5,
                        help='Determine the dropout rate')
    #parser for determine the number of topks
    parser.add_argument('--topk',
                        type=int,
                        default=5,
                        help='Determine the number of topks')
    parser.add_argument('--category_file_path',
                         type = str, 
                        default = './cat_to_name.json',
                        help = 'path for catgory file labels')
    #parser for determine the path for save checkpoint
    parser.add_argument(
        
        '--save_dir',
        type=str,
        default='./checkpoint.pth', 
        help='path to save your checkpoint.')
    
    #parser for catch the image path
    parser.add_argument('--image_path',
                        type = str,
                        default = './flowers/test/19/image_06197.jpg',
                        help = 'the path for image you want to predict')
    return parser.parse_args()
                   