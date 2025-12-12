import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict 
import torchvision
import utilsfun
import parser

def main():
   args = parser.parse_args()
    #set the device to using GPU
# CPU or gpu
   if (args.device == 'gpu'):
       torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       args.device = 'cuda'
   else:
        args.device = 'cpu'
    # Load and preprocess the data
   trainloader, testloader, validloader, train_data = utilsfun.load_data(args.data_dir)

    # Setup the model, criterion, and optimizer
   model, criterion, optimizer =utilsfun.nn_setup(args.arch, args.lr)

    # Train the model
   utilsfun.train_model(model, trainloader, validloader, criterion, optimizer, args.device, args.epochs)
   utilsfun.evaluate_model(model, testloader, criterion, device='cuda')


    # Save the trained model checkpoint
    #utilsfun.save_checkpoint(model, args.save_dir)
    # TODO: Save the checkpoint 
   model.class_to_idx = train_data.class_to_idx
   checkpoint = {
        'arch': args.arch,  
        'input_size': args.input_layers, 
        'learning_rate':args.lr,
        'optimizer_dict': optimizer.state_dict(),
        'epochs':args.epochs,
        'hidden_layers': args.hidden_layers, 
        'output_layers': args.output_layers,  
        'dropout': args.dropout_rate,  
        'model_state_dict': model.state_dict(),
        'topk':args.topk,
        'class_to_idx': model.class_to_idx
    }
   torch.save(checkpoint, args.save_dir)
   print("Checkpoint saved successfully!")


if __name__ == "__main__":
    main()