import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict 
import torchvision

#first fun for load data
def load_data(data_dir='flowers'):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)

    return trainloader, testloader, validloader, train_data
#==========================================================================================#
#second fun
def nn_setup(structure='vgg16', lr=0.001, input_size=25088):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'resnet18':
        model = models.resnet18(pretrained=True)
    # Add more elif blocks for other supported architectures
    else:
        raise ValueError("Unsupported model architecture: {}".format(structure))

    # Freeze parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False

    # Modify classifier with dropout layers
    classifier_layers = [
        ('fc1', nn.Linear(input_size, 2048)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(2048, 256)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(256, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]

    model.classifier = nn.Sequential(OrderedDict(classifier_layers))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion, optimizer
#==============================================================================#
#third fun
def train_model(model, trainloader, validloader, criterion, optimizer, device='cuda', epochs=1, print_every=5):
    model.to(device)
    model.train()

    steps = 0
    running_loss = 0

    for e in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_metrics = [
                        (criterion(model(inputs.to(device)), labels.to(device)).item(),
                         (torch.exp(model(inputs.to(device))).argmax(dim=1) == labels.to(device)).float().mean().item())
                        for inputs, labels in validloader
                    ]

                avg_train_loss = running_loss / print_every
                avg_valid_loss, avg_accuracy = map(lambda x: sum(x) / len(validloader), zip(*valid_metrics))

                print(f"Epoch {e + 1}/{epochs}.. "
                      f"Loss: {avg_train_loss:.3f}.. "
                      f"Validation Loss: {avg_valid_loss:.3f}.. "
                      f"Accuracy: {avg_accuracy:.3f}")

                running_loss = 0
                model.train()
#=================================================================#
#fourth fun
#define fun for loadcheckpoint
def load_checkpoint(filepath):
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    model, criterion, optimizer = nn_setup()
    Mstructure = checkpoint['Mstructure']


    # Load the saved model state_dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the class to index mapping
    model.class_to_idx = checkpoint['class_to_idx']

    return model
#===========================================================#
   
    #==================================================#
#sisth fun

def evaluate_model(model, testloader, criterion, device='cuda'):
    test_loss = 0
    correct_predictions = 0
    total_samples = 0

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model(inputs)
            batch_loss = criterion(log_ps, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(log_ps)
            _, predicted = ps.max(1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_accuracy = correct_predictions / total_samples
    test_loss /= len(testloader)

    print(f"Test accuracy: {test_accuracy:.3f}")
    print(f"Test loss: {test_loss:.3f}")
#=========================================================================================================================#


