# AI programming with python 🧐.
**This is a graduation project for (udacity program) AI programming with python🎓.**
## project is consist form Two parts:

**🎯part one:Development Notebook**

**🎯part two: Command Line Application**

**overview**:
this a python program used neueral network to classify images using python conepts to apply it , it's a great and useful program.

**🔎Package Imports**

All the necessary packages and modules are imported in the first cell of the notebook


**🔎Training data augmentation**

torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping

**🔎Data normalization**
The training, validation, and testing data is appropriately cropped and normalized

**🛢Data loading**
The data for each set (train, validation, test) is loaded with torchvision's ImageFolder

**🛢Data batching**
The data for each set is loaded with torchvision's DataLoader

**🛢Pretrained Network**

A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen

**⏳Feedforward Classifier**

A new feedforward network is defined for use as a classifier using the features as input

**⏳Training the network**

The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static

**⏳Validation Loss**

and Accuracy	During training, the validation loss and accuracy are displayed

**💣Testing Accuracy**

The network's accuracy is measured on the test data

**💣Saving the model**
The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary

**💣Loading checkpoints:**

There is a function that successfully loads a checkpoint and rebuilds the model

**📌Image Processing:**

The process_image function successfully converts a PIL image into an object that can be used as input to a trained model

**📌Class Prediction**

The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image

**📌Sanity Checking :**

with matplotlib	A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

