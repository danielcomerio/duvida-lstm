from torchvision import models
import torch.nn as nn


def initialize_pretrained_model(model_name: str, num_classes: int, use_pretrained: bool=True) -> None: # typerror # tuple[Any, int]
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    feature_size = 0

    if model_name == "resnext50":
        """ Resnext50
        """
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnext101":
        """ Resnext50
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        model_ft.classifier[2] = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        feature_size = 1024

    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        feature_size = 1664

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size, feature_size

