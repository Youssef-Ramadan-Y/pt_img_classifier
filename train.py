import argparse
# Imports here
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler

from collections import OrderedDict

import json
import pandas as pd
from torch import nn, optim

from tqdm import tqdm

from PIL import Image
from model_trainer import DataLoader, Model, Classifier, Trainer

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('--train_dir', type=str, default="flowers/train", help='Directory of the train dataset')
    parser.add_argument('--test_dir', type=str, default="flowers/test", help='Directory of the test dataset')
    parser.add_argument('--valid_dir', type=str, default="flowers/valid", help='Directory of the valid dataset')
    parser.add_argument('--model', type=str, choices=['vgg', 'resnet'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', default=1024, type=int, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    args = parser.parse_args()

    # Define transformations for training data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

    # Create a DataLoader instance
    
    train_loader = DataLoader(args.train_dir, train_transforms).get_loader()
    valid_loader = DataLoader(args.valid_dir, validation_transforms).get_loader()
    test_loader = DataLoader(args.test_dir, test_transforms).get_loader()

    # Create a Model instance
    model = Model(args.model)
    model = model.get_model()

    # Create a Classifier instance
    classifier = Classifier(model.classifier[0].in_features,
                            args.hidden_units, 256, len(train_loader.dataset.classes), 0.5)

    
    model.classifier = classifier.get_classifier()

    # Set up criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Create a Trainer instance
    trainer = Trainer(train_loader, valid_loader, test_loader, model, classifier)
    
    # Train the model
    trainer.train_model(criterion, optimizer, args.epochs)

    # Save the checkpoint
    save_path = f'{args.save_dir}/{args.arch}_checkpoint.pth'
    trainer.save_checkpoint(save_path)
    print(f'Model checkpoint saved to {save_path}')

    if args.gpu:
        print('Training complete on GPU!')
    else:
        print('Training complete on CPU.')

if __name__ == '__main__':
    main()
