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

import os
import random

class DataLoader:
    def __init__(self, data_dir, transforms):
        self.data = datasets.ImageFolder(data_dir, transform=transforms)
        self.loader = torch.utils.data.DataLoader(self.data, batch_size=32, shuffle=True)

    def get_loader(self):
        return self.loader

class Model:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        if self.model_name == "vgg":
            model = models.vgg16(pretrained=True)
        elif self.model_name == "resnet":
            model = models.resnet50(pretrained=True)
        # Add more pretrained models as needed

        for param in model.parameters():
          param.requires_grad = False
          
        return model

class Classifier:
    def __init__(self, in_features, hidden_layer_1, hidden_layer_2, out_features, dropout):
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc1', nn.Linear(in_features, hidden_layer_1)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layer_1, hidden_layer_2)),
            ('output', nn.Linear(hidden_layer_2, out_features)),
            ('softmax', nn.LogSoftmax(dim=1))
        ]))

    def get_classifier(self):
        return self.classifier

class ImageProcessor:
    @staticmethod
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns a PyTorch tensor
        '''
        img_pil = Image.open(image)

        predict_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        img_tensor = predict_transforms(img_pil)

        return img_tensor

class Trainer:
    def __init__(self, trainloader, validloader, testloader, model, classifier):
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.model = model
        self.classifier = classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, criterion, optimizer, epochs=5):
        steps = 0
        print_every = 40
        self.model.to(self.device)
        pbar = tqdm(total=epochs)
        for e in range(epochs):
            running_loss = 0
            for images, labels in self.trainloader:
                steps += 1
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    self.model.eval()
                    val_loss = 0
                    accuracy = 0

                    with torch.no_grad():
                        for val_images, val_labels in self.validloader:
                            val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                            val_outputs = self.model.forward(val_images)
                            val_loss += criterion(val_outputs, val_labels)

                            ps = torch.exp(val_outputs)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == val_labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))

                    pbar.set_description(f"Epoch {e + 1}/{epochs}, Training Loss: {running_loss / print_every:.3f}, Validation Loss: {val_loss / len(self.validloader):.3f}, Validation Accuracy: {accuracy / len(self.validloader):.3f}")
                    pbar.update(1)

                    running_loss = 0
                    self.model.train()
            if ((accuracy / len(self.validloader) > .97)):
              break

    def test_model(self):
        correct = 0
        total = 0
        self.model.to(self.device)

        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

    def save_checkpoint(self, filepath):
        self.model.class_to_idx = self.trainloader.dataset.class_to_idx
        checkpoint = {'class_to_idx': self.model.class_to_idx,
                      'state_dict': self.model.state_dict()}

        torch.save(checkpoint, filepath)

    @staticmethod
    def load_checkpoint(filepath, model):
        checkpoint = torch.load(filepath)

        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

        return model
        
    def predict(self, image_path, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        self.model.to('cpu')
        self.model.eval()

        img_torch = ImageProcessor.process_image(image_path)
        img_torch = img_torch.unsqueeze_(0)
        img_torch = img_torch.float()

        with torch.no_grad():
            output = self.model.forward(img_torch)

            probs = torch.exp(output)
            top_p, top_class = probs.topk(topk, dim=1)

            top_p = top_p.numpy()[0]
            top_class = top_class.numpy()[0]

        idx_to_class = {val: key for key, val in
                                          self.model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class]

        return top_p, top_class



    
    
