import argparse
import torch
from torchvision import models
from PIL import Image
import json
from model_trainer import Trainer

# Imports here
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models




def main():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained deep learning model')
    parser.add_argument('--image_path', type=str, help='Path of the image')
    parser.add_argument('--checkpoint', type=str, help='Path of the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Print out the top K classes along with associated probabilities')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file that maps the class values to other category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction if available')

    args = parser.parse_args()

    # Load the checkpoint
    model = models.vgg16(pretrained=False)
    classifier = Trainer.load_checkpoint(args.checkpoint, model)

    # Create a Trainer instance for prediction
    trainer = Trainer(None, None, None, model, classifier)

    # Predict the class for the input image
    probs, classes = trainer.predict(args.image_path, args.top_k)

    # Load category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(c)] for c in classes]
    else:
        class_names = classes

    # Print the results
    for i in range(args.top_k):
        print(f'Top {i + 1}: {class_names[i]} - Probability: {probs[i]:.4f}')

    if args.gpu:
        print('Prediction complete on GPU!')
    else:
        print('Prediction complete on CPU.')

if __name__ == '__main__':
    main()

# how to run
# python predict.py --image_path /flowers/test/1/image_06754.jpg --checkpoint checkpoint --top_k 5 --category_names category_names.json --gpu