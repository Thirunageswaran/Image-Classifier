import argparse
import json
from PIL import Image
import torch
import numpy as np
from torchvision import models


def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network Prediction script")

    parser.add_argument('image', type = str, help="Provide the image file.")
    
    parser.add_argument('checkpoint', type=str, help='Provide the checkpoint path files')

    parser.add_argument('--top_k', type = int, help = "Provide the top k most likely matches.")

    parser.add_argument('--gpu', action='store_true', help='Chooes either gpu or cpu to run the network.')

    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True);
    elif checkpoint['arch'] == "vgg13":
        model = models.vgg13(pretrained=True);
    elif checkpoint['arch'] == "alexnet":
        model = models.alexnet(pretrained=True);
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict=(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def image_processing(image):
    print("Converting the image into tensors")
    test_image = Image.open(image)

    width, height = test_image.size
    aspect_ratio = width / height
    if aspect_ratio > 1:
        test_image = test_image.resize((round(aspect_ratio * 256), 256))
    else:
        test_image = test_image.resize((256, round(256 / aspect_ratio)))

    new_widht, new_height = test_image.size

    left, top, right, bottom = round((new_widht-244)/2), round((new_height-244)/2), round((new_widht+244)/2), round((new_height+244)/2)
    test_image = test_image.crop((left, top, right, bottom))

    np_image = np.array(test_image) / 255

    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image, model, device, cat_to_name, top_k):
    model.to(device)

    model.eval();

    image = torch.from_numpy(np.expand_dims(image, axis=0)).type(torch.FloatTensor).to(device)

    probs, labels = torch.exp(model(image)).topk(top_k)

    probs = np.array(probs.detach())[0]
    labels = np.array(labels.detach())[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    labels = [idx_to_class[lab] for lab in labels]
    flowers = [cat_to_name[lab] for lab in labels]

    return probs, labels, flowers

def main():
    args = arg_parser()

    if not args.gpu:
        device = torch.device("cpu")
        print("Device is using cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device is using : {}".format(device))

    if args.top_k:
        top_k = args.top_k
    else:
        top_k = 5
    
    print("The top k is : {}".format(top_k))
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    print("Loading checkpoint form {}".format(args.checkpoint))
    
    model = load_checkpoint(args.checkpoint)

    image = image_processing(args.image)
    
    print("Predicting the image")

    probs, labels, flowers = predict(image, model, device, cat_to_name, top_k)

    for i in range(top_k):
        print("Rank : {}".format(i+1,), "Flower name: {}".format(flowers[i]), "Probability: {}".format(probs[i]*100))
    
    print("Done")

if __name__=='__main__': main()
