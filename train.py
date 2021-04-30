import argparse
import torch
import json
from torch import nn
from torch import optim
from os.path import isdir
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Neural Network training script")

    parser.add_argument('data_dir', type=str, help='Provide the data directory to train the network.')

    parser.add_argument('--arch', type = str, help='Choose an architecture from torchvision.models either vgg16 or alexnet or vgg13', default='vgg16')

    parser.add_argument('--epochs', type=int, help='Number of training epochs.')

    parser.add_argument('--hidden_units', type=int, help='Number of hidden units for classifier.')

    parser.add_argument('--learning_rate', type=float, help='Gradient descent learning rate.')

    parser.add_argument('--gpu', action='store_true', help='Chooes either gpu or cpu to run the network.')

    parser.add_argument('--save_dir', type=str, help='Directory to save the checkpoint.')


    return parser.parse_args()

def train_transform(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def test_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

def valid_transform(valid_dir):
    data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)
    return valid_data

def data_loader(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=64)
    return loader

def check_gpu(gpu):
    if not gpu:
        device = torch.device("cpu")
        print("Device is using cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device is using : {}".format(device))
    return device

def load_pretrained_model(architecture):
    if architecture == "vgg13":
        model = models.vgg13(pretrained = True)
        model.name = "vgg13"
    elif architecture == "alexnet":
        model = models.alexnet(pretrained = True)
        model.name = "alexnet"
    elif architecture == "vgg16":
        model = models.vgg16(pretrained = True)
        model.name = "vgg16"

    for param in model.parameters():
        param.requires_grad = False
    return model

def classifier(model, hidden_units, arch):
    if type(hidden_units) == type(None):
        hidden_units = 4096

    print("Number of Hidden layers : {}".format(hidden_units))

    if arch == "vgg16" or arch == "vgg13":
        model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    elif arch == "alexnet":
        model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, hidden_units, bias=True)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(hidden_units, 102, bias = True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return model

def train_network(model, trainloader, validloader, device, criterion, optimizer, epochs):
    print_every = 40
    running_loss = 0
    steps = 0
    for epoch in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    model.to(device)
                    test_loss = 0
                    accuracy = 0
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()
    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Valid loss: {test_loss/len(validloader):.3f}.. "
                        f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model

def validate_newtork(model, testloader, device):
    correct=0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

def checkpoint(model, train_data, save_dir, arch):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'arch': arch,
                  'model_state_dict': model.state_dict()}

    if type(save_dir) != type(None) and isdir(save_dir):
        torch.save(checkpoint, save_dir + '/checkpoint.pth')
        print("Checkpoint path is : {}/checkpoint.pth.".format(save_dir))
    else:
        torch.save(checkpoint, 'checkpoint.pth')
        print("Checkpoint path is : checkpoint.pth")

def main():
    args = arg_parser()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_data = test_transform(train_dir)
    valid_data = train_transform(valid_dir)
    test_data = train_transform(test_dir)

    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    device = check_gpu(gpu=args.gpu)
 
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_pretrained_model(architecture=args.arch)
    print("The model name is : {}".format(args.arch))
    
    model = classifier(model, hidden_units=args.hidden_units, arch = args.arch)
    
    if type(args.learning_rate) == type(None):
        lr = 0.001
    else:
        lr = args.learning_rate
    print("learning rate : {}.".format(lr))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    model.to(device)

    if type(args.epochs) == type(None):
        epochs = 6
    else:
        epochs = args.epochs
    print("Number of epochs : {}.".format(epochs))

    print("Training process initializing...\n")
    
    trained_model = train_network(model, trainloader, validloader, device, criterion, optimizer, epochs)
    
    print("Training is complete.")
    
    validate_newtork(trained_model, testloader, device)

    checkpoint(trained_model, train_data, args.save_dir, args.arch)


if __name__=='__main__': main()
