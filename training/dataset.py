
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def compute_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std

def initialise_dataset(batch_size, ds="cifar"):

    if ds == "cifar":
        transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        return trainset,trainloader,testset,testloader,classes


    if ds == "cifar-aug":
        transform_train = transforms.Compose([
            transforms.RandomRotation(15),  # randomly rotate images in the range (degrees, 0 to 180)
            transforms.RandomHorizontalFlip(),  # randomly flip images horizontally
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # randomly crop and resize images
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # randomly change the brightness, contrast, saturation and hue
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        return trainset, trainloader, testset, testloader, classes
    
    if ds == "cifar-aug-std":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

        # Compute mean and std
        mean, std = compute_mean_and_std(trainset)
        transform_train = transforms.Compose([
            transforms.RandomRotation(15),  # randomly rotate images in the range (degrees, 0 to 180)
            transforms.RandomHorizontalFlip(),  # randomly flip images horizontally
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # randomly crop and resize images
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # randomly change the brightness, contrast, saturation and hue
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        return trainset, trainloader, testset, testloader, classes
        
    if ds == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        
        classes = tuple(str(i) for i in range(10))
        
        return trainset, trainloader, testset, testloader, classes
        
    if ds == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                     download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        
        classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
        
        return trainset, trainloader, testset, testloader, classes

    if ds == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        trainset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
        # EMNIST 'byclass' split contains 62 classes: 10 digits and 52 letters
        classes = (
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
            'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z'
        )
        
        return trainset, trainloader, testset, testloader, classes
    if ds == "usps":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        trainset = torchvision.datasets.USPS(root='./data', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.USPS(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
        # USPS contains 10 classes, similar to MNIST
        classes = tuple(str(i) for i in range(10))
        
        return trainset, trainloader, testset, testloader, classes

    if ds == "stl10":
        transform = transforms.Compose([
            transforms.Resize((96, 96)),  # Ensure images are resized to 96x96
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB channels
        ])
        
        trainset = torchvision.datasets.STL10(root='./data', split='train',
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.STL10(root='./data', split='test',
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
        # STL-10 contains 10 classes
        classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
        
        return trainset, trainloader, testset, testloader, classes
    if ds == "imagenette":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize using ImageNet standards
        ])
        
        trainset = torchvision.datasets.Imagenette(root='./data', split='train',
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.Imagenette(root='./data', split='val',
                                                download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
        # Imagenette contains 10 classes
        classes = ('tench', 'English_springer', 'cassette_player', 'chain_saw', 
                'church', 'French_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute')
        
        return trainset, trainloader, testset, testloader, classes
    if ds == "tinyimagenet":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        
        trainset = ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        testset = ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Tiny ImageNet has 200 classes
        classes = [f'class_{i}' for i in range(200)]
        
        return trainset, trainloader, testset, testloader, classes


def test_trial_batch_print(trainloader, batch_size,classes):
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))