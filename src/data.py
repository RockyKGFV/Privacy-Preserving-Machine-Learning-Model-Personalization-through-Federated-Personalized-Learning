import numpy as np
from torchvision import datasets, transforms

# def load_data():
#     transform = transforms.Compose([transforms.ToTensor()])
    
#     train = datasets.MNIST("./data", train=True, download=True, transform=transform)
#     test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
#     return train, test

from torchvision import datasets, transforms

def load_data(dataset="mnist"):
    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test = datasets.MNIST("./data", train=False, download=True, transform=transform)

    elif dataset == "cifar":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    return train, test

def dirichlet_split(dataset, num_clients, alpha):
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx_by_class[c])).astype(int)[:-1]
        splits = np.split(idx_by_class[c], proportions)

        for i in range(num_clients):
            client_indices[i] += splits[i].tolist()

    return client_indices