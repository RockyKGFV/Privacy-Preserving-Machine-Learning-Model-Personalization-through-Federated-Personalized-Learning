import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset

from model import Net, get_shared_params, set_shared_params, get_params
from utils import test
from data import load_data, dirichlet_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from model import Net, ResNetPersonalized



# PERSONALIZED = False  
PERSONALIZED = True  

# DATASET_ALPHA = 1.0  
# DATASET_ALPHA = 0.5  
# DATASET_ALPHA = 0.1  

DATASET = "cifar" 


class Client(fl.client.NumPyClient):
    def __init__(self, train_data, test_data):
        if DATASET == "mnist":
            self.model = Net().to(DEVICE)
        else:
            self.model = ResNetPersonalized().to(DEVICE)
        self.trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
        self.testloader = DataLoader(test_data, batch_size=64)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    # def get_parameters(self, config):
    #     return get_shared_params(self.model)

    def get_parameters(self, config):
        return get_params(self.model, PERSONALIZED)

    # def set_parameters(self, parameters):
    #     set_shared_params(self.model, parameters)

    def set_parameters(self, parameters):
        if PERSONALIZED:
            set_shared_params(self.model, parameters)
        else:
            for p, new_p in zip(self.model.parameters(), parameters):
                p.data = torch.tensor(new_p)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        for _ in range(5):
            for x, y in self.trainloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)
    #     loss, acc = test(self.model, self.testloader, DEVICE)
    #     return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}
    
    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)

    #     loss, acc = test(self.model, self.testloader, DEVICE)

    #     print(f"Client accuracy: {acc}")  # <-- IMPORTANT

    #     return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, acc = test(self.model, self.testloader, DEVICE)

        return float(loss), len(self.testloader.dataset), {
            "accuracy": float(acc),
            "client_acc": float(acc)
        }

# ---------- CLIENT START ----------

def main():
    train, test_data = load_data(DATASET)

    client_id = int(input("Enter client ID (0-9): "))
    # splits = dirichlet_split(train, num_clients=10, alpha=0.5)
    splits = dirichlet_split(train, num_clients=10, alpha=0.1)

    train_data = Subset(train, splits[client_id])

    client = Client(train_data, test_data)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
    )

if __name__ == "__main__":
    main()