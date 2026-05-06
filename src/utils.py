import torch

def test(model, testloader, device):
    model.eval()
    correct, total, loss = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return loss / len(testloader), correct / total