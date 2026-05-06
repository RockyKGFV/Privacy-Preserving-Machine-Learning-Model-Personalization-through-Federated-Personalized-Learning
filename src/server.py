import flwr as fl
import json
import os
import sys

EXPERIMENT = "globallogic"
results = []
client_accs = []

# def weighted_average(metrics):
#     acc = sum([m["accuracy"] * n for n, m in metrics]) / sum([n for n, _ in metrics])
#     results.append(acc)
#     return {"accuracy": acc}

# def weighted_average(metrics):
#     acc = sum([m["accuracy"] * n for n, m in metrics]) / sum([n for n, _ in metrics])
    
#     # store per-client
#     for _, m in metrics:
#         client_accs.append(m["client_acc"])
    
#     results.append(acc)
#     return {"accuracy": acc}

def weighted_average(metrics):
    acc = sum([m["accuracy"] * n for n, m in metrics]) / sum([n for n, _ in metrics])
    
    # store global accuracy
    results.append(acc)

    # store client accuracies if available
    for _, m in metrics:
        if "client_acc" in m:
            client_accs.append(m["client_acc"])

    return {"accuracy": acc}

def main():
    os.makedirs("results", exist_ok=True)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

    os.makedirs("results", exist_ok=True)

    with open(f"results/cifar_{EXPERIMENT}.json", "w") as f:
        json.dump(results, f)

    if len(client_accs) > 0:
        with open(f"results/cifar_{EXPERIMENT}_clients.json", "w") as f:
            json.dump(client_accs, f)

if __name__ == "__main__":
    main()