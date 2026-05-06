# import json
# import matplotlib.pyplot as plt

# # Load all files
# with open("results/mnist_fedavg.json") as f:
#     fedavg = json.load(f)

# with open("results/mnist_personalized.json") as f:
#     personalized = json.load(f)

# # (Optional: your earlier run)
# try:
#     with open("results/mnist_acc.json") as f:
#         old = json.load(f)
# except:
#     old = None

# # Plot
# plt.figure(figsize=(8, 5))

# plt.plot(fedavg, label="FedAvg", linewidth=2)
# plt.plot(personalized, label="Personalized", linewidth=2)

# if old:
#     plt.plot(old, linestyle="--", label="Old Run")

# plt.xlabel("Communication Rounds")
# plt.ylabel("Accuracy")
# plt.title("FedAvg vs Personalized (MNIST)")
# plt.legend()
# plt.grid()

# plt.savefig("results/comparison.png")
# plt.show()

import json
import matplotlib.pyplot as plt

# with open("results/mnist_iid.json") as f:
#     iid = json.load(f)

# with open("results/mnist_noniid.json") as f:
#     noniid = json.load(f)

# plt.plot(iid, label="IID (alpha=1.0)", linewidth=2)
# plt.plot(noniid, label="Non-IID (alpha=0.1)", linewidth=2)

# plt.xlabel("Rounds")
# plt.ylabel("Accuracy")
# plt.title("IID vs Non-IID (Personalized)")
# plt.legend()
# plt.grid()

# plt.savefig("results/iid_vs_noniid.png")
# plt.show()

with open("results/global.json") as f:
    global_acc = json.load(f)

with open("results/local.json") as f:
    local_acc = json.load(f)

plt.figure(figsize=(8,5))

plt.plot(global_acc, label="Global Accuracy", linewidth=2)
plt.axhline(sum(local_acc)/len(local_acc), linestyle="--", label="Avg Local Accuracy")

plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Global vs Local Tradeoff")
plt.legend()
plt.grid()

plt.savefig("results/global_vs_local.png")
plt.show()