import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class FixedDropout(nn.Module):
    def __init__(self, layer, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.weight_shape = layer.weight.data.shape
        self.bias_shape = layer.bias.data.shape

    def forward(self, p):
        torch.manual_seed(self.seed)
        return torch.bernoulli(torch.ones(self.weight_shape) * (1 - p())),\
               torch.bernoulli(torch.ones(self.bias_shape) * (1 - p()))

class MyMLP(nn.Module):
    def __init__(self, set_bandwidth=False, hidden_units=256):
        super().__init__()
        self.set_bandwidth = set_bandwidth
        l1 = nn.Linear(1, hidden_units)
        l2 = nn.Linear(hidden_units, hidden_units)
        self.net = nn.Sequential(
            FixedDropout(l1) if set_bandwidth else nn.Identity(),
            l1,
            nn.ReLU(),
            FixedDropout(l2) if set_bandwidth else nn.Identity(),
            l2,
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x, bandwidth, test=False):
        for i, layer in enumerate(self.net[:-1]):
            if self.set_bandwidth and isinstance(layer, FixedDropout):
                with torch.no_grad():
                    dropout_w, dropout_b = self.net[i](bandwidth)
                    old_w = self.net[i + 1].weight.data.detach().clone()
                    old_b = self.net[i + 1].bias.data.detach().clone()
                    self.net[i + 1].weight.data *= dropout_w
                    self.net[i + 1].bias.data *= dropout_b
                x = self.net[i + 1](x)
                with torch.no_grad():
                    self.net[i + 1].weight.data += torch.logical_not(dropout_w) * old_w
                    self.net[i + 1].bias.data += torch.logical_not(dropout_b) * old_b
            elif self.set_bandwidth and isinstance(layer, nn.Linear):
                continue
            else:
                x = layer(x)
        return self.net[-1](x)


def visualize_normal(samples, splits=9):
    min_val = samples.min()
    max_val = samples.max()
    dsplit = (max_val - min_val) / splits
    for split in range(splits):
        split_samples = ((min_val + dsplit * split) <= samples) * ((min_val + dsplit * (split + 1)) >= samples)
        n_samples = split_samples.sum()
        plt.bar(min_val + split * dsplit + dsplit / 2, n_samples, dsplit)
    plt.show()


random_samples = np.random.uniform(0, 2 * np.pi, int(40 * np.pi))
eq_samples = np.arange(0, 2 * np.pi, 0.01)  # equidistant samples
eq_values = np.sin(eq_samples)
tensor_eq_samples = torch.from_numpy(eq_samples).float().view(eq_samples.shape[0], 1)
tensor_eq_values = torch.from_numpy(eq_values).float()

if "-lm" not in sys.argv:
    TRAINIG_STEPS = 80000
    BATCH_SIZE = 128
    LR = 0.000005
    step = 0

    samples = random_samples if "-r" in sys.argv else eq_samples
    bandwidth = "-b" in sys.argv

    units = 256
    if "-u" in sys.argv:
        str_units = sys.argv[sys.argv.index("-u") + 1]
        assert str_units.isnumeric()
        units = int(str_units)

    mode = "default"
    if "-sl" in sys.argv:
        BANDWIDTH_SCHEDULER = lambda: max(0.9 - (step / TRAINIG_STEPS + 0.05), 0)
        mode = "sl"
    elif "-su" in sys.argv:
        BANDWIDTH_SCHEDULER = lambda: np.random.uniform(0.0, 0.9)
        mode = "su"
    elif "-sn" in sys.argv:
        BANDWIDTH_SCHEDULER = lambda: np.clip(np.random.normal(0.45, 0.2), 0.0, 0.9)
        mode = "sn"
        # visualize_normal(np.clip(np.random.normal(0.45, 0.2, 10000), 0.0, 0.9))
    else:
        BANDWIDTH_SCHEDULER = lambda: max(0.9 - (step / TRAINIG_STEPS + 0.05), 0)

    path = mode + "_" + str(units) + ".pth"
    sin_values = np.sin(samples)

    net = MyMLP(set_bandwidth=bandwidth, hidden_units=units)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    criterion_loss = torch.nn.MSELoss()
    samples = torch.from_numpy(samples).float()
    min_loss = 100.0
    min_loss_idx = 0
    losses = []

    for i in range(TRAINIG_STEPS):
        step = i
        batch_idxs = torch.randint(0, len(samples), (BATCH_SIZE, 1))
        batch = samples[batch_idxs]
        values = torch.from_numpy(sin_values[batch_idxs]).float()
        predictions = net(batch, BANDWIDTH_SCHEDULER)
        loss = criterion_loss(predictions, values)
        loss.backward()
        opt.step()
        print("Loss (", i, "): ", loss.item())

        # test_predictions = net(tensor_eq_samples, BANDWIDTH_SCHEDULER)
        # test_loss = criterion_loss(test_predictions, tensor_eq_values)
        losses.append(loss.detach().item())
        if loss.detach().item() < float(min_loss):
            min_loss = loss
            min_loss_idx = i
            torch.save(net.state_dict(), path)

    torch.save(net.state_dict(), "last_" + path)
    plt.plot(np.arange(0, len(losses)), losses)
    print("Min loss at idx: ", min_loss_idx, ", value of: ", min_loss.item())
    plt.title("Mode " + mode + str(units) + "_loss")
    plt.savefig(mode + "_" + str(units) + "_loss.png")
    plt.close()
    # plt.show()
else:
    path = sys.argv[sys.argv.index('-lm') + 1]


mode = path.split("_")[0]
units = path.split("_")[1].split(".")[0]
net = MyMLP(set_bandwidth=True, hidden_units=int(units))
net.load_state_dict(torch.load(path))

with torch.no_grad():
    for prob in [0.9, 0.7, 0.5, 0.3, 0.0]:
        predictions = net(tensor_eq_samples, lambda: prob)
        plt.plot(eq_samples, predictions.detach().numpy(), color="blue", alpha=0.5)
        plt.plot(eq_samples, np.sin(eq_samples), color="red", alpha=0.5)
        plt.title("Mode " + mode + str(units) + ", dorpout prob " + str(prob))
        plt.savefig(mode + "_" + str(units) + "_" + str(prob) + ".png")
        plt.close()
        # plt.show()
