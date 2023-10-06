import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


HELP_MESSAGE="""\
Usage: python sin_mlp.py [options]
    -h\t\t(bool) optional, show this help message
    -r\t\t(bool) optional, train with random uniform sampled data
    -sl\t\t(bool) optional, train with bandwidth scheduling sampled linearly
    -su\t\t(bool) optional, train with bandwidth scheduling sampled uniformly
    -sn\t\t(bool) optional, train with bandwidth scheduling sampled with clipped normal distribution
    -u\t\t(int) optional, number of hidden units of a single hidden layer MLP
    -lm\t\t(string) optional, load a model from given path and test it
    -lv\t\t(bool) optional, show plots live
    -bm\t\t(enum) optional, bandwidth mode: either dropout (d) or weight dropout (wd)
"""


class WeightDropLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, device=None, dtype=None, seed=0):
        super().__init__(in_f, out_f, bias, device, dtype)
        self.seed = seed
        self.weight_shape = self.weight.data.shape
        self.bias_shape = self.bias.data.shape

    def forward(self, x, p):
        torch.manual_seed(self.seed)
        old_weight = self.weight.data.detach().clone()
        old_bias = self.bias.data.detach().clone()
        with torch.no_grad():
            drop_weight = torch.bernoulli(torch.full(self.weight_shape, float(1 - p())))
            drop_bias = torch.bernoulli(torch.full(self.bias_shape, float(1 - p())))
        self.weight.data *= drop_weight
        self.bias.data *= drop_bias
        prediction = super().forward(x)
        with torch.no_grad():
            self.weight.data += torch.logical_not(drop_weight) * old_weight
            self.bias.data += torch.logical_not(drop_bias) * old_bias
        return prediction


class FixedDropout(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    def forward(self, x, p):
        torch.manual_seed(self.seed)
        return F.dropout(x, p(), True)


class MyMLP(nn.Module):
    def __init__(self, bandwidth_mode=None, hidden_units=256, bandwidth=0.5):
        super().__init__()
        self.weight_drop = bandwidth_mode == "wd"
        self.dropout = bandwidth_mode == "d"
        self.net = nn.Sequential(
            WeightDropLinear(1, hidden_units)
            if self.weight_drop
            else nn.Linear(1, hidden_units),
            nn.Dropout(bandwidth) if self.dropout else nn.Identity(),
            nn.ReLU(),
            WeightDropLinear(hidden_units, hidden_units)
            if self.weight_drop
            else nn.Linear(hidden_units, hidden_units),
            nn.Dropout(bandwidth) if self.dropout else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x, bandwidth, test=False):
        for i, layer in enumerate(self.net[:-1]):
            if self.weight_drop and isinstance(layer, WeightDropLinear):
                x = layer(x, bandwidth)
            elif self.weight_drop and isinstance(layer, nn.Linear):
                continue
            elif self.dropout and isinstance(layer, FixedDropout):
                x = layer(x, bandwidth)
            else:
                x = layer(x)
        return self.net[-1](x)


def visualize_normal(samples, splits=9):
    min_val = samples.min()
    max_val = samples.max()
    dsplit = (max_val - min_val) / splits
    for split in range(splits):
        split_samples = ((min_val + dsplit * split) <= samples) * (
            (min_val + dsplit * (split + 1)) >= samples
        )
        n_samples = split_samples.sum()
        plt.bar(min_val + split * dsplit + dsplit / 2, n_samples, dsplit)
    plt.show()


if "-h" in sys.argv:
    print(HELP_MESSAGE[:-1])
    exit()

live = "-lv" in sys.argv

random_samples = np.random.uniform(0, 2 * np.pi, int(40 * np.pi))
eq_samples = np.arange(0, 2 * np.pi, 0.01)  # equidistant samples
eq_values = np.sin(eq_samples)
tensor_eq_samples = torch.from_numpy(eq_samples).float().view(eq_samples.shape[0], 1)
tensor_eq_values = torch.from_numpy(eq_values).float()

bandwidth_mode = ""
if "-bm" in sys.argv:
    bandwidth_mode = sys.argv[sys.argv.index("-bm") + 1]

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
    mode += "_" + bandwidth_mode

    path = "models/" + mode + "_" + str(units) + ".pth"
    sin_values = np.sin(samples)

    net = MyMLP(bandwidth_mode=bandwidth_mode, hidden_units=units)
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

    split_idx = path.index("/")
    torch.save(
        net.state_dict(), path[: split_idx + 1] + "last_" + path[split_idx + 1 :]
    )
    plt.plot(np.arange(0, len(losses)), losses)
    print("Min loss at idx: ", min_loss_idx, ", value of: ", min_loss.item())
    plt.title("Mode " + mode + str(units) + "_loss")
    if not live:
        plt.savefig("plots/" + mode + "_" + str(units) + "_loss.png")
        plt.close()
    else:
        plt.show()
else:
    path = sys.argv[sys.argv.index("-lm") + 1]

if bandwidth_mode == "":
    bandwidth_mode = "d"

if path[path.index("/") + 1 :].split("_")[0] == "default":
    mode = "default_" + bandwidth_mode
else:
    mode = "_".join(path[path.index("/") + 1 :].split("_")[:2])
units = path.split("_")[2].split(".")[0]

with torch.no_grad():
    for prob in [0.9, 0.7, 0.5, 0.3, 0.0]:
        net = MyMLP(
            bandwidth_mode=bandwidth_mode, hidden_units=int(units), bandwidth=prob
        )
        net.load_state_dict(torch.load(path))
        predictions = net(tensor_eq_samples, lambda: prob)
        plt.plot(eq_samples, predictions.detach().numpy(), color="blue", alpha=0.5)
        plt.plot(eq_samples, np.sin(eq_samples), color="red", alpha=0.5)
        plt.title("Mode " + mode + str(units) + ", dorpout prob " + str(prob))
        if not live:
            plt.savefig("plots/" + mode + "_" + str(units) + "_" + str(prob) + ".png")
            plt.close()
        else:
            plt.show()
