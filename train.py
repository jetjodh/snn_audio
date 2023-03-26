from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import pandas as pd
import snntorch as snn
import numpy as np
from LIF import lif_coding
from cochleagram import cochleagram
import librosa

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_steps = 25
dtype = torch.float

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, fold, train=True):
        self.file_path = file_path
        metadata = pd.read_csv('metadata/UrbanSound8K.csv')
        if train:
            self.metadata = metadata.drop(metadata[metadata.fold != fold].index)
        else:
            self.metadata = metadata.drop(metadata[metadata.fold == fold].index)
        self.audio = metadata['slice_file_name']
        self.labels = metadata['classID']
        self.train = train
        self.fold = metadata['fold']

    def __len__(self):
            'Denotes the total number of samples'
            return len(self.metadata)

    def __getitem__(self, index):
            'Generates one sample of data'
            self.cochleagram = cochleagram(librosa.load("audio/"+f"fold{self.fold[index]}/"+self.audio[index], sr=16000)[0],
            n=8,
            sr=1000,
            low_lim=10,
            hi_lim=20000,
            sample_factor=2,
            padding_size=None,
            downsample=10000,
            nonlinearity=None,
            fft_mode="auto",
            ret_mode="envs",
            strict=True,
        )
            self.lif_coding = lif_coding(self.cochleagram, 0.001, 0.1)
            return np.array(self.lif_coding), self.labels[index]

train = SimpleDataset('metadata/UrbanSound8K.csv', 1)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

valid = SimpleDataset('metadata/UrbanSound8K.csv', 1, train=False)
valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)

def print_batch_accuracy(model, data, targets, train=False):
    output, _ = model(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    model, data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(model, data, targets, train=True)
    print_batch_accuracy(model, test_data, test_targets, train=False)
    print("\n")


def train_model(model, criterion, optimizer, num_epochs=25):
    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0

    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            model.train()
            spk_rec, mem_rec = model(data.view(batch_size, -1))

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += criterion(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                model.eval()
                test_data, test_targets = next(iter(valid_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = model(test_data.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += criterion(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer(
                        model, data, targets, epoch,
                        counter, iter_counter,
                        loss_hist, test_loss_hist,
                        test_data, test_targets)
                counter += 1
                iter_counter +=1
    return loss_hist, test_loss_hist, model

class Net(nn.Module):
    """
    defining the spiking neural network
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(21, 100)
        self.lif1 = snn.Leaky(beta=0.99)
        self.fc2 = nn.Linear(100, 10)
        self.lif2 = snn.Leaky(beta=0.99, output=True)
    
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []
        for step in range(25):
            cur1 = self.fc1(x.float())
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_hist, test_loss_hist, model = train_model(model, criterion, optimizer, num_epochs=1)

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()