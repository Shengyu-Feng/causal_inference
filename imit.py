import torch
import torch.nn.functional as F
import numpy as np
import heapq
import matplotlib.pyplot as plt
from betaVAE import VAE
from a2c_ppo_acktr.envs import make_vec_envs
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch import nn, optim

epoch_size = 70 # training epoch
latent_size = 30

class DataLoaderRepeater:
    """
    Create repeat data loader to make larger dataset out of smaller.
    """
    def __init__(self, loader, batches_per_epoch):
        self.i = 0
        self.batches_per_epoch = batches_per_epoch
        self.loader = loader

    @staticmethod
    def cycle(iterable):
        """Cycle iterable non-caching."""
        while True:
            for x in iterable:
                yield x

    def __iter__(self):
        self.iterator = self.cycle(self.loader)
        self.i = 0
        return self

    def __len__(self):
        return self.batches_per_epoch

    def __next__(self):
        if self.i == self.batches_per_epoch:
            raise StopIteration()

        self.i += 1
        return next(self.iterator)

def visualize_states(model, obs):
    # model: VAE
    # obs of shape [N, 4,84,84]
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)
    batch_size = obs.size(0)
    obs = obs.cuda()
    model.cuda()
    with torch.no_grad():
        recon_img, _, _ = model(obs)
    fig=plt.figure(figsize=(6*(batch_size+1), 20))
    for i in range(batch_size):
        fig.add_subplot(2*batch_size,1,2*i+1)
        plt.imshow(obs[i].permute(0,2,1).cpu().view(84*4,84).permute(1,0))
        fig.add_subplot(2*batch_size,1,2*i+2)
        plt.imshow(recon_img[i].permute(0,2,1).cpu().view(84*4,84).permute(1,0))
    plt.show()

class MLP(nn.Module):
    """Helper module to create MLPs."""
    def __init__(self, dims, activation=nn.ReLU):
        super().__init__()
        blocks = nn.ModuleList()

        for i, (dim_in, dim_out) in enumerate(zip(dims, dims[1:])):
            blocks.append(nn.Linear(dim_in, dim_out))

            if i < len(dims)-2:
                blocks.append(activation())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
    

class Imitator(nn.Module):
    def __init__(self,encoder, net):
        super(Imitator, self).__init__()
        self.encoder = encoder
        self.net = net;
        
    def forward(self, X, G):
        # G (b, s, d)
        # s is sample size
        with torch.no_grad():
            self.z = self.encoder(X)[:,:latent_size]
        self.z = self.z.unsqueeze(1)
        G = G.to(self.z)
        self.z = self.z.repeat((1,G.size(1),1))
        action = self.net(torch.cat([self.z*G,G],axis=2))
        return action

class ImitDataset(Dataset):
    """Atari dataset."""

    def __init__(self, obs, action):
        self.obs = obs
        self.action = action

    def __len__(self):
        return self.obs.size(0)

    def __getitem__(self, idx):
        return self.obs[idx], self.action[idx]

if __name__ == '__main__':
    expert = torch.load("expert_states/expert400000")   
    expert["obs"] = expert["obs"]/255
    imit_dataset = ImitDataset(expert["obs"], expert["action"])
    train_dataset, test_dataset = random_split(imit_dataset, [400000,0])
    train_loader = DataLoader(train_dataset, batch_size=64,
                        shuffle=True)
    if len(test_dataset)>0:
        test_loader = DataLoader(test_dataset, batch_size=64,
                            shuffle=True)
    disentangle_model = VAE(latent_size)
    states = torch.load("checkpoints/checkpoint-30D-130")
    disentangle_model.load_state_dict(states["model_states"])

    model = Imitator(disentangle_model.encoder, MLP([60, 500, 500, 300, 6]))
    model.cuda()
    loss_function = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(epoch_size):
        model.train()
        running_loss = []
        correct_pred = 0
        for obs, expert_action in train_loader:
            obs = obs.to("cuda")
            expert_action = expert_action.to("cuda")
            G = torch.randint(0,2, size=((obs.size(0),1,latent_size)),device="cuda")
            optimizer.zero_grad()
            imit_action = model(obs,G)
            imit_action = imit_action.squeeze(1)
            loss = loss_function(imit_action, expert_action)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())    
            correct_pred += (torch.argmax(imit_action,1)==expert_action).sum().item()
        print("Epch: {} Training loss: {:.3f}  Training acc: {:.3f}".format(
            epoch+1, np.mean(running_loss),correct_pred/len(train_dataset)))

        # test on test_dataset
        if len(test_dataset) > 0:
            running_loss = []
            correct_pred = 0
            model.eval()
            with torch.no_grad():
                for obs, expert_action in test_loader:
                    obs = obs.to("cuda")
                    expert_action = expert_action.to("cuda")
                    G = torch.randint(0,2, size=((obs.size(0),1,latent_size)),device="cuda")
                    optimizer.zero_grad()
                    imit_action = model(obs,G)
                    imit_action = imit_action.squeeze(1)
                    loss = loss_function(imit_action, expert_action)
                    running_loss.append(loss.item())    
                    correct_pred += (torch.argmax(imit_action,1)==expert_action).sum().item()
            print("Epch: {} Test loss: {:.3f}  Test acc: {:.3f}".format(epoch, np.mean(running_loss),correct_pred/len(test_dataset)))

    torch.save(model.state_dict(), "models/graph_parameterized.pt")