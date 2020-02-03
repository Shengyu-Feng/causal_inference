import os
import numpy as np
import argparse
import torch
from torch import nn, optim
import torch.nn.init as init
from scipy.misc import imsave
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

class AtariDataset(Dataset):
    """Atari dataset."""

    def __init__(self, path):
        imgs = np.load(path).astype(np.float32)/255
        self.imgs = torch.FloatTensor(imgs)

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, idx):
        return self.imgs[idx]

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class AddCoords(nn.Module):
    # Add coordinates for tensor
    def __init__(self, x_dim, y_dim, skiptile=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.skiptile = skiptile

    def forward(self, tensor):
        
        if not self.skiptile:
            tensor = tensor.repeat(1, 1, self.x_dim, self.y_dim)
            
        batch_size_tensor = tensor.size(0)
        xx_ones = torch.ones([batch_size_tensor, self.x_dim]).unsqueeze(-1)
        xx_range = torch.arange(self.y_dim).unsqueeze(0).repeat(batch_size_tensor, 1)
        xx_range = xx_range.unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range.to(torch.float))
        xx_channel = xx_channel.unsqueeze(1)
        
        yy_ones = torch.ones([batch_size_tensor, self.y_dim]).unsqueeze(-1)
        yy_range = torch.arange(self.x_dim).unsqueeze(0).repeat(batch_size_tensor, 1)
        yy_range = yy_range.unsqueeze(1)
        yy_channel = torch.matmul(yy_ones, yy_range.to(torch.float))
        yy_channel = yy_channel.unsqueeze(1)
        
        xx_channel /= (self.x_dim-1)
        yy_channel /= (self.y_dim-1)
        
        xx_channel = xx_channel*2 - 1
        
        yy_channel = yy_channel*2 - 1
     
        ret = torch.cat([tensor,xx_channel.cuda(),yy_channel.cuda()], axis=1)
        return ret
    
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        # recognition model
        self.encoder = nn.Sequential(
            AddCoords(84,84,True),
            nn.Conv2d(6, 32, 6, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 6, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 6, 2, 1),
            nn.ReLU(True),
            View((-1,4096)),
            nn.Linear(4096,512),
            nn.ReLU(True),
            nn.Linear(512,2*z_dim)
        )
        # generation model
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 4096),
            View((-1, 64, 8, 8)),               
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 6, 2, 1),      
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 6, 2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 4,  6, 2),   
            nn.Sigmoid()
        )
            
    def reparametrize(self, mu, logvar):
        epsilon = mu.data.new(mu.size()).normal_()
        sigma = logvar.mul(0.5).exp_()
        z = mu + sigma * epsilon
        return z
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def loss_function(x_hat, x, mu, logvar):
    MSE = F.mse_loss(x_hat, x, size_average=False)
    DKL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    MSE /= x.shape[0]
    DKL /= x.shape[0]
    return MSE, DKL

def save_checkpoint(model, global_iter, file_path):
        states = {'model_states':model.state_dict(),
               'global_iter':global_iter
              }
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)

def load_checkpoint(args):
    file_path = args.checkpoint
    model = VAE(args.latent_size) 
    global_iter = 0
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_states'])
        global_iter = checkpoint["global_iter"]
        print("=> loaded checkpoint '{}'".format(file_path))
    else:
        print("=> no checkpoint found at '{}'".format(file_path))
    return model, global_iter   



def main(args):   
    C_max = args.capacity_limit 
    stop_iter = args.capacity_change_duration
    beta = args.beta 
    model, global_iter = load_checkpoint(args.load_path) 
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    atari_dataset = AtariDataset(args.dataset)
    train_loader = DataLoader(atari_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)
    for epoch in range(1, args.epoch_size+1):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            global_iter += 1
            data = data.cuda()
            recon_batch, mu, logvar = model(data)
            optimizer.zero_grad()
            mse, dkl = loss_function(recon_batch, data, mu, logvar)
            C = min(C_max/stop_iter*global_iter, C_max)
            loss = mse + beta*(dkl-C).abs()        
            loss.backward()
            train_loss += loss.cpu().data.numpy()
            optimizer.step()
            if batch_idx % 500 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMSE: {:.6f}\tDKL: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.cpu().data.numpy(), mse.item(), dkl.item()))
        save_checkpoint(model, global_iter, args.save_path)
        org_img = data.cpu().numpy()[0][0]
        reconstr_img = recon_batch.data.cpu().numpy()[0][0]
        imsave("org_img.png",      org_img)
        imsave("reconstr_img.png", reconstr_img)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beta-VAE')

    parser.add_argument('--epoch_size', default=20, type=int, help='epoch size')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--capacity_limit', default=25.0, type=float, help='encoding capacity limit param for latent loss')
    parser.add_argument('--capacity_change_duration', default=40000, type=int, help='encoding capacity change duration')
    parser.add_argument('--latent_size', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=1, type=int, help='beta param for latent loss')
    parser.add_argument('--load_path', default="checkpoints/checkpoint-0", type=str, help='path to load the checkpoint')
    parser.add_argument('--save_path', default="checkpoints/checkpoint-0", type=str, help='path to save the checkpoint')
    parser.add_argument('--dataset', default="ministates.npy", type=str, help='path to the dataset')
    args = parser.parse_args()
    
    main(args)

