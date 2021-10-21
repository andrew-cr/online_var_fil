#%%
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torchvision.transforms as transforms
import PIL
import torch.nn.functional as F

BATCH_SIZE = 32

class MyData(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        transform = torchvision.transforms.ToTensor()
        self.data = torch.zeros(4010, 3, 32, 32)
        print("Loading images")
        for i in tqdm(range(4010)):
            img_path = os.path.join(img_dir, 'image_{}.png'.format(i))
            pil_image = PIL.Image.open(img_path)
            image = transform(pil_image)
            self.data[i, :, :, :] = image

        self.data = self.data.to('cuda')

        self.data_mean = torch.mean(self.data, dim=0).cuda()
        self.data_std = torch.std(self.data, dim=0).cuda()


        self.data = (self.data - self.data_mean) / self.data_std


    def __len__(self):
        return 4010

    def __getitem__(self, index):
        img = self.data[index]
        return img

def unnormalize(x, dataset):
    return dataset.data_mean + x * dataset.data_std

dataset = MyData(r'datasets\train1')
dataloader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

# Models from https://github.com/karpathy/deep-vector-quantization
class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out


class DeepMindEncoder(nn.Module):

    def __init__(self, input_channels=3, n_hid=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
        )

        self.output_channels = 2 * n_hid
        self.output_stide = 4

    def forward(self, x):
        return self.net(x)


class DeepMindDecoder(nn.Module):

    def __init__(self, n_init=32, n_hid=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_init, 2*n_hid, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            nn.ConvTranspose2d(2*n_hid, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_hid, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.nn = nn.Sequential(
            DeepMindEncoder(input_channels=3),
            nn.Flatten(),
            nn.Linear(128*8*8, 2*self.z_dim)
        )

    def forward(self, x):
        return self.nn(x)

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(self.z_dim, 32*8*8)
        self.dmdecoder = DeepMindDecoder(output_channels=3)

    def forward(self, z):
        x1 = self.linear(z)
        x2 = x1.view(-1, 32, 8, 8)
        return self.dmdecoder(x2)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.z_dim = 32
        self.device = 'cuda'

        self.encoder = Encoder(self.z_dim)
        self.decoder = Decoder(self.z_dim)

        self.recon_loss = nn.L1Loss(reduction='none')

        self.learning_rate = 0.001


    def get_loss(self, batch):
        """
            x (batch_num, num_channels, H, W)
        """
        x = batch
        batch_size = x.shape[0]
        z_stats = self.encoder(x)
        z = z_stats[:, 0:self.z_dim] + \
            torch.exp(z_stats[:, self.z_dim:]) * \
                torch.randn(batch_size, self.z_dim, device=self.device)
        images = self.decoder(z)

        neg_log_p_x_z = torch.sum(self.recon_loss(images, x), dim=[1,2,3])

        KL = 0.5 * torch.sum(
            z_stats[:, 0:self.z_dim]**2 + \
            torch.exp(2*z_stats[:, self.z_dim:]) - 1 - \
            2*z_stats[:, self.z_dim:], dim=1)

        mean_neg_elbo = torch.mean(neg_log_p_x_z + KL)

        return mean_neg_elbo

    def forward(self, num_samples):
        z = torch.randn(num_samples, self.z_dim, device=self.device)
        images = self.decoder(z)
        return images

model = Model().cuda()
#%%

optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)

losses = []
print("Training model")
for epoch in tqdm(range(100)):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
#%%
plt.plot(losses)
plt.show()

#%%
z = torch.randn(16, model.z_dim, device=model.device)
image_batch = model.decoder(z)
fig, ax = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        ax[i, j].imshow(
            unnormalize(image_batch[i * 4 + j, :, :, :], dataset)\
                .transpose(0,1).transpose(1,2).cpu().detach().numpy()
        )
plt.show()

# %%
torch.save(model.decoder.state_dict(), 'vae_decoder.pt')
torch.save(model.encoder.state_dict(), 'vae_encoder.pt')
# %%
