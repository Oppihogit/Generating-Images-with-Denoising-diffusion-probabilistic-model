#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
#source fromhttps://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/tree/main/Diffusion
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import os
from typing import Dict
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
import math
from torch.nn import init
from torch.optim.lr_scheduler import _LRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Diffusion.py
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Diffusion.py
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Diffusion.py
class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            if time_step%50==0:
                print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Scheduler.py
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


##train function
#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Train.py
def train(modelConfig: Dict):
  # Set the device to run on
  device = torch.device(modelConfig["device"])

  # Load the data using the LSUN dataset and apply transformations
  dataloader = torch.utils.data.DataLoader(
      torchvision.datasets.LSUN(
          root="/content/drive/MyDrive/Deep learning assignment", 
          classes=['church_outdoor_train'],
          transform=torchvision.transforms.Compose([
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Resize((modelConfig["img_size"], modelConfig["img_size"])),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
          ])
      ),
      shuffle=True, 
      batch_size=modelConfig["batch_size"], 
      drop_last=True
  )

  # Initialize the UNet model
  net_model = UNet(
      T=modelConfig["T"], 
      ch=modelConfig["channel"], 
      ch_mult=modelConfig["channel_mult"], 
      attn=modelConfig["attn"],
      num_res_blocks=modelConfig["num_res_blocks"], 
      dropout=modelConfig["dropout"]
  ).to(device)

  # Load a pre-trained model if specified
  if modelConfig["training_load_weight"] is not None:
      net_model.load_state_dict(
          torch.load(
              os.path.join(
                  modelConfig["save_weight_dir"], 
                  modelConfig["training_load_weight"]
              ), 
              map_location=device
          )
      )

  # Initialize the optimizer and schedulers
  optimizer = torch.optim.AdamW(
      net_model.parameters(), 
      lr=modelConfig["lr"], 
      weight_decay=1e-4
  )
  cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer=optimizer, 
      T_max=modelConfig["epoch"], 
      eta_min=0, 
      last_epoch=-1
  )
  warmUpScheduler = GradualWarmupScheduler(
      optimizer=optimizer, 
      multiplier=modelConfig["multiplier"], 
      warm_epoch=modelConfig["epoch"] // 10, 
      after_scheduler=cosineScheduler
  )

  # Initialize the GaussianDiffusionTrainer
  trainer = GaussianDiffusionTrainer(
      net_model, 
      modelConfig["beta_1"], 
      modelConfig["beta_T"], 
      modelConfig["T"]
  ).to(device)

  # start training
  for e in range(modelConfig["epoch"]):
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
      for images, labels in tqdmDataLoader:
          # train
          optimizer.zero_grad()
          x_0 = images.to(device)
          loss = trainer(x_0).sum() / 1000.
          loss.backward()
          torch.nn.utils.clip_grad_norm_(
              net_model.parameters(), modelConfig["grad_clip"])
          optimizer.step()
          tqdmDataLoader.set_postfix(ordered_dict={
              "epoch": e,
              "loss: ": loss.item(),
              "img shape: ": x_0.shape,
              "LR": optimizer.state_dict()['param_groups'][0]["lr"]
          
              })
      warmUpScheduler.step()
      torch.save(net_model.state_dict(), os.path.join(modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

      with torch.no_grad():
        sampler = GaussianDiffusionSampler(net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        noisyImage = torch.randn(size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        #print('loss ' + str(loss.mean()))
        plt.rcParams['figure.dpi'] = 100
        plt.grid(False)
        plt.imshow(torchvision.utils.make_grid(sampledImgs[:8]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
        plt.show()
        plt.pause(0.0001)


def sample_with_weight():
    # the code used to generate 64 samples based on the 150th weight
    with torch.no_grad():
        net_model = UNet(T=1000, ch=Parameter["channel"], ch_mult=Parameter["channel_mult"], attn=Parameter["attn"],
                         num_res_blocks=Parameter["num_res_blocks"], dropout=Parameter["dropout"]).to(device)
        weight = torch.load(("ckpt_149_.pt"), map_location=device)
        net_model.load_state_dict(weight)

        sampler = GaussianDiffusionSampler(net_model, Parameter["beta_1"], Parameter["beta_T"], Parameter["T"]).to(
            device)
        noisy = torch.randn(size=[64, 3, 48, 48], device=device)
        # saveNoisy = torch.clamp(noisy * 0.5 + 0.5, 0, 1)
        noisy_img = sampler(noisy)
        noisy_img_re = noisy_img * 0.5 + 0.5
        plt.rcParams['figure.dpi'] = 175
        plt.grid(False)
        plt.imshow(torchvision.utils.make_grid(noisy_img_re).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0),
                   cmap=plt.cm.binary)
        plt.show()

        # show some interpolations
        import numpy as np
        # source from:https://colab.research.google.com/gist/cwkx/ef95ceb3184ead1d364246c9047b2aef/dl-generative-model-assignment.ipynb
        # now show some interpolations (note you do not have to do linear interpolations as shown here, you can do non-linear or gradient-based interpolation if you wish)
        col_size = int(np.sqrt(64))

        z0 = noisy[0:col_size].repeat(col_size, 1, 1, 1)  # z for top row
        z1 = noisy[64 - col_size:].repeat(col_size, 1, 1, 1)  # z for bottom row

        t = torch.linspace(0, 1, col_size).unsqueeze(1).repeat(1, col_size).view(64, 1, 1, 1).to(device)

        lerp_z = (1 - t) * z0 + t * z1  # linearly interpolate between two points in the latent space
        lerp_g = sampler(lerp_z)  # sample the model at the resulting interpolated latents
        lerp_g = lerp_g * 0.5 + 0.5
        plt.rcParams['figure.dpi'] = 175
        plt.grid(False)
        plt.imshow(torchvision.utils.make_grid(lerp_g).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0),
                   cmap=plt.cm.binary)
        plt.show()

    # the code used to select the best 6 samples from the 64 samples by lpips package
    import lpips
    # source code: https://blog.csdn.net/m0_49629753/article/details/121547634
    loss_img = lpips.LPIPS(net='alex', spatial=True)
    loss_img.cuda()
    img_train = []

    distance = []

    # load the Lsun dataset
    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.LSUN(
            root="church",
            classes=['church_outdoor_train'],
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((48, 48)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        ),
        shuffle=True,
        batch_size=64,
        drop_last=True
    )
    num = 0

    for features, labels in dataloader:
        features = features.to(device)
        break
        if num % 64 == 0:
            print(num * 64)
        img_train = features
        img_sample = noisy_img
        loss1 = loss_img.forward(img_train, img_sample)
        distance.append(loss1.mean().item())
        num = num + 1

    score_list = [1 for i in list(range(64))]
    index = 0

    # calculate the lpips matrix of every samples
    for i in noisy_img:
        imgs = i.repeat(64, 1, 1, 1)
        loss1 = loss_img.forward(imgs, features)
        score_list[index] = loss1.mean().item()
        index = index + 1
        print(index)
    a1 = score_list.index(min(score_list))
    print(score_list[a1])
    score_list[a1] = 1
    a2 = score_list.index(min(score_list))
    print(score_list[a2])
    score_list[a2] = 1
    a3 = score_list.index(min(score_list))
    print(score_list[a3])
    score_list[a3] = 1
    a4 = score_list.index(min(score_list))
    print(score_list[a4])
    score_list[a4] = 1
    a5 = score_list.index(min(score_list))
    print(score_list[a5])
    score_list[a5] = 1
    a6 = score_list.index(min(score_list))
    print(score_list[a6])
    score_list[a6] = 1
    plt.grid(False)

    # show the best samples with the lowest lpips matrix.
    imgbar = torch.cat((noisy_img[a1], noisy_img[a2], noisy_img[a3], noisy_img[a4], noisy_img[a5], noisy_img[a6]),
                       dim=2)
    plt.imshow(
        torchvision.utils.make_grid(imgbar * 0.5 + 0.5).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0),
        cmap=plt.cm.binary)
    plt.savefig('nice.png', transparent=True)
    plt.show()
    plt.pause(0.0001)

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Main.py
##sample from the model
Parameter = {
    "epoch": 100,
    "batch_size": 256,
    "T": 1000,
    "channel": 128,
    "channel_mult": [1, 2, 3, 4],
    "attn": [2],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 1e-4,
    "multiplier": 2.,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    "img_size": 48,
    "grad_clip": 1.,
    "device": "cuda",
    "training_load_weight": None,
    "save_weight_dir": "./Checkpoints/",
    "test_load_weight": "ckpt_000_.pt",
    "sampled_dir": "./SampledImgs/",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "SampledNoGuidenceImgs.png",
    "nrow": 8
    }

#use this code can train the model
#train(Parameter)

#this code is used to sample
sample_with_weight()




