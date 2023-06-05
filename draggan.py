import torch, math
import numpy as np
from stylegan2 import Generator
import torch.nn.functional as functional

def linear(feature, p0, p1, d, axis=0):
    f0 = feature[..., p0[0], p0[1]]
    f1 = feature[..., p1[0], p1[1]]
    weight = abs(d[axis])
    f = (1 - weight) * f0 + weight * f1
    return f

def bilinear(feature, qi, d):
    y0, x0 = qi
    dy, dx = d
    d = (dx, dy)
    dx = 1 if dx >= 0 else -1
    dy = 1 if dy >= 0 else -1
    x1 = x0 + dx
    y1 = y0 + dy
    fx1 = linear(feature, (x0, y0), (x1, y0), d, axis=0)
    fx2 = linear(feature, (x0, y1), (x1, y1), d, axis=0)
    weight = abs(d[1])
    fx = (1 - weight) * fx1 + weight * fx2
    return fx

def motion_supervision(F0, F, pi, ti, r1=3, M=None):
    F = functional.interpolate(F, [256, 256], mode="bilinear")
    F0 = functional.interpolate(F0, [256, 256], mode="bilinear")

    dw, dh = ti[0] - pi[0], ti[1] - pi[1]
    norm = math.sqrt(dw**2 + dh**2)
    w = (max(0, pi[0] - r1), min(256, pi[0] + r1))
    h = (max(0, pi[1] - r1), min(256, pi[1] + r1))
    d = torch.tensor(
        (dw / norm, dh / norm),
        dtype=F.dtype, device=F.device,
    ).reshape(1, 1, 1, 2)
    grid_h, grid_w = torch.meshgrid(
        torch.tensor(range(h[0], h[1])),
        torch.tensor(range(w[0], w[1])),
        indexing='xy',
    )
    grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0)
    grid = (grid / 255 - 0.5) * 2
    grid_d = grid + 2 * d / 255

    sample = functional.grid_sample(
        F, grid, mode='bilinear', padding_mode='border',
        align_corners=True,
    )
    sample_d = functional.grid_sample(
        F, grid_d, mode='bilinear', padding_mode='border',
        align_corners=True,
    )

    loss = (sample_d - sample.detach()).abs().mean(1).sum()

    return loss

@torch.no_grad()
def point_tracking(F0, F, pi, p0, r2=12):
    F = functional.interpolate(F, [256, 256], mode="bilinear")
    F0 = functional.interpolate(F0, [256, 256], mode="bilinear")
    x = (max(0, pi[0] - r2), min(256, pi[0] + r2))
    y = (max(0, pi[1] - r2), min(256, pi[1] + r2))
    base = F0[..., p0[1], p0[0]].reshape(1, -1, 1, 1)
    diff = (F[..., y[0]:y[1], x[0]:x[1]] - base).abs().mean(1)
    idx = diff.argmin()
    dy = int(idx / (x[1] - x[0]))
    dx = int(idx % (x[1] - x[0]))
    npi = (x[0] + dx, y[0] + dy)
    return npi

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class DragGAN():
    def __init__(self, device, layer_index=6):
        self.generator = Generator(256, 512, 8).to(device)
        requires_grad(self.generator, False)
        self._device = device
        self.layer_index = layer_index
        self.latent = None
        self.F0 = None
        self.optimizer = None
        self.p0 = None

    def load_ckpt(self, path):
        print(f'loading checkpoint from {path}')
        ckpt = torch.load(path, map_location=self._device)
        self.generator.load_state_dict(ckpt["g_ema"], strict=False)
        print('loading checkpoint successed!')

    def to(self, device):
        if self._device != device:
            self.generator = self.generator.to(device)
            self._device = device

    @torch.no_grad()
    def generate_image(self, seed):
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, 512).astype(np.float32)
        ).to(self._device)
        image, self.latent, self.F0 = self.generator(
            [z], return_latents=True, return_features=True, randomize_noise=False,
        )
        image, self.F0 = image[0], self.F0[self.layer_index*2+1].detach()
        image = image.detach().cpu().permute(1, 2, 0).numpy()
        image = (image / 2 + 0.5).clip(0, 1).reshape(-1)
        return image

    @property
    def device(self):
        return self._device

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)

    def step(self, points):
        if self.optimizer is None:
            len_pts = (len(points) // 2) * 2
            if len_pts == 0:
                print('Select at least one pair of points')
                return False, None
            self.trainable = self.latent[:, :self.layer_index*2, :].detach(
            ).requires_grad_(True)
            self.fixed = self.latent[:, self.layer_index*2:, :].detach(
            ).requires_grad_(False)
            self.optimizer = torch.optim.Adam([self.trainable], lr=2e-3)
            points = points[:len_pts]
            self.p0 = points[::2]
        self.optimizer.zero_grad()
        trainable_fixed = torch.cat([self.trainable, self.fixed], dim=1)
        image, _, features = self.generator(
            [trainable_fixed], input_is_latent=True,
            return_features=True, randomize_noise=False,
        )
        features = features[self.layer_index*2+1]
        loss = 0
        for i in range(len(self.p0)):
            loss += motion_supervision(self.F0, features, points[2*i], points[2*i+1])
        print(loss)
        loss.backward()
        self.optimizer.step()
        image, _, features = self.generator(
            [trainable_fixed], input_is_latent=True,
            return_features=True, randomize_noise=False,
        )
        features = features[self.layer_index*2+1]
        image = image[0].detach().cpu().permute(1, 2, 0).numpy()
        image = (image / 2 + 0.5).clip(0, 1).reshape(-1)
        for i in range(len(self.p0)):
            points[2*i] = point_tracking(self.F0, features, points[2*i], self.p0[i])
        return True, (points, image)
