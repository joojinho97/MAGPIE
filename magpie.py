"""MAGPIE: Multi-contrast Accelerated Generative Prior for Integrated Estimation.

Two-stage, scan-specific (untrained) generative-prior reconstruction of multi-contrast 3D MRI (e.g. 3D-QALAS) from undersampled
k-space.  No training data: the networks are fitted to the scan being reconstructed.

Stage 1 - volume-sharing training
    Every slice i has a fixed random latent z_i; neighbouring latents are correlated by an AR(1) walk along the slice axis.
    N GenNets are SHARED by all slices: each outer epoch fits them to the current images of randomly drawn slices, the prior of
    slice i is the mean of the N GenNet outputs, and a CG-SENSE step with that prior gives the next images:
        x_i <- argmin_x ||A_i x - y_i||^2 + lambda ||x - mean_n G_n(z_i)||^2
Stage 2 - slice-wise fine-tuning
    Each slice gets its own copy of the stage-1 weights, fine-tuned on that slice only (starting again from the prior-free CG
    image), alternating with the same CG-SENSE step.  The last CG-SENSE output is the reconstruction.

The forward model A_i of slice i is 2D SENSE (coil maps, 2D FFT, sampling mask); the readout direction is assumed fully sampled
and already inverse-Fourier transformed, so the 3D volume splits into independent 2D problems along it.

Input: one .npz file with
    kspace : (E, H, W, S, C) complex   undersampled k-space (zeros where not sampled); E contrasts, S slices, C coils
    coils  : (H, W, S, C)    complex   coil sensitivity maps
    mask   : (H, W) or (H, W, E)       sampling mask
Output: <out>/recon.npy, complex (S, H, W, E), on the input k-space scale.

usage: python magpie.py --data scan.npz --out results/ [--num_models 2 --ft_epochs 3 ...]
"""
import argparse, os, time, json, numpy as np, torch, torch.nn as nn


# ----------------------------------------------------------------------------------------------------------------------------
# network
# ----------------------------------------------------------------------------------------------------------------------------
class GenNet(nn.Module):
    """Decoder from a low-resolution latent (C_in x ceil(H/32) x ceil(W/32)) to 2E output channels (real, imaginary parts of E
    contrasts): five x2 transposed-convolution upsampling stages, the three middle ones sharing their weights."""

    def __init__(self, in_ch, out_ch, nx, ny):
        super().__init__()
        self.nx, self.ny = nx, ny
        up = lambda c: nn.ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1, bias=False)
        conv = lambda a, b: nn.Conv2d(a, b, 3, padding=1, bias=False)
        self.layer1 = nn.Sequential(conv(in_ch, 48), nn.ReLU(inplace=True), up(48))
        self.layer2 = nn.Sequential(conv(48, 96), nn.ReLU(inplace=True), conv(96, 96), nn.ReLU(inplace=True), up(96))
        self.middle = nn.Sequential(conv(96, 96), nn.ReLU(inplace=True), conv(96, 96), nn.ReLU(inplace=True), up(96))
        self.layer4 = nn.Sequential(conv(96, 64), nn.ReLU(inplace=True), conv(64, 32), nn.ReLU(inplace=True), conv(32, out_ch))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, z):
        x = self.layer2(self.layer1(z))
        for _ in range(3):                                   # tied middle stages
            x = self.middle(x)
        x = self.layer4(x)
        sh, sw = (x.shape[-2] - self.nx) // 2, (x.shape[-1] - self.ny) // 2
        return x[..., sh:sh + self.nx, sw:sw + self.ny]


def to_complex(out, ne):                                     # (1, 2E, H, W) real -> (E, H, W) complex
    return torch.complex(out[0, :ne], out[0, ne:])


def to_real(x):                                              # (E, H, W) complex -> (1, 2E, H, W) real
    return torch.cat((x.real, x.imag), 0)[None]


def ar1_latents(n, shape, alpha, gen, device):
    """z_0 ~ U(0,1), z_i = alpha z_{i-1} + (1 - alpha) U(0,1): smooth along the slice axis."""
    z = [torch.rand(shape, generator=gen)]
    for _ in range(1, n):
        z.append(alpha * z[-1] + (1 - alpha) * torch.rand(shape, generator=gen))
    return [v.to(device) for v in z]


# ----------------------------------------------------------------------------------------------------------------------------
# 2D SENSE operator and CG-SENSE with a prior
# ----------------------------------------------------------------------------------------------------------------------------
def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(x):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


class Sense2D:
    """A x = M F (S x) for one slice; x (E, H, W), coils S (C, 1, H, W), mask M (1, E or 1, H, W), data y (C, E, H, W)."""

    def __init__(self, k, coils, mask):
        self.S, self.M = coils, mask
        self.y = k * mask

    def A(self, x):
        return self.M * fft2c(self.S * x[None])

    def AH(self, k):
        return (self.S.conj() * ifft2c(self.M * k)).sum(0)

    def cg(self, prior=None, lmbd=0.0, iters=10):
        """argmin_x ||A x - y||^2 + lmbd ||x - prior||^2 by conjugate gradients, started from A^H y."""
        b = self.AH(self.y)
        if prior is not None:
            b = b + lmbd * prior
        op = lambda v: self.AH(self.A(v)) + lmbd * v
        x = self.AH(self.y)
        r = b - op(x); p = r.clone(); rs = (r.abs() ** 2).sum()
        for _ in range(iters):
            q = op(p); a = rs / (p.conj() * q).sum().real
            x = x + a * p; r = r - a * q; rn = (r.abs() ** 2).sum()
            if rn.sqrt() < 1e-6:
                break
            p = r + (rn / rs) * p; rs = rn
        return x


def load_ops(path, device):
    z = np.load(path)
    k, coils, mask = z["kspace"], z["coils"], z["mask"]                # (E,H,W,S,C), (H,W,S,C), (H,W[,E])
    E, H, W, S, C = k.shape
    keep = np.where((np.abs(coils) ** 2).sum((0, 1, 3)) > 0)[0]        # slices with coil coverage
    scale = float(np.abs(k).max())                                       # one k-space scale for the whole volume
    m = mask.astype(np.float32)
    m = m[None] if m.ndim == 2 else np.transpose(m, (2, 0, 1))          # (1|E, H, W)
    mt = torch.tensor(m, dtype=torch.cfloat, device=device)[None]       # (1, 1|E, H, W)
    ops = []
    for s in keep:
        ks = torch.tensor(np.transpose(k[:, :, :, s, :], (3, 0, 1, 2)) / scale, dtype=torch.cfloat, device=device)   # (C,E,H,W)
        cs = torch.tensor(np.transpose(coils[:, :, s, :], (2, 0, 1))[:, None], dtype=torch.cfloat, device=device)   # (C,1,H,W)
        ops.append(Sense2D(ks, cs, mt))
    return ops, keep, scale, (E, H, W, S)


# ----------------------------------------------------------------------------------------------------------------------------
# MAGPIE
# ----------------------------------------------------------------------------------------------------------------------------
def fit(net, opt, z, target, steps):
    """`steps` Adam steps of ||G(z) - target||^2 (target: (1, 2E, H, W))."""
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((net(z) - target) ** 2).mean()
        loss.backward(); opt.step()


def magpie(ops, E, H, W, a, device, log=print):
    n = len(ops)
    gen = torch.Generator().manual_seed(a.seed)
    torch.manual_seed(a.seed)
    shape = (1, a.in_ch, -(-H // 32), -(-W // 32))
    nets = [GenNet(a.in_ch, 2 * E, H, W).to(device) for _ in range(a.num_models)]
    lat = [ar1_latents(n, shape, a.latent_alpha, gen, device) for _ in range(a.num_models)]   # one latent set per GenNet
    opts = [torch.optim.AdamW(m.parameters(), lr=a.lr) for m in nets]                          # kept across epochs

    def prior(i):                                                                             # ensemble mean of the GenNets
        with torch.no_grad():
            return sum(to_complex(m(zz[i]), E) for m, zz in zip(nets, lat)) / len(nets)

    # ---- stage 1: volume-sharing training ----
    t0 = time.time()
    with torch.no_grad():
        cur = [op.cg(iters=a.cg_iters) for op in ops]                                         # prior-free CG-SENSE start
    for ep in range(a.outer_epochs):
        for m, zz, o in zip(nets, lat, opts):
            m.train()
            for i in torch.randint(0, n, (a.gen_iters,), generator=gen).tolist():           # one random slice per step
                fit(m, o, zz[i], to_real(cur[i]), 1)
            m.eval()
        with torch.no_grad():
            cur = [op.cg(prior(i), a.lmbd, a.cg_iters) for i, op in enumerate(ops)]
        if ep == 0 or (ep + 1) % 5 == 0:
            log(f"[stage 1] epoch {ep + 1}/{a.outer_epochs}  {time.time() - t0:.0f}s")
    shared = [{k: v.clone() for k, v in m.state_dict().items()} for m in nets]

    # ---- stage 2: slice-wise fine-tuning ----
    t1 = time.time(); out = []
    for i, op in enumerate(ops):
        for m, s in zip(nets, shared):
            m.load_state_dict(s)                                                              # every slice starts from stage 1
        o2 = [torch.optim.AdamW(m.parameters(), lr=a.ft_lr or a.lr) for m in nets]
        with torch.no_grad():
            x = op.cg(iters=a.cg_iters)                                                       # restart from the prior-free image
        for _ in range(a.ft_epochs):
            for m, zz, o in zip(nets, lat, o2):
                m.train(); fit(m, o, zz[i], to_real(x), a.ft_iters); m.eval()
            with torch.no_grad():
                x = op.cg(prior(i), a.lmbd, a.cg_iters)
        out.append(x.permute(1, 2, 0).cpu())                                                  # (H, W, E)
        if (i + 1) % 20 == 0 or i == n - 1:
            log(f"[stage 2] slice {i + 1}/{n}  {time.time() - t1:.0f}s")
    return torch.stack(out, 0), time.time() - t0                                              # (S, H, W, E)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_models", type=int, default=2, help="GenNets in the ensemble")
    ap.add_argument("--outer_epochs", type=int, default=55, help="stage-1 epochs (GenNet fit + CG-SENSE)")
    ap.add_argument("--gen_iters", type=int, default=667, help="stage-1 Adam steps per epoch and GenNet (one slice each)")
    ap.add_argument("--ft_epochs", type=int, default=3, help="stage-2 epochs per slice")
    ap.add_argument("--ft_iters", type=int, default=667, help="stage-2 Adam steps per epoch and GenNet")
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--ft_lr", type=float, default=0.0, help="0: same as --lr")
    ap.add_argument("--lmbd", type=float, default=0.08, help="prior weight of CG-SENSE")
    ap.add_argument("--cg_iters", type=int, default=10)
    ap.add_argument("--in_ch", type=int, default=24, help="latent channels")
    ap.add_argument("--latent_alpha", type=float, default=0.98, help="AR(1) correlation of neighbouring slice latents")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = torch.device(a.device)
    ops, keep, scale, (E, H, W, S) = load_ops(a.data, dev)
    print(f"[data] {E} contrasts, {H}x{W}, {len(keep)}/{S} slices with coil coverage, device {dev}", flush=True)
    rec, sec = magpie(ops, E, H, W, a, dev, log=lambda s: print(s, flush=True))
    vol = np.zeros((S, H, W, E), np.complex64); vol[keep] = rec.numpy() * scale
    np.save(os.path.join(a.out, "recon.npy"), vol)
    json.dump({**vars(a), "seconds": round(sec, 1)}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"[done] {sec:.0f}s -> {os.path.join(a.out, 'recon.npy')}", flush=True)


if __name__ == "__main__":
    main()
