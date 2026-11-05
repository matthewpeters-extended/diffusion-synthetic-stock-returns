import torch
import torch.nn.functional as F

from src.diffusion import DDPM, Conv1d, Denoiser


def test_fast_conv_matches_torch_conv():
    torch.manual_seed(0)
    for d in (1, 4, 32):
        c = Conv1d(3, 5, 3, padding=d, dilation=d)
        x = torch.randn(2, 3, 64)
        ref = F.conv1d(x, c.weight, c.bias, padding=d, dilation=d)
        torch.testing.assert_close(c(x), ref, atol=1e-5, rtol=1e-5)


def test_denoiser_shape():
    m = Denoiser(channels=16, dilations=(1, 2))
    x = torch.randn(4, 64)
    assert m(x, torch.randint(0, 1000, (4,))).shape == (4, 64)


def test_loss_finite_and_decreases():
    torch.manual_seed(0)
    m = Denoiser(channels=16, dilations=(1, 2, 4))
    ddpm = DDPM(T=100)
    x0 = torch.randn(64, 64) * torch.linspace(0.5, 2.0, 64)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss = ddpm.loss(m, x0)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10


def test_q_sample_endpoints():
    ddpm = DDPM(T=1000)
    x0, noise = torch.ones(2, 64), torch.zeros(2, 64)
    torch.testing.assert_close(ddpm.q_sample(x0, torch.tensor([0, 0]), noise), x0, atol=1e-2, rtol=0)
    assert ddpm.abar[-1] < 1e-3  # fully noised at t = T


def test_sample_shape_and_finite():
    m = Denoiser(channels=16, dilations=(1,))
    out = DDPM(T=20).sample(m, 3, 64, torch.Generator().manual_seed(0))
    assert out.shape == (3, 64) and torch.isfinite(out).all()
