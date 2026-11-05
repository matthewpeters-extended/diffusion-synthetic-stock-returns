"""Train the diffusion model from a config, then sample N synthetic windows.

    python scripts/run_experiment.py --config configs/default.json

Writes runs/<name>/{model.pt, samples.npy, losses.npy, config.json} (runs/ is gitignored).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src import config as C
from src import data
from src.diffusion import DDPM, Denoiser


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--steps", type=int, help="override training steps (e.g. a quick smoke run)")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.steps:
        cfg["steps"] = args.steps
    out = data.ROOT / "runs" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    x = torch.from_numpy(data.load_dataset()["train"])
    model = Denoiser(cfg["channels"], tuple(cfg["dilations"]))
    ema = copy.deepcopy(model).requires_grad_(False)
    ddpm = DDPM(cfg["T"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    losses, t0 = [], time.time()
    for step in range(1, cfg["steps"] + 1):
        batch = x[torch.randint(0, len(x), (cfg["batch_size"],))]
        loss = ddpm.loss(model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - cfg["ema_decay"])
        losses.append(loss.item())
        if step % 1000 == 0 or step == cfg["steps"]:
            print(f"step {step:6d}  loss {np.mean(losses[-1000:]):.4f}  {time.time() - t0:.0f}s", flush=True)

    torch.save(ema.state_dict(), out / "model.pt")
    np.save(out / "losses.npy", np.array(losses))

    ema.eval()
    g = torch.Generator().manual_seed(cfg["seed"])
    samples = torch.cat([ddpm.sample(ema, 500, C.WINDOW, g) for _ in range(cfg["n_samples"] // 500)])
    np.save(out / "samples.npy", samples.numpy().astype(np.float32))
    print(f"saved {tuple(samples.shape)} samples to {out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
