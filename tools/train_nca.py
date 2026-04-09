#!/usr/bin/env python3
"""
Offline trainer for the Rust/WGPU CellForge runtime.

Exports weights in a binary format consumed by src/model.rs:
- magic: 4 bytes: NCA1
- dims: state_dim, input_dim, hidden_dim, reserved (u32 LE)
- payload: w1, b1, w2, b2 as f32 LE contiguous arrays
"""

from __future__ import annotations

import argparse
import math
import os
import random
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_DIM = 8
ANCHOR_DIM = 4
INPUT_DIM = STATE_DIM * 2 + ANCHOR_DIM
HIDDEN_DIM = 32


@dataclass
class TrainConfig:
    width: int = 128
    height: int = 128
    batch_size: int = 8
    epochs: int = 1200
    min_steps: int = 24
    max_steps: int = 72
    damage_every: int = 8
    learning_rate: float = 2e-3

    dt: float = 0.18
    anchor_gain: float = 0.90
    neighbor_mix: float = 0.62
    fire_rate: float = 0.58
    max_delta: float = 0.35
    alive_threshold: float = 0.08


class NCAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, STATE_DIM)
        self.reset_seeded()

    def reset_seeded(self) -> None:
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

            for i in range(INPUT_DIM):
                self.fc1.weight[i, i] = 1.12

            g = torch.Generator().manual_seed(1337)
            self.fc1.weight[INPUT_DIM:, :] = torch.randn(
                (HIDDEN_DIM - INPUT_DIM, INPUT_DIM), generator=g
            ) * 0.32
            self.fc1.bias[INPUT_DIM:] = torch.randn((HIDDEN_DIM - INPUT_DIM,), generator=g) * 0.06

            for o in range(STATE_DIM):
                self.fc2.weight[o, o] = -0.65
                self.fc2.weight[o, STATE_DIM + o] = 0.82
                self.fc2.weight[o, STATE_DIM * 2 + (o % ANCHOR_DIM)] = 0.58
                self.fc2.weight[o, (o + 1) % STATE_DIM] += 0.08
                self.fc2.weight[o, STATE_DIM + ((o + 2) % STATE_DIM)] += 0.05

            self.fc2.weight[:, INPUT_DIM:] += torch.randn(
                (STATE_DIM, HIDDEN_DIM - INPUT_DIM), generator=g
            ) * 0.08
            self.fc2.bias[:] = -0.02

    @staticmethod
    def _channelwise_conv3x3(state: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = state.shape
        x = state.permute(0, 3, 1, 2)
        weight = kernel.expand(channels, 1, 3, 3)
        y = F.conv2d(x, weight, padding=1, groups=channels)
        return y.permute(0, 2, 3, 1)

    def step(self, state: torch.Tensor, anchor: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
        kernel = state.new_tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
        ) / 9.0
        avg = self._channelwise_conv3x3(state, kernel)

        mixed = state + (avg - state) * cfg.neighbor_mix
        inp = torch.cat([state, mixed, anchor * cfg.anchor_gain], dim=-1)

        hidden = torch.tanh(self.fc1(inp))
        delta = torch.tanh(self.fc2(hidden)).clamp(-cfg.max_delta, cfg.max_delta)

        fire_mask = (torch.rand_like(state[..., :1]) <= cfg.fire_rate).to(state.dtype)
        next_state = torch.clamp(state + cfg.dt * delta * fire_mask, 0.0, 1.0)

        alpha = state[..., :1].permute(0, 3, 1, 2)
        alive = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1).permute(0, 2, 3, 1)
        alive_mask = (alive > cfg.alive_threshold) | (anchor[..., :1] > 0.04)
        next_state = torch.where(alive_mask, next_state, torch.clamp(next_state - 0.01, 0.0, 1.0))
        return next_state


def build_anchor_field(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(1.0, -1.0, steps=height, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    r = torch.sqrt(xx * xx + yy * yy)
    theta = torch.atan2(yy, xx)

    anchor = torch.zeros((height, width, ANCHOR_DIM), device=device)
    anchor[..., 0] = torch.clamp(1.0 - r * 1.35, 0.0, 1.0)
    anchor[..., 1] = 0.5 + 0.5 * torch.sin(11.0 * r)
    anchor[..., 2] = 0.5 + 0.5 * torch.cos(7.0 * theta)
    anchor[..., 3] = torch.clamp(1.0 - torch.abs(r - 0.38) * 6.0, 0.0, 1.0)
    return anchor


def build_target(anchor: torch.Tensor) -> torch.Tensor:
    height, width, _ = anchor.shape
    target = torch.zeros((height, width, STATE_DIM), device=anchor.device)

    target[..., 0] = torch.clamp(anchor[..., 0] * 0.75 + anchor[..., 3] * 0.55, 0.0, 1.0)
    target[..., 1] = anchor[..., 1]
    target[..., 2] = anchor[..., 2]
    target[..., 3] = anchor[..., 3]
    target[..., 4] = torch.clamp(0.4 * anchor[..., 0] + 0.6 * anchor[..., 1], 0.0, 1.0)
    target[..., 5] = torch.clamp(0.5 * anchor[..., 2] + 0.5 * anchor[..., 3], 0.0, 1.0)
    target[..., 6] = torch.clamp(anchor[..., 1] * anchor[..., 2], 0.0, 1.0)
    target[..., 7] = torch.clamp(0.3 + 0.7 * anchor[..., 0], 0.0, 1.0)
    return target


def random_seed_state(
    target: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    state = torch.rand((batch_size, target.shape[0], target.shape[1], STATE_DIM), device=device) * 0.03
    center = target.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    state = torch.clamp(state + center * 0.12, 0.0, 1.0)
    return state


def apply_random_damage(state: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    batch, height, width, _ = state.shape
    yy = torch.arange(height, device=state.device).view(height, 1)
    xx = torch.arange(width, device=state.device).view(1, width)

    for b in range(batch):
        cx = random.randint(width // 8, width * 7 // 8)
        cy = random.randint(height // 8, height * 7 // 8)
        radius = random.randint(min(width, height) // 10, min(width, height) // 4)

        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = (dist2 <= radius * radius).to(state.dtype).unsqueeze(-1)
        state[b] = state[b] * (1.0 - mask * strength)

    return state


def export_weights(model: NCAModel, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    w1 = model.fc1.weight.detach().cpu().numpy().astype("<f4", copy=False)
    b1 = model.fc1.bias.detach().cpu().numpy().astype("<f4", copy=False)
    w2 = model.fc2.weight.detach().cpu().numpy().astype("<f4", copy=False)
    b2 = model.fc2.bias.detach().cpu().numpy().astype("<f4", copy=False)

    with open(path, "wb") as f:
        f.write(struct.pack("<4sIIII", b"NCA1", STATE_DIM, INPUT_DIM, HIDDEN_DIM, 0))
        f.write(w1.tobytes(order="C"))
        f.write(b1.tobytes(order="C"))
        f.write(w2.tobytes(order="C"))
        f.write(b2.tobytes(order="C"))


def train(cfg: TrainConfig, output_path: str, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = NCAModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    anchor = build_anchor_field(cfg.height, cfg.width, device)
    target = build_target(anchor)

    anchor_batch = anchor.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1)
    target_batch = target.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        state = random_seed_state(target, cfg.batch_size, device)

        steps = random.randint(cfg.min_steps, cfg.max_steps)
        for t in range(steps):
            state = model.step(state, anchor_batch, cfg)
            if (t + 1) % cfg.damage_every == 0:
                state = apply_random_damage(state, strength=1.0)

        visible_loss = F.mse_loss(state[..., :4], target_batch[..., :4])
        hidden_reg = (state[..., 4:] ** 2).mean() * 5e-4
        loss = visible_loss + hidden_reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"epoch={epoch:04d} steps={steps:02d} "
                f"loss={loss.item():.6f} visible={visible_loss.item():.6f}"
            )

    export_weights(model, output_path)
    print(f"Saved trained weights to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CellForge weights for Rust/WGPU runtime")
    parser.add_argument("--output", default="assets/nca_weights.bin", help="Output weights file")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        width=args.width,
        height=args.height,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    train(cfg=cfg, output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
