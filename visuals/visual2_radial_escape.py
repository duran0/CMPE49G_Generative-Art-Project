from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1200
HEIGHT = 1200
SEED = 42
PARTICLES = 1500
STEPS = 280
DT = 1.0
DIFFUSION = 0.0045
SNAPSHOT_EVERY = 4
BACKGROUND = "#010203"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def generate_visual2() -> Path:
    rng = np.random.default_rng(SEED)
    positions = np.zeros((PARTICLES, 2), dtype=float)
    history = np.empty((STEPS + 1, PARTICLES, 2), dtype=float)
    history[0] = positions
    diffusion_scale = np.sqrt(2.0 * DIFFUSION * DT)

    for step in range(1, STEPS + 1):
        positions = positions + diffusion_scale * rng.normal(size=(PARTICLES, 2))
        history[step] = positions

    sampled_history = history[::SNAPSHOT_EVERY]
    sample_times = np.repeat(
        np.linspace(0.0, 1.0, sampled_history.shape[0]),
        PARTICLES,
    )
    sampled_points = sampled_history.reshape(-1, 2)
    colors = plt.cm.plasma(0.08 + 0.88 * sample_times)

    fig, ax = _prepare_axes()
    ax.scatter(
        sampled_points[:, 0],
        sampled_points[:, 1],
        s=0.7,
        c=colors,
        alpha=0.08,
        linewidths=0,
    )
    ax.scatter(
        history[-1, :, 0],
        history[-1, :, 1],
        s=3,
        c=plt.cm.plasma(np.full(PARTICLES, 0.98)),
        alpha=0.18,
        linewidths=0,
    )

    limit = float(np.max(np.abs(sampled_points))) * 1.08
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    output_path = _project_root() / "outputs" / "radial_escape.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_visual2()
