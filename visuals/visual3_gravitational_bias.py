from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1200
HEIGHT = 1200
SEED = 42
PARTICLES = 340
STEPS = 720
DT = 1.0
DIFFUSION = 0.0022
ALPHA = 0.032
ATTRACTOR = np.array([0.24, -0.18])
BACKGROUND = "#03040a"
LINE_WIDTH = 0.38
LINE_ALPHA = 0.08


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def generate_visual3() -> Path:
    rng = np.random.default_rng(SEED)
    positions = rng.uniform(-1.15, 1.15, size=(PARTICLES, 2))
    trajectories = np.empty((STEPS + 1, PARTICLES, 2), dtype=float)
    trajectories[0] = positions
    diffusion_scale = np.sqrt(2.0 * DIFFUSION * DT)

    for step in range(1, STEPS + 1):
        force = -ALPHA * (positions - ATTRACTOR)
        diffusion = diffusion_scale * rng.normal(size=(PARTICLES, 2))
        positions = positions + force * DT + diffusion
        trajectories[step] = positions

    initial_distance = np.linalg.norm(trajectories[0] - ATTRACTOR, axis=1)
    normalized_distance = initial_distance / np.max(initial_distance)
    colors = plt.cm.cividis(0.15 + 0.8 * normalized_distance)

    fig, ax = _prepare_axes()
    for particle_index, color in enumerate(colors):
        ax.plot(
            trajectories[:, particle_index, 0],
            trajectories[:, particle_index, 1],
            color=color,
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
        )

    ax.scatter(
        trajectories[-1, :, 0],
        trajectories[-1, :, 1],
        s=4,
        c=colors,
        alpha=0.45,
        linewidths=0,
    )
    ax.scatter(
        ATTRACTOR[0],
        ATTRACTOR[1],
        s=45,
        c="#fff3c2",
        alpha=0.95,
        linewidths=0,
    )

    limit = float(np.max(np.abs(trajectories))) * 1.04
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    output_path = _project_root() / "outputs" / "gravitational_bias.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_visual3()
