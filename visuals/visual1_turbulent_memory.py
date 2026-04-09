from pathlib import Path

import matplotlib
import numpy as np
from noise import pnoise2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1920
HEIGHT = 1080
SEED = 77
PARTICLES = 300
STEPS = 600
DT = 1.0
DIFFUSION = 0.0036
DRIFT = 0.014
NOISE_SCALE = 2.4
OCTAVES = 2
LINE_WIDTH = 0.35
LINE_ALPHA = 0.085
BACKGROUND = "#02050b"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect(HEIGHT / WIDTH)
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    return fig, ax


def _flow_angle(x: float, y: float) -> float:
    noise_value = pnoise2(
        x * NOISE_SCALE,
        y * NOISE_SCALE,
        octaves=OCTAVES,
        persistence=0.5,
        lacunarity=2.0,
        repeatx=2048,
        repeaty=2048,
        base=SEED,
    )
    return (noise_value + 1.0) * np.pi * 2.2


def _reflect_to_unit_square(positions: np.ndarray) -> np.ndarray:
    positions = np.where(positions < 0.0, -positions, positions)
    positions = np.where(positions > 1.0, 2.0 - positions, positions)
    return np.clip(positions, 0.0, 1.0)


def generate_visual1() -> Path:
    rng = np.random.default_rng(SEED)
    positions = rng.uniform(0.12, 0.88, size=(PARTICLES, 2))
    trajectories = np.empty((STEPS + 1, PARTICLES, 2), dtype=float)
    trajectories[0] = positions
    diffusion_scale = np.sqrt(2.0 * DIFFUSION * DT)

    for step in range(1, STEPS + 1):
        for particle_index in range(PARTICLES):
            x, y = positions[particle_index]
            angle = _flow_angle(float(x), float(y))
            drift = DRIFT * np.array([np.cos(angle), np.sin(angle)])
            diffusion = diffusion_scale * rng.normal(size=2)
            positions[particle_index] = positions[particle_index] + drift + diffusion

        positions = _reflect_to_unit_square(positions)
        trajectories[step] = positions

    fig, ax = _prepare_axes()
    colors = plt.cm.magma(np.linspace(0.12, 0.95, PARTICLES))

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
        s=3,
        c=colors,
        alpha=0.45,
        linewidths=0,
    )

    output_path = _project_root() / "outputs" / "turbulent_memory.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_visual1()
