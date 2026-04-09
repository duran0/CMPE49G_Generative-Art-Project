from pathlib import Path

import matplotlib
import numpy as np
from noise import pnoise2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

WIDTH = 8192
HEIGHT = 4128
SEED = 42
GRID_SIZE = 29
NOISE_SCALE = 1.55
BACKGROUND = "#040308"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    return fig, ax


def _normalized_noise(x: float, y: float, base: int) -> float:
    return 0.5 * (
        pnoise2(
            x * NOISE_SCALE,
            y * NOISE_SCALE,
            octaves=4,
            persistence=0.55,
            lacunarity=2.1,
            repeatx=2048,
            repeaty=2048,
            base=base,
        )
        + 1.0
    )


def generate_visual4() -> Path:
    fig, ax = _prepare_axes()
    grid = np.linspace(-0.92, 0.92, GRID_SIZE)
    colormap = plt.cm.inferno

    for x in grid:
        for y in grid:
            radius_noise = _normalized_noise(x + 1.3, y - 0.4, SEED)
            angle_noise = _normalized_noise(x - 0.7, y + 1.2, SEED + 11)
            radius = 0.009 + 0.034 * radius_noise**1.4
            angle = np.pi * 2.0 * angle_noise
            color = colormap(0.18 + 0.75 * radius_noise)
            alpha = 0.2 + 0.45 * radius_noise
            dx = np.cos(angle) * radius * 1.75
            dy = np.sin(angle) * radius * 1.75

            ax.add_patch(
                Circle(
                    (x, y),
                    radius=radius,
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                )
            )
            ax.plot(
                [x - dx, x + dx],
                [y - dy, y + dy],
                color=color,
                linewidth=0.6 + 1.2 * radius_noise,
                alpha=0.45 + 0.35 * radius_noise,
                solid_capstyle="round",
            )

    output_path = _project_root() / "outputs" / "noise_geometry.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_visual4()
