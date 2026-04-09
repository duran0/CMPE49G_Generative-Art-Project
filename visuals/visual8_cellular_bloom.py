from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1920
HEIGHT = 1080
SEED = 48
RESOLUTION = 2000
SEED_COUNT = 48
BACKGROUND = "#040208"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect(HEIGHT / WIDTH)
    ax.axis("off")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    return fig, ax


def generate_visual8() -> Path:
    rng = np.random.default_rng(SEED)
    grid = np.linspace(-1.0, 1.0, RESOLUTION)
    x_grid, y_grid = np.meshgrid(grid, grid)
    seeds = rng.uniform(-0.82, 0.82, size=(SEED_COUNT, 2))
    amplitudes = rng.uniform(0.75, 1.3, size=SEED_COUNT)
    sigmas = rng.uniform(0.08, 0.15, size=SEED_COUNT)

    nearest = np.full_like(x_grid, np.inf)
    second_nearest = np.full_like(x_grid, np.inf)
    glow = np.zeros_like(x_grid)

    for seed, amplitude, sigma in zip(seeds, amplitudes, sigmas):
        distance = np.hypot(x_grid - seed[0], y_grid - seed[1])

        replace_nearest = distance < nearest
        previous_nearest = nearest.copy()
        nearest = np.where(replace_nearest, distance, nearest)
        second_nearest = np.where(
            replace_nearest,
            previous_nearest,
            np.minimum(second_nearest, distance),
        )

        glow += amplitude * np.exp(-(distance * distance) / (2.0 * sigma * sigma))

    membranes = np.exp(-18.0 * (second_nearest - nearest))
    membranes = gaussian_filter(membranes, sigma=1.2)
    glow = gaussian_filter(glow, sigma=1.5)

    texture = gaussian_filter(rng.random((RESOLUTION, RESOLUTION)), sigma=16.0)
    texture = (texture - texture.mean()) / texture.std()
    radial_falloff = np.exp(-0.45 * (x_grid * x_grid + y_grid * y_grid))

    field = (0.84 * glow + 0.58 * membranes + 0.08 * texture) * radial_falloff
    field = gaussian_filter(field, sigma=1.0)

    fig, ax = _prepare_axes()
    field_min = float(np.percentile(field, 34))
    field_max = float(np.percentile(field, 99.2))
    levels = np.linspace(field_min, field_max, 16)

    ax.contourf(
        x_grid,
        y_grid,
        field,
        levels=levels,
        cmap="inferno",
        alpha=0.95,
    )
    ax.contour(
        x_grid,
        y_grid,
        field,
        levels=levels,
        colors="#fdecc7",
        linewidths=0.42,
        alpha=0.34,
    )
    ax.scatter(
        seeds[:, 0],
        seeds[:, 1],
        s=18 + 34 * amplitudes,
        c="#fff4cc",
        alpha=0.22,
        linewidths=0,
    )

    output_path = _project_root() / "outputs" / "cellular_bloom.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_visual8()
