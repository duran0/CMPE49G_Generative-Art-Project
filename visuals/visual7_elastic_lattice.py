from pathlib import Path

import matplotlib
import numpy as np
from noise import pnoise2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 8192
HEIGHT = 4128
SEED = 42
LINE_COUNT = 27
POINTS_PER_LINE = 180
NOISE_SCALE = 1.85
NOISE_DISPLACEMENT = 0.035
BACKGROUND = "#03050a"
FIELD_CENTERS = np.array(
    [
        [-0.48, 0.34],
        [0.38, -0.40],
        [0.22, 0.50],
    ]
)
FIELD_STRENGTHS = np.array([0.017, -0.015, 0.012])


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


def _noise_displacement(x: float, y: float) -> np.ndarray:
    noise_value = pnoise2(
        (x + 1.7) * NOISE_SCALE,
        (y - 0.9) * NOISE_SCALE,
        octaves=3,
        persistence=0.55,
        lacunarity=2.0,
        repeatx=2048,
        repeaty=2048,
        base=SEED,
    )
    angle = np.pi * (noise_value + 1.0) * 2.0
    return NOISE_DISPLACEMENT * np.array([np.cos(angle), np.sin(angle)])


def _warp_points(points: np.ndarray) -> np.ndarray:
    displacement = np.zeros_like(points)

    for center, strength in zip(FIELD_CENTERS, FIELD_STRENGTHS):
        delta = points - center
        dist2 = np.sum(delta * delta, axis=1, keepdims=True) + 0.045
        displacement += strength * delta / dist2

    curl = np.column_stack(
        (
            -points[:, 1] * np.sin(np.pi * points[:, 0]),
            points[:, 0] * np.cos(np.pi * points[:, 1]),
        )
    )
    displacement += 0.03 * curl

    noise = np.array([_noise_displacement(float(x), float(y)) for x, y in points])
    warped = points + displacement + noise
    return np.clip(warped, -1.04, 1.04)


def generate_visual7() -> Path:
    fig, ax = _prepare_axes()
    base_values = np.linspace(-0.94, 0.94, LINE_COUNT)
    samples = np.linspace(-0.96, 0.96, POINTS_PER_LINE)

    for index, y_value in enumerate(base_values):
        points = np.column_stack((samples, np.full_like(samples, y_value)))
        warped = _warp_points(points)
        color = plt.cm.copper(0.18 + 0.72 * index / (LINE_COUNT - 1))
        ax.plot(
            warped[:, 0],
            warped[:, 1],
            color=color,
            linewidth=0.9,
            alpha=0.63,
        )

    for index, x_value in enumerate(base_values):
        points = np.column_stack((np.full_like(samples, x_value), samples))
        warped = _warp_points(points)
        color = plt.cm.winter(0.2 + 0.72 * index / (LINE_COUNT - 1))
        ax.plot(
            warped[:, 0],
            warped[:, 1],
            color=color,
            linewidth=0.85,
            alpha=0.5,
        )

    ax.scatter(
        FIELD_CENTERS[:, 0],
        FIELD_CENTERS[:, 1],
        s=38,
        c="#f7f1d8",
        alpha=0.75,
        linewidths=0,
    )

    output_path = _project_root() / "outputs" / "elastic_lattice.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_visual7()
