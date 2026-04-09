from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1920
HEIGHT = 1080
SEED = 45
RESOLUTION = 6000
BACKGROUND = "#020208"
SOURCES = np.array(
    [
        [-0.56, -0.10],
        [0.47, 0.26],
        [0.06, -0.56]
    ]
)
AMPLITUDES = np.array([1.0, 0.86, 0.74])
FREQUENCIES = np.array([28.0, 24.0, 31.0])
PHASES = np.array([0.0, 1.2, 2.4])
DECAYS = np.array([1.30, 1.12, 1.24])


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect(HEIGHT / WIDTH)
    ax.axis("off")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    return fig, ax


def generate_visual6() -> Path:
    grid = np.linspace(-1.08, 1.08, RESOLUTION)
    x_grid, y_grid = np.meshgrid(grid, grid)
    field = np.zeros_like(x_grid)

    for source, amplitude, frequency, phase, decay in zip(
        SOURCES,
        AMPLITUDES,
        FREQUENCIES,
        PHASES,
        DECAYS,
    ):
        distance = np.hypot(x_grid - source[0], y_grid - source[1]) + 1e-6
        field += amplitude * np.sin(frequency * distance - phase) * np.exp(-decay * distance)

    field += 0.18 * np.cos(9.5 * (x_grid + y_grid)) * np.exp(
        -0.85 * (x_grid * x_grid + y_grid * y_grid)
    )
    field *= np.exp(-0.42 * (x_grid * x_grid + y_grid * y_grid))
    glow = np.abs(field) ** 0.95

    fig, ax = _prepare_axes()
    ax.imshow(
        glow,
        extent=(-1.08, 1.08, -1.08, 1.08),
        origin="lower",
        cmap="magma",
        alpha=0.84,
    )

    contour_min = float(np.percentile(field, 6))
    contour_max = float(np.percentile(field, 94))
    levels = np.linspace(contour_min, contour_max, 18)
    ax.contour(
        x_grid,
        y_grid,
        field,
        levels=levels,
        cmap="viridis",
        linewidths=0.8,
        alpha=0.68,
    )
    ax.contour(
        x_grid,
        y_grid,
        field,
        levels=levels[::3],
        colors="#f8f1dc",
        linewidths=0.5,
        alpha=0.28,
    )
    ax.scatter(
        SOURCES[:, 0],
        SOURCES[:, 1],
        s=95,
        c="#fff1bf",
        alpha=0.72,
        linewidths=0,
    )

    output_path = _project_root() / "outputs" / "wavefront_interference.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_visual6()
