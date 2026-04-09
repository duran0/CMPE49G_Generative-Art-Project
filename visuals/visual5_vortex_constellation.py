from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1920
HEIGHT = 1080
SEED = 77
PARTICLES = 350
STEPS = 600
DT = 1.0
DIFFUSION = 0.0028
CORE_RADIUS = 0.105
CONFINEMENT = 0.0035
BACKGROUND = "#01060a"
LINE_WIDTH = 0.35
LINE_ALPHA = 0.15
VORTEX_CENTERS = np.array(
    [
        [-0.58, -0.32],
        [0.52, -0.36],
        [-0.16, 0.47],
        [0.44, 0.39]     
    ]
)
VORTEX_STRENGTHS = np.array([0.100, -0.099, 0.175, -0.105])


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_aspect(HEIGHT / WIDTH)
    ax.axis("off")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    return fig, ax


def _reflect_to_bounds(positions: np.ndarray) -> np.ndarray:
    positions = np.where(positions < -1.0, -2.0 - positions, positions)
    positions = np.where(positions > 1.0, 2.0 - positions, positions)
    return np.clip(positions, -1.0, 1.0)


def generate_visual5() -> Path:
    rng = np.random.default_rng(SEED)
    positions = rng.uniform(-0.92, 0.92, size=(PARTICLES, 2))
    initial_positions = positions.copy()
    trajectories = np.empty((STEPS + 1, PARTICLES, 2), dtype=float)
    trajectories[0] = positions
    diffusion_scale = np.sqrt(2.0 * DIFFUSION * DT)

    for step in range(1, STEPS + 1):
        velocity = np.zeros_like(positions)

        for center, strength in zip(VORTEX_CENTERS, VORTEX_STRENGTHS):
            delta = positions - center
            dist2 = np.sum(delta * delta, axis=1, keepdims=True) + CORE_RADIUS**2
            tangential = np.column_stack((-delta[:, 1], delta[:, 0]))
            velocity += strength * tangential / dist2

        velocity += -CONFINEMENT * positions
        velocity += 0.0022 * np.column_stack(
            (
                np.sin(np.pi * 3.5 * positions[:, 1]),
                np.cos(np.pi * 3.0 * positions[:, 0]),
            )
        )
        positions = positions + velocity * DT + diffusion_scale * rng.normal(size=positions.shape)
        positions = _reflect_to_bounds(positions)
        trajectories[step] = positions

    color_values = (np.arctan2(initial_positions[:, 1], initial_positions[:, 0]) + np.pi) / (
        2.0 * np.pi
    )
    colors = plt.cm.twilight_shifted(0.08 + 0.84 * color_values)

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
        VORTEX_CENTERS[:, 0],
        VORTEX_CENTERS[:, 1],
        s=60,
        c="#fff2cf",
        alpha=0.8,
        linewidths=0,
    )

    output_path = _project_root() / "outputs" / "vortex_constellation.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    generate_visual5()
