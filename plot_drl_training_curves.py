"Plot per-episode training curves for the four learned scheduling methods."

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
import numpy as np


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Helvetica", "DejaVu Sans", "sans-serif",
    ],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.0,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


@dataclass(frozen=True)
class MethodStyle:
    directory: str
    label: str
    method_id: str
    color: str
    linestyle: str


METHODS = (
    MethodStyle(
        "ours", "Heterogeneous SMDP-MAPPO",
        "heterogeneous_smdp_mappo", "#0072B2", "-",
    ),
    MethodStyle(
        "ppo", "Centralized PPO", "centralized_ppo", "#E69F00", "-",
    ),
    MethodStyle("qmix", "QMIX", "qmix", "#D55E00", "-"),
    MethodStyle("mappo", "MAPPO", "mappo", "#009E73", "-"),
)
SEEDS = (101, 202, 303, 404, 505)
EXPECTED_EPISODES = 500
DEFAULT_SMOOTHING_WINDOW = 51
REQUIRED_CONDITION_FIELDS = (
    "nominal_finite_load",
    "initial_jobs_configured",
    "total_jobs_configured",
    "due_date_factor",
    "degradation_multiplier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("runs/e1_learned_method_training"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW,
        help="Odd window width for the centered moving mean (default: 25).",
    )
    return parser.parse_args()


def read_episode_log(
        root: Path, method: MethodStyle, seed: int) -> dict[str, np.ndarray]:
    path = root / method.directory / f"seed-{seed}" / "Training_Episodes.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode-level training log: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_EPISODES:
        raise ValueError(
            f"Expected {EXPECTED_EPISODES} training episodes but found "
            f"{len(rows)}: {path}"
        )

    episodes = np.asarray([int(row["episode"]) for row in rows], dtype=int)
    tardiness = np.asarray(
        [float(row["mean_tardiness"]) for row in rows], dtype=float,
    )
    expected = np.arange(1, EXPECTED_EPISODES + 1, dtype=int)
    if not np.array_equal(episodes, expected):
        raise ValueError(f"Training episodes are incomplete or unordered: {path}")
    if not np.all(np.isfinite(tardiness)) or np.any(tardiness <= 0.0):
        raise ValueError(
            f"Log-scale curves require finite positive mean tardiness: {path}"
        )
    if {row["method_id"] for row in rows} != {method.method_id}:
        raise ValueError(f"Unexpected canonical method identifier: {path}")
    if {int(row["training_seed"]) for row in rows} != {seed}:
        raise ValueError(f"Unexpected training seed: {path}")

    condition = tuple(rows[0][field] for field in REQUIRED_CONDITION_FIELDS)
    if any(
            tuple(row[field] for field in REQUIRED_CONDITION_FIELDS) != condition
            for row in rows[1:]):
        raise ValueError(f"Training condition changes within log: {path}")
    return {
        "episodes": episodes,
        "tardiness": tardiness,
        "condition": np.asarray(condition, dtype=str),
    }


def centered_moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered moving mean with unbiased boundary denominators."""

    if window < 3 or window % 2 == 0:
        raise ValueError("The smoothing window must be an odd integer >= 3.")
    if window > len(values):
        raise ValueError("The smoothing window cannot exceed the episode count.")
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="constant")
    available = np.pad(
        np.ones_like(values, dtype=float), (radius, radius), mode="constant",
    )
    kernel = np.ones(window, dtype=float)
    totals = np.convolve(padded, kernel, mode="valid")
    counts = np.convolve(available, kernel, mode="valid")
    return totals / counts


def load_all(root: Path) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    data: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    reference_condition = None
    for seed in SEEDS:
        data[seed] = {}
        for method in METHODS:
            series = read_episode_log(root, method, seed)
            if reference_condition is None:
                reference_condition = series["condition"]
            data[seed][method.directory] = series
    return data


def log_tick_label(value: float, _position: int) -> str:
    if value >= 1.0:
        return f"{value:g}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def plot_seed(
        seed: int,
        series_by_method: dict[str, dict[str, np.ndarray]],
        output_dir: Path,
        y_limits: tuple[float, float],
        smoothing_window: int) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=False)

    for method in METHODS:
        series = series_by_method[method.directory]
        episodes, values = series["episodes"], series["tardiness"]
        smoothed = centered_moving_mean(values, smoothing_window)

        # The complete observed trajectory remains visible as the source layer.
        ax.plot(
            episodes, values,
            color=method.color,
            linewidth=0.55,
            alpha=0.14,
            solid_capstyle="round",
            rasterized=False,
            zorder=1,
        )
        # A darker curve exposes the convergence tendency without hiding noise.
        ax.plot(
            episodes, smoothed,
            color=method.color,
            linestyle=method.linestyle,
            linewidth=1.5,
            alpha=0.98,
            solid_capstyle="round",
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlim(1, EXPECTED_EPISODES)
    ax.set_ylim(*y_limits)
    ax.set_xticks(np.arange(0, EXPECTED_EPISODES + 1, 100))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(log_tick_label))
    ax.yaxis.set_minor_locator(LogLocator(
        base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0),
    ))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="major", length=3.2, width=0.7)
    ax.tick_params(which="minor", length=1.8, width=0.5)
    ax.grid(axis="y", which="major", color="#D8D8D8", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Training mean tardiness (log scale)")
    ax.set_title(f"Training seed {seed}", fontsize=9.5, fontweight="bold", pad=8)

    method_handles = [
        Line2D(
            [0], [0], color=method.color, linestyle=method.linestyle,
            linewidth=1.5, label=method.label,
        )
        for method in METHODS
    ]
    ax.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
        columnspacing=1.8,
        handlelength=3.0,
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.82, bottom=0.17)

    output = output_dir / f"drl_training_curve_seed_{seed}.pdf"
    fig.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": f"Per-episode DRL training curves for seed {seed}",
            "Subject": (
                "Observed and smoothed training mean tardiness over 500 episodes"
            ),
        },
    )
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    if args.smoothing_window < 3 or args.smoothing_window % 2 == 0:
        raise ValueError("--smoothing-window must be an odd integer >= 3.")
    root = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(root)
    all_values = np.concatenate([
        data[seed][method.directory]["tardiness"]
        for seed in SEEDS for method in METHODS
    ])
    y_limits = (
        float(all_values.min() / 1.15),
        float(all_values.max() * 1.15),
    )
    outputs = [
        plot_seed(
            seed, data[seed], output_dir, y_limits, args.smoothing_window,
        )
        for seed in SEEDS
    ]
    print(
        f"Generated {len(outputs)} per-episode training-curve PDFs "
        f"with a {args.smoothing_window}-episode moving mean:",
        flush=True,
    )
    for output in outputs:
        print(output, flush=True)


if __name__ == "__main__":
    main()
