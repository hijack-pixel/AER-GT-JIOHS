"""Recording-position sensitivity of the Fig. 4 statistics (block analysis).

The state characterization of the manuscript (Sec. 2.3) uses one 20-s analysis
segment per recording: the 200 frames that follow the 100-frame normalization
reference. This script tests whether the two statistics of Fig. 4 - the state
effect on mean signal amplitude and the state effect on network deviation -
depend on which part of a recording is analysed.

On the window-level dataset the analysed unit is a block of consecutive
analysis windows (five by default): every window keeps the features computed
from its own 200 frames, and the block statistic is their mean, so that a
block of five windows spans 36 s of recording (200 + 4 x 40 frames). A block of
five windows has 29 possible start positions in a 33-window recording; with
``--block-windows 1`` single 200-frame windows (33 positions) are evaluated
instead. Three computations are run on the recordings of the labeled dataset:

(a) full-recording aggregation (all windows of each recording);
(b) randomly positioned blocks (``--draws`` draws);
(c) a systematic sweep over every block start position.

Per animal and state, the amplitude is the mean of the first node feature (the
within-window mean of the normalized ROI signal) and the network deviation is
1 - r between the upper-triangle profiles of the mean correlation matrix of the
block and the animal's baseline template, built from the baseline recordings of
the same animal at the same block position. The state effect of each statistic
is tested with a Friedman test over animals; Kendall's W is reported.

Usage:
    python scripts/supplementary_analyses/segment_position_sensitivity.py \
        --config configs/full_model.yaml --draws 1000

Input: labeled window samples of the run configuration (``data.labeled_dir``).

Output:
    outputs/supplementary_analyses/segment_position/
        position_sweep.csv      block start, Friedman p of both statistics, W, significance-pattern flag
        random_draws.csv        the same for every random draw
        summary.json
        sensitivity_figure.png / .pdf
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from _common import STATE_NAMES, ROOT, display_map, load_run_config, mean_sd, resolve_path  # noqa: E402


def recording_key(sample) -> str:
    """Recording identifier of a window: its ``OriginalFile`` field, or the file stem."""
    return sample.original_file or sample.source_file.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[0]


def load_recordings(cfg: dict) -> dict:
    """Arrays per recording: label, amplitude per window, upper-triangle correlation per window."""
    from aergt.data.samples import load_samples

    num_nodes, in_channels = int(cfg["model"]["num_nodes"]), int(cfg["model"]["in_channels"])
    samples = load_samples(resolve_path(cfg["data"]["labeled_dir"]), labeled=True,
                           num_nodes=num_nodes, in_channels=in_channels, use_abs_adj=False)
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample.mouse_id, recording_key(sample))].append(sample)
    upper = np.triu_indices(num_nodes, k=1)
    recordings = {}
    for key, group in groups.items():
        group.sort(key=lambda s: s.window_index)
        labels = {s.label for s in group}
        if len(labels) != 1:
            raise SystemExit(f"recording {key} carries more than one label: {sorted(labels)}")
        recordings[key] = {
            "label": int(group[0].label),
            "amplitude": np.array([float(np.mean(s.x[:, 0])) for s in group]),
            "upper": np.vstack([s.adj[upper] for s in group]),
        }
    return recordings


def block_values(recording: dict, start: int, block: int, full: bool) -> tuple[float, np.ndarray]:
    if full:
        return float(recording["amplitude"].mean()), recording["upper"].mean(axis=0)
    window = slice(start, start + block)
    return float(recording["amplitude"][window].mean()), recording["upper"][window].mean(axis=0)


def animal_state_table(recordings: dict, animals: list[str], start: int | None, block: int) -> dict:
    """Amplitude and deviation per (animal, state) for one block position (or full recordings)."""
    full = start is None
    table = {}
    for animal in animals:
        keys = [key for key in recordings if key[0] == animal]
        baseline = [recordings[key] for key in keys if recordings[key]["label"] == 0]
        if not baseline:
            raise SystemExit(f"animal {animal} has no baseline recording")
        template = np.mean([block_values(rec, start, block, full)[1] for rec in baseline], axis=0)
        for label, state in enumerate(STATE_NAMES):
            group = [recordings[key] for key in keys if recordings[key]["label"] == label]
            if not group:
                continue
            values = [block_values(rec, start, block, full) for rec in group]
            deviations = [1.0 - float(np.corrcoef(profile, template)[0, 1]) for _, profile in values]
            table[(animal, state)] = (float(np.mean([amplitude for amplitude, _ in values])), float(np.mean(deviations)))
    return table


def state_effects(table: dict, animals: list[str]) -> dict:
    """Friedman tests of the state effect on amplitude and deviation, and the flag for the Fig. 4
    significance pattern (amplitude non-significant, deviation significant)."""
    amplitude = np.array([[table[(animal, state)][0] for state in STATE_NAMES] for animal in animals])
    deviation = np.array([[table[(animal, state)][1] for state in STATE_NAMES] for animal in animals])
    chi_amplitude, p_amplitude = stats.friedmanchisquare(*[amplitude[:, index] for index in range(3)])
    chi_deviation, p_deviation = stats.friedmanchisquare(*[deviation[:, index] for index in range(3)])
    kendall = chi_deviation / (len(animals) * 2.0)
    return {"p_amplitude": float(p_amplitude), "p_deviation": float(p_deviation),
            "W_deviation": float(kendall),
            "same_pattern": bool(p_amplitude >= 0.05 and p_deviation < 0.05)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml")
    parser.add_argument("--block-windows", type=int, default=5, help="windows per 200-frame block")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/supplementary_analyses/segment_position")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    recordings = load_recordings(cfg)
    names = display_map(key[0] for key in recordings)
    animals = sorted(names, key=lambda identifier: int(names[identifier].split()[1]))   # dataset ids in display order
    shortest = min(len(recording["amplitude"]) for recording in recordings.values())
    positions = list(range(shortest - args.block_windows + 1))
    if not positions:
        raise SystemExit("recordings are shorter than one block")

    full_table = animal_state_table(recordings, animals, None, args.block_windows)
    full = state_effects(full_table, animals)

    rng = np.random.default_rng(args.seed)
    draws = []
    for draw in range(args.draws):
        start = int(rng.integers(0, len(positions)))
        draws.append({"draw": draw + 1, "block_start": start + 1, **state_effects(
            animal_state_table(recordings, animals, start, args.block_windows), animals)})
    draws = pd.DataFrame(draws)
    draws["same_pattern"] = draws["same_pattern"].astype(int)

    sweep = pd.DataFrame([{"block_start": start + 1, "start_frame": start * 40 + 1, **state_effects(
        animal_state_table(recordings, animals, start, args.block_windows), animals)} for start in positions])
    sweep["same_pattern"] = sweep["same_pattern"].astype(int)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "position_sweep.csv", index=False)
    draws.to_csv(out / "random_draws.csv", index=False)
    summary = {
        "n_recordings": len(recordings),
        "n_animals": len(animals),
        "windows_per_recording_min": int(shortest),
        "block_windows": args.block_windows,
        "n_positions": len(positions),
        "full_recording_aggregation": full,
        "random_blocks": {"draws": args.draws,
                          "same_pattern_rate": float(draws["same_pattern"].mean()),
                          "p_amplitude_median": float(draws["p_amplitude"].median()),
                          "p_amplitude_share_nonsignificant": float((draws["p_amplitude"] >= 0.05).mean()),
                          "p_deviation_median": float(draws["p_deviation"].median()),
                          "W_deviation_median": float(draws["W_deviation"].median()),
                          "p_deviation_share_significant": float((draws["p_deviation"] < 0.05).mean())},
        "position_sweep": {"same_pattern_rate": float(sweep["same_pattern"].mean()),
                           "p_amplitude_range": [float(sweep["p_amplitude"].min()), float(sweep["p_amplitude"].max())],
                           "p_deviation_range": [float(sweep["p_deviation"].min()), float(sweep["p_deviation"].max())]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(sweep["block_start"], sweep["p_amplitude"], "o-", color="#2d5a1e", label="amplitude")
    axes[0].plot(sweep["block_start"], sweep["p_deviation"], "s-", color="#C00000", label="network deviation")
    axes[0].axhline(0.05, color="grey", ls="--", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Block start (window index)")
    axes[0].set_ylabel("Friedman p")
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Block-position sweep", fontsize=11)
    axes[1].hist(draws["p_amplitude"], bins=40, alpha=0.6, color="#2d5a1e", label="amplitude")
    axes[1].hist(draws["p_deviation"], bins=40, alpha=0.6, color="#C00000", label="network deviation")
    axes[1].axvline(0.05, color="grey", ls="--", lw=0.8)
    axes[1].set_xlabel("Friedman p (random block)")
    axes[1].set_ylabel("draws")
    axes[1].legend(fontsize=9, frameon=False)
    axes[1].grid(alpha=0.25)
    axes[1].set_title(f"Randomly positioned blocks (n = {args.draws})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_figure.png", dpi=300)
    fig.savefig(out / "sensitivity_figure.pdf")

    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
