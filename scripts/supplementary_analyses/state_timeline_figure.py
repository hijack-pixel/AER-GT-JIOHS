"""Reference-state timeline of the training recordings.

Draws one row per animal and one block per recording at its post-injection time,
colored by the reference state of the recording (baseline, seizure, or recovery),
with the per-animal window counts. The vertical probes mark post-injection times
at which the state composition is annotated (default 30 and 60 min).

The schedule is built either from the labeled window dataset of a run
configuration together with a table of recording times, or from a schedule CSV.

Usage:
    python scripts/supplementary_analyses/state_timeline_figure.py \
        --config configs/full_model.yaml --recording-times recording_times.csv

    python scripts/supplementary_analyses/state_timeline_figure.py --schedule recording_schedule.csv

Inputs:
    --recording-times CSV   columns mouse_id, recording, t_post_inj_min (one row per
                            recording; ``recording`` is the OriginalFile field of
                            the window samples)
    --schedule CSV          columns animal, t_post_inj_min, label, n_windows

Output:
    outputs/supplementary_analyses/state_timeline/
        recording_schedule.csv   the schedule that was drawn
        timeline_states.png / .pdf
        summary.json              recording and window counts per state
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from _common import STATE_NAMES, ROOT, display_map, load_run_config, read_recording_times, resolve_path  # noqa: E402

FILL = {"Baseline": "#b8d8b8", "Seizure": "#f2b8b8", "Recovery": "#f5e6b8"}
EDGE = {"Baseline": "#548235", "Seizure": "#C00000", "Recovery": "#E6A817"}


def schedule_from_dataset(cfg: dict, times: pd.DataFrame) -> pd.DataFrame:
    from aergt.data.samples import load_samples

    samples = load_samples(resolve_path(cfg["data"]["labeled_dir"]), labeled=True,
                           num_nodes=int(cfg["model"]["num_nodes"]), in_channels=int(cfg["model"]["in_channels"]),
                           use_abs_adj=bool(cfg["data"].get("use_abs_adjacency", True)))
    counts = Counter()
    for sample in samples:
        recording = sample.original_file or Path(sample.source_file).stem
        counts[(sample.mouse_id, recording, int(sample.label))] += 1
    rows = []
    for (mouse, recording, label), n_windows in sorted(counts.items()):
        rows.append({"mouse_id": mouse, "recording": recording, "label": STATE_NAMES[label], "n_windows": n_windows})
    schedule = pd.DataFrame(rows).merge(times, on=["mouse_id", "recording"], how="left", validate="one_to_one")
    if schedule["t_post_inj_min"].isna().any():
        missing = schedule.loc[schedule["t_post_inj_min"].isna(), ["mouse_id", "recording"]].values.tolist()
        raise SystemExit(f"no post-injection time for recordings: {missing}")
    return schedule


def schedule_from_csv(path: str) -> pd.DataFrame:
    schedule = pd.read_csv(resolve_path(path))
    required = {"animal", "t_post_inj_min", "label", "n_windows"}
    if not required.issubset(schedule.columns):
        raise SystemExit(f"schedule CSV needs the columns {sorted(required)}")
    schedule = schedule.rename(columns={"animal": "mouse_id"})
    schedule["label"] = schedule["label"].str.strip().str.capitalize()
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/full_model.yaml")
    parser.add_argument("--recording-times", help="CSV with mouse_id, recording, t_post_inj_min")
    parser.add_argument("--schedule", help="CSV with animal, t_post_inj_min, label, n_windows")
    parser.add_argument("--probes", type=float, nargs="*", default=[30, 60], help="annotated post-injection times (min)")
    parser.add_argument("--block-width", type=float, default=5.0, help="visual width of a recording block (min)")
    parser.add_argument("--out", default="outputs/supplementary_analyses/state_timeline")
    args = parser.parse_args()

    if args.schedule:
        schedule = schedule_from_csv(args.schedule)
    elif args.recording_times:
        schedule = schedule_from_dataset(load_run_config(args.config), read_recording_times(args.recording_times))
    else:
        parser.error("pass --schedule or (--config and) --recording-times")

    names = display_map(schedule["mouse_id"])
    schedule = schedule.assign(animal=schedule["mouse_id"].astype(str).map(names)).sort_values(["animal", "t_post_inj_min"])
    animals = sorted(set(names.values()), key=lambda label: int(label.split()[1]))
    unknown = set(schedule["label"]) - set(STATE_NAMES)
    if unknown:
        raise SystemExit(f"unknown labels in the schedule: {sorted(unknown)}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    schedule[["animal", "t_post_inj_min", "label", "n_windows"]].to_csv(out / "recording_schedule.csv", index=False)
    per_state = schedule.groupby("label")["n_windows"].agg(["count", "sum"])
    summary = {
        "n_recordings": int(len(schedule)),
        "n_windows": int(schedule["n_windows"].sum()),
        "recordings_per_state": {state: int(per_state.loc[state, "count"]) for state in STATE_NAMES if state in per_state.index},
        "windows_per_state": {state: int(per_state.loc[state, "sum"]) for state in STATE_NAMES if state in per_state.index},
        "windows_per_animal": {animal: int(group["n_windows"].sum()) for animal, group in schedule.groupby("animal")},
        "probes": {},
    }
    for probe in args.probes:
        covering = schedule[np.abs(schedule["t_post_inj_min"] - probe) <= args.block_width / 2]
        composition = Counter(covering["label"])
        summary["probes"][f"{probe:g}"] = {state: int(composition.get(state, 0)) for state in STATE_NAMES}

    # ---------- figure ----------
    height = 1.75 + 0.62 * len(animals)
    fig, ax = plt.subplots(figsize=(10.0, height))
    block_height = 0.52
    latest = float(schedule["t_post_inj_min"].max())
    span = max(latest, max(args.probes) if args.probes else latest, 60.0)
    half = args.block_width / 2
    for row, animal in enumerate(animals):
        y = len(animals) - 1 - row
        group = schedule[schedule["animal"] == animal]
        ax.plot([group["t_post_inj_min"].min() - half, group["t_post_inj_min"].max() + half], [y, y],
                color="#c9c9c9", lw=1.1, zorder=1)
        for _, recording in group.iterrows():
            t = float(recording["t_post_inj_min"])
            ax.add_patch(FancyBboxPatch((t - half, y - block_height / 2), args.block_width, block_height,
                                        boxstyle="round,pad=0,rounding_size=1.2",
                                        facecolor=FILL[recording["label"]], edgecolor=EDGE[recording["label"]],
                                        lw=1.3, zorder=3))
        ax.text(-span * 0.03, y, animal, ha="right", va="center", fontsize=10.5)
        ax.text(span * 1.03, y, f"{int(group['n_windows'].sum())} windows", ha="left", va="center", fontsize=8.5, color="#666666")

    for index, probe in enumerate(args.probes):
        composition = summary["probes"][f"{probe:g}"]
        text = f"{probe:g} min:\n" + ", ".join(f"{composition[state]} {state.lower()}" for state in STATE_NAMES)
        ax.axvline(probe, color="#555555", ls="--", lw=1.2, zorder=2)
        ax.text(probe, len(animals) - 1 + 0.55 + 1.05 * (index % 2), text, ha="center", va="bottom", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#555555", lw=0.9))

    top_row = len(animals) - 1
    injection_y = top_row + 0.55 + 1.05 * min(len(args.probes), 2)
    ax.annotate("Pilocarpine\ninjection", xy=(0, top_row + 0.05), xytext=(0, injection_y),
                ha="center", va="bottom", fontsize=9.5, arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3))
    ticks = np.arange(0, span + 1, 30 if span > 90 else 15)
    ax.set_xlim(-span * 0.14, span * 1.2)
    ax.set_ylim(-0.75, injection_y + 0.95)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.set_xlabel("Time after injection (min)", fontsize=10.5)
    ax.tick_params(labelsize=9.5)
    ax.legend(handles=[Patch(facecolor=FILL[state], edgecolor=EDGE[state], lw=1.2, label=state) for state in STATE_NAMES],
              loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / "timeline_states.png", dpi=300)
    fig.savefig(out / "timeline_states.pdf")

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(schedule[["animal", "t_post_inj_min", "label", "n_windows"]].to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
