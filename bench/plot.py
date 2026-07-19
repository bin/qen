#!/usr/bin/env python3
"""Parse rpmalloc-benchmark output (results/raw.log) into results.csv and
render the result figures.

Chart styling follows the repo's dataviz conventions: validated categorical
palette (fixed slot order — color follows the allocator, never its rank),
recessive grid, thin marks, selective direct labels, one axis per chart.
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# --- palette (validated: scripts/validate_palette.js, light surface) -------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8984"
GRID = "#e4e3df"

# Fixed slot order — qen is the subject (slot 1); crt is the system baseline
# and wears neutral gray (reference series), keeping six colored slots.
SERIES = {
    "qen":      {"color": "#2a78d6", "z": 10, "lw": 2.6},
    "mimalloc": {"color": "#1baf7a", "z": 5, "lw": 1.8},
    "snmalloc": {"color": "#eda100", "z": 5, "lw": 1.8},
    "tcmalloc": {"color": "#008300", "z": 5, "lw": 1.8},
    "jemalloc": {"color": "#4a3aa7", "z": 5, "lw": 1.8},
    "rpmalloc": {"color": "#e34948", "z": 5, "lw": 1.8},
    "crt":      {"color": "#8a8984", "z": 4, "lw": 1.8},
}
ORDER = list(SERIES)

LINE_RE = re.compile(
    r"^(?P<alloc>[\w-]+)\s+(?P<threads>\d+) threads random (?P<dist>\w+) size "
    r"\[(?P<min>\d+),(?P<max>\d+)\] (?P<loops>\d+) loops (?P<allocs>\d+) allocs "
    r"(?P<ops>\d+) ops: \.*\s*(?P<mops>\d+) memory ops/CPU second \(peak (?P<peak>\d+)MiB\)"
)


def parse(raw: Path):
    rows = []
    for line in raw.read_text().splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        scenario = f"{d['dist']} [{d['min']},{d['max']}]"
        rows.append(
            {
                "allocator": d["alloc"],
                "scenario": scenario,
                "dist": d["dist"],
                "min_size": int(d["min"]),
                "max_size": int(d["max"]),
                "threads": int(d["threads"]),
                "mops_per_cpu_second": int(d["mops"]),
                "peak_mib": int(d["peak"]),
            }
        )
    return rows


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def plot_throughput(rows):
    scenarios = sorted({(r["scenario"], r["min_size"], r["max_size"]) for r in rows},
                       key=lambda t: (t[2], t[1]))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), facecolor=SURFACE)
    axes = axes.ravel()

    for i, (scenario, _mn, _mx) in enumerate(scenarios):
        ax = axes[i]
        style_axes(ax)
        for alloc in ORDER:
            pts = sorted(
                (r["threads"], r["mops_per_cpu_second"] / 1e6)
                for r in rows
                if r["allocator"] == alloc and r["scenario"] == scenario
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            s = SERIES[alloc]
            ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=s["lw"],
                    color=s["color"], zorder=s["z"], label=alloc,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
            # Selective direct labels (relief for low-contrast slots):
            # subject + baseline only; the legend carries the rest.
            if alloc in ("qen", "crt"):
                ax.annotate(alloc, (xs[-1], ys[-1]), xytext=(5, 0),
                            textcoords="offset points", fontsize=8,
                            color=s["color"], va="center", fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.set_xticklabels(["1", "2", "4", "8", "16"])
        ax.set_title(f"sizes {scenario}", fontsize=9.5, color=TEXT_PRIMARY, pad=6)
        ax.set_ylim(bottom=0)
        if i % 3 == 0:
            ax.set_ylabel("M ops / CPU-second", fontsize=8.5, color=TEXT_SECONDARY)
        if i >= 2:
            ax.set_xlabel("threads", fontsize=8.5, color=TEXT_SECONDARY)

    # Legend panel in the unused 6th slot.
    ax = axes[len(scenarios)]
    ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(handles, labels, loc="center", frameon=False, fontsize=9.5,
              labelcolor=TEXT_PRIMARY, title="allocator",
              title_fontsize=10)

    fig.suptitle("Allocator throughput — rpmalloc-benchmark, random mixed alloc/free "
                 "with cross-thread frees (higher is better)",
                 fontsize=12, color=TEXT_PRIMARY, y=0.985)
    fig.text(0.01, 0.005,
             "Apple M-series (16 cores), macOS · rpmalloc-benchmark runall scenarios, loops ÷4 · "
             "qen includes free(ptr) size-recovery shim (see RESULTS.md)",
             fontsize=7.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = RESULTS / "throughput.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


def plot_memory(rows):
    scenarios = sorted({(r["scenario"], r["min_size"], r["max_size"]) for r in rows},
                       key=lambda t: (t[2], t[1]))
    threads_shown = 16
    fig, ax = plt.subplots(figsize=(12.5, 4.6), facecolor=SURFACE)
    style_axes(ax)

    n_alloc = len(ORDER)
    group_w = 0.86
    bar_w = group_w / n_alloc
    for j, alloc in enumerate(ORDER):
        xs, ys = [], []
        for i, (scenario, _mn, _mx) in enumerate(scenarios):
            for r in rows:
                if (r["allocator"] == alloc and r["scenario"] == scenario
                        and r["threads"] == threads_shown):
                    xs.append(i - group_w / 2 + bar_w * (j + 0.5))
                    ys.append(r["peak_mib"])
        ax.bar(xs, ys, width=bar_w * 0.92, color=SERIES[alloc]["color"],
               label=alloc, edgecolor=SURFACE, linewidth=1.0,
               zorder=SERIES[alloc]["z"])

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([f"sizes {s}" for s, _, _ in scenarios], fontsize=8.5)
    ax.set_ylabel("peak RSS (MiB)", fontsize=8.5, color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, ncols=7,
              loc="upper left")
    ax.set_title(f"Peak process memory at {threads_shown} threads (lower is better)",
                 fontsize=12, color=TEXT_PRIMARY, pad=10)
    fig.text(0.01, 0.01,
             "Apple M-series (16 cores), macOS · getrusage peak RSS as reported by the harness",
             fontsize=7.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = RESULTS / "peak-memory.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    raw = RESULTS / "raw.log"
    rows = parse(raw)
    if not rows:
        sys.exit(f"no benchmark lines parsed from {raw}")
    with open(RESULTS / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"parsed {len(rows)} runs -> {RESULTS / 'results.csv'}")
    plot_throughput(rows)
    plot_memory(rows)


if __name__ == "__main__":
    main()
