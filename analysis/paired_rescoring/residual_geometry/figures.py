"""Deterministic Agg figures for the residual geometry analysis (no titles).

Each panel plots Delta (symmetric log scale) against B (log scale) for one
case. Color encodes log10(max interior residual / B) on a diverging scale
whose gray midpoint is 0, the overshoot boundary; first calls are triangles.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib
from PIL import Image, features
import PIL

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ft2font  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = ("#0d366b", "#3987e5", "#a9a8a2", "#e34948", "#7a1616")
RC = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 6.5,
    "axes.labelsize": 6.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "pdf.compression": 0, "svg.hashsalt": "residual-geometry", "path.simplify": False,
}


def colormap():
    return LinearSegmentedColormap.from_list("residual_ratio", DIVERGING, N=256)


def render(rows, protocol, output):
    spec = protocol["figures"]
    vmin, vmax = spec["log10_ratio_limits"]
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = colormap()
    clipped = {}
    # Start from matplotlib's defaults so global rcParams set elsewhere cannot leak in.
    with plt.style.context("default"), matplotlib.rc_context(RC):
        for figure in spec["figures"]:
            fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0))
            fig.subplots_adjust(left=0.085, right=0.885, bottom=0.1, top=0.94, wspace=0.36, hspace=0.38)
            n_clipped = 0
            for ax, panel in zip(axes.flat, figure["panels"]):
                members = [r for r in rows if r["case"] == panel["case"] and r["paired_status"] == "defined"]
                if not members:
                    raise ValueError(f"Empty figure panel: {panel['case']}")
                ax.axhline(0.0, color=AXIS, linewidth=0.6, zorder=1)
                for first, marker, size in ((False, "o", 4.0), (True, "^", 13.0)):
                    group = [r for r in members if r["first_call"] == first]
                    ratios = [math.log10(r["ratio_max_interior_to_baseline"]) for r in group]
                    n_clipped += sum(v < vmin or v > vmax for v in ratios)
                    ax.scatter([r["residual_baseline"] for r in group], [r["delta"] for r in group],
                               c=[min(max(v, vmin), vmax) for v in ratios], cmap=cmap, norm=norm,
                               s=size, marker=marker, linewidths=0.35 if first else 0.0,
                               edgecolors=INK if first else "none", zorder=3 if first else 2)
                ax.set_xscale("log")
                ax.set_yscale("symlog", linthresh=spec["symlog_linthresh"], linscale=1.0)
                ax.margins(x=0.05, y=0.08)
                ax.grid(True, which="major", color=GRID, linewidth=0.4, zorder=0)
                ax.text(0.0, 1.03, panel["label"], transform=ax.transAxes, color=SECONDARY,
                        fontsize=5.5, ha="left", va="bottom")
            fig.supxlabel(spec["x_label"], fontsize=6.5, y=0.015)
            fig.supylabel(spec["y_label"], fontsize=6.5, x=0.012)
            bar_axis = fig.add_axes([0.92, 0.2, 0.012, 0.6])
            bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar_axis)
            bar.set_label(spec["color_label"], color=INK)
            bar.outline.set_linewidth(0.4)
            bar.ax.tick_params(labelsize=5.5, width=0.4)
            handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=2.6, color=MUTED, label=spec["later_label"]),
                       Line2D([0], [0], marker="^", linestyle="none", markersize=3.6, color=MUTED,
                              markeredgecolor=INK, markeredgewidth=0.35, label=spec["first_label"])]
            fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, 0.005), frameon=False,
                       fontsize=5.5, handletextpad=0.3, borderaxespad=0.2)
            fig.savefig(Path(output) / f"{figure['name']}.pdf", metadata={"CreationDate": None})
            fig.savefig(Path(output) / f"{figure['name']}.png", dpi=300)
            plt.close(fig)
            clipped[figure["name"]] = n_clipped
    return dict(matplotlib=matplotlib.__version__, backend="Agg", freetype=ft2font.__freetype_version__,
                pillow=PIL.__version__, pillow_zlib=features.version("zlib"), pdf_compression=0,
                png_dpi=300, pdf_metadata={"CreationDate": None}, color_values_clipped_to_limits=clipped)


def png_pixels_sha256(path):
    """Hash of the decoded RGBA pixels, independent of the PNG deflate build."""
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()
