"""
Qualitative artifact renderers for per-step IG sidecars.

Pure functions: load a sidecar .pt, extract the per-modality attributions,
render to three file layouts:

    out/overlays/<task>/ep{EP}_t{T}.png     render_overlay_only_png
    out/tokens/<task>/ep{EP}_t{T}.png       render_tokens_only_figure
    out/episodes/<task>/ep{EP}_summary.png  render_episode_summary

render_step_figure produces the combined three-panel layout used by the
single-shot demo `ig_rdt.py`.

No RDT or SigLIP import here, these run on any machine with matplotlib
(the 150 GB of sidecars never need to leave the pod; PNGs come home).
"""

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from per_step_attribution import MANISKILL_INDICES, JOINT_NAMES

NUM_PATCHES = 729
GRID_SIZE = 27  # sqrt(NUM_PATCHES)
EXT_CAM_SLOT = 3  # [t-1 ext, t-1 wrist_r, t-1 wrist_l, t ext, t wrist_r, t wrist_l]
POS_COLOR = "#d32f2f"  # red
NEG_COLOR = "#1976d2"  # blue


def observation_array(value):
    """The supported collector saves the exact 384-square RGB input image."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.shape != (384, 384, 3) or value.dtype != np.uint8:
        raise ValueError("Expected the recorded 384x384 RGB uint8 observation; unknown crop geometry is not inferred")
    return value


def model_input_rgb(tensor, mean, std):
    """Display the exact normalized image coordinates used by a vision model."""
    value = tensor.detach().float().cpu().squeeze(0)
    if value.ndim != 3 or value.shape[0] != 3 or not torch.isfinite(value).all():
        raise ValueError("Expected a finite three-channel model input")
    value = value * torch.tensor(std).reshape(3, 1, 1) + torch.tensor(mean).reshape(3, 1, 1)
    return value.permute(1, 2, 0).clamp(0, 1).numpy()


def save_new_figure(figure, path, **kwargs):
    """Refuse to overwrite any historical or existing image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        figure.savefig(stream, format="png", **kwargs)


def extract_vision_heatmap(vision_attr):
    """(1, 4374, H) -> (GRID_SIZE, GRID_SIZE) normalized to [0, 1] from the
    external-camera slot. vision_attr can be bf16 or float, on any device."""
    if vision_attr.ndim != 3 or vision_attr.shape[0] != 1 or vision_attr.shape[1] != 6 * NUM_PATCHES or not torch.isfinite(vision_attr).all():
        raise ValueError("Expected finite attribution for six 729-patch image slots")
    # Signed hidden-coordinate sum followed by absolute value is a plotting
    # convention, distinct from the evaluator's sum of absolute coordinates.
    per_pos = vision_attr.detach().float().squeeze(0).sum(dim=-1).cpu().numpy()
    ext = per_pos[EXT_CAM_SLOT * NUM_PATCHES:(EXT_CAM_SLOT + 1) * NUM_PATCHES]
    grid = np.abs(ext).reshape(GRID_SIZE, GRID_SIZE)
    return grid / grid.max() if grid.max() else grid


def extract_language_per_token(lang_attr, lang_attn_mask=None, n_real=None):
    """
    (1, 1024, H) -> (n_real,) per-token attribution (sum over hidden dim).
    Pass either `lang_attn_mask` (1, 1024) bool tensor, or `n_real` int.
    """
    if lang_attr.ndim != 3 or lang_attr.shape[0] != 1 or not torch.isfinite(lang_attr).all():
        raise ValueError("Expected finite language attribution")
    values = lang_attr.detach().float().squeeze(0).sum(dim=-1)
    if lang_attn_mask is not None:
        mask = torch.as_tensor(lang_attn_mask, device=values.device).bool().reshape(-1)
        if mask.numel() != values.numel():
            raise ValueError("Language attention mask length differs from attribution")
        values = values[mask]
    else:
        if type(n_real) is not int or not 0 < n_real <= values.numel():
            raise ValueError("Supply the exact mask or a valid historical prefix length")
        values = values[:n_real]
    per_tok = values.cpu().numpy()
    return per_tok


def extract_state_per_joint(state_attr):
    """(1, 1, 128) -> (8,) per-joint attribution at MANISKILL_INDICES."""
    if tuple(state_attr.shape) != (1, 1, 128) or not torch.isfinite(state_attr).all():
        raise ValueError("Expected finite raw-state attribution with shape (1,1,128)")
    flat = state_attr.squeeze(0).squeeze(0).detach().cpu().float().numpy()
    return flat[MANISKILL_INDICES]


def selected_token_labels(labels, mask):
    mask = torch.as_tensor(mask).bool().reshape(-1)
    indices = mask.nonzero().flatten().tolist()
    if labels is None:
        return [f"position_{index}" for index in indices]
    if len(labels) == mask.numel():
        return [labels[index] for index in indices]
    if indices == list(range(len(indices))) and len(labels) == len(indices):
        return list(labels)
    raise ValueError("Token labels do not map to the recorded attended positions")


def render_vision_panel(ax, obs_image, heatmap_grid, title="vision patch attribution"):
    """
    Render the vision heatmap overlaid on the observation image on a given
    matplotlib axis. `heatmap_grid` is the GRID_SIZE x GRID_SIZE normalized
    heatmap returned by extract_vision_heatmap.
    """
    observation = observation_array(obs_image)
    if heatmap_grid.shape != (GRID_SIZE, GRID_SIZE) or not np.isfinite(heatmap_grid).all():
        raise ValueError("Expected a finite 27x27 representation map")
    # Put both arrays in the same coordinate extent. Nearest display preserves
    # patch boundaries and does not claim pixel-level attribution within a patch.
    extent = (-.5, 383.5, 383.5, -.5)
    ax.imshow(observation, extent=extent)
    im = ax.imshow(heatmap_grid, cmap="hot", alpha=0.5, interpolation="nearest", extent=extent)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return im


def render_language_panel(ax, lang_attr_per_token, token_labels,
                          title="language token attribution"):
    n_lang = len(lang_attr_per_token)
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in lang_attr_per_token]
    ax.barh(range(n_lang), lang_attr_per_token, color=colors)
    ax.set_yticks(range(n_lang))
    if token_labels is not None and len(token_labels) >= n_lang:
        labels = list(token_labels[:n_lang])
    else:
        labels = [f"tok_{i}" for i in range(n_lang)]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("attribution (sum over hidden dim)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axvline(x=0, color="0.3", linewidth=0.8)
    legend = [Patch(facecolor=POS_COLOR, label="positive"),
              Patch(facecolor=NEG_COLOR, label="negative")]
    ax.legend(handles=legend, loc="lower right", fontsize=9)


def render_state_panel(ax, state_attr_per_joint,
                       title="state attribution (per joint)"):
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in state_attr_per_joint]
    ax.barh(JOINT_NAMES, state_attr_per_joint, color=colors)
    ax.invert_yaxis()
    ax.axvline(x=0, color="0.3", linewidth=0.8)
    ax.set_xlabel("attribution", fontsize=10)
    ax.set_title(title, fontsize=11)
    legend = [Patch(facecolor=POS_COLOR, label="positive"),
              Patch(facecolor=NEG_COLOR, label="negative")]
    ax.legend(handles=legend, loc="lower right", fontsize=9)


def render_step_figure(sidecar, lang_attn_mask, token_labels, out_path,
                       title=None, subtitle=None, dpi=150):
    """
    Three-panel layout (vision | language | state) for one policy call,
    saved to out_path. Mirrors output/ig_rdt.png.

    Args:
        sidecar: dict as saved by per_step_ig.py (keys: vision_attr, lang_attr,
            state_attr, obs_image, ref_action, proprio).
        lang_attn_mask: (1, 1024) bool tensor or None. If None, inferred from
            len(token_labels).
        token_labels: list[str] of SentencePiece pieces, or None.
        out_path: str. Parent dirs created if missing.
    """
    heatmap = extract_vision_heatmap(sidecar["vision_attr"])
    if lang_attn_mask is not None:
        n_real = int(lang_attn_mask.sum().item()) if torch.is_tensor(lang_attn_mask) \
                 else int(np.asarray(lang_attn_mask).sum())
    else:
        n_real = len(token_labels) if token_labels is not None else sidecar["lang_attr"].shape[1]
    lang_per_tok = extract_language_per_token(sidecar["lang_attr"], lang_attn_mask=lang_attn_mask, n_real=n_real)
    if lang_attn_mask is not None:
        token_labels = selected_token_labels(token_labels, lang_attn_mask)
    state_per_joint = extract_state_per_joint(sidecar["state_attr"])
    obs_image = Image.fromarray(sidecar["obs_image"]) \
                if isinstance(sidecar["obs_image"], np.ndarray) else sidecar["obs_image"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 6), dpi=dpi,
        gridspec_kw={"width_ratios": [1.2, 1, 0.8]})
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    im = render_vision_panel(ax1, obs_image, heatmap,
                             title="vision patch attribution (external camera)")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="normalized attribution")
    render_language_panel(ax2, lang_per_tok, token_labels)
    render_state_panel(ax3, state_per_joint)

    if subtitle:
        fig.text(0.5, -0.02, subtitle, ha="center", fontsize=9,
                 fontstyle="italic", color="0.4")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    save_new_figure(fig, out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_overlay_only_png(sidecar, out_path, dpi=150):
    """
    Just the vision heatmap over the observation, no axes or titles.
    Output: H x W x 3 RGB, channel-summed IG, normalized, blended PNG.
    """
    heatmap = extract_vision_heatmap(sidecar["vision_attr"])
    obs_image = Image.fromarray(sidecar["obs_image"]) \
                if isinstance(sidecar["obs_image"], np.ndarray) else sidecar["obs_image"]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    render_vision_panel(ax, obs_image, heatmap, title="")
    ax.set_title("")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_new_figure(fig, out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_tokens_only_figure(sidecar, token_labels, out_path, n_real=None, dpi=150, lang_attn_mask=None):
    """Just the language bar chart. n_real defaults to len(token_labels)."""
    if n_real is None:
        n_real = len(token_labels) if token_labels is not None else sidecar["lang_attr"].shape[1]
    lang_per_tok = extract_language_per_token(sidecar["lang_attr"], lang_attn_mask=lang_attn_mask, n_real=n_real)
    if lang_attn_mask is not None:
        token_labels = selected_token_labels(token_labels, lang_attn_mask)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(lang_per_tok))), dpi=dpi)
    render_language_panel(ax, lang_per_tok, token_labels)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    save_new_figure(fig, out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_episode_summary(sidecars_in_order: Iterable[dict], out_path,
                           n_frames=4, dpi=150):
    """
    Stitch key frames (default 4) from an episode into a single figure.
    Picks evenly-spaced indices from the full list of sidecars.
    """
    sidecars = list(sidecars_in_order)
    if len(sidecars) == 0:
        raise ValueError("Cannot render an empty episode")
    if type(n_frames) is not int or n_frames < 2:
        raise ValueError("Episode summaries require at least two requested frames")
    if len(sidecars) <= n_frames:
        picks = list(range(len(sidecars)))
    else:
        picks = [int(round(i * (len(sidecars) - 1) / (n_frames - 1)))
                 for i in range(n_frames)]
    picks = sorted(set(picks))

    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), dpi=dpi)
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, idx in enumerate(picks):
        sc = sidecars[idx]
        obs_image = Image.fromarray(sc["obs_image"]) \
                    if isinstance(sc["obs_image"], np.ndarray) else sc["obs_image"]
        heatmap = extract_vision_heatmap(sc["vision_attr"])
        axes[0, col].imshow(observation_array(obs_image))
        axes[0, col].set_title(f"call {sc.get('policy_call_idx', idx)}", fontsize=10)
        axes[0, col].axis("off")
        render_vision_panel(axes[1, col], obs_image, heatmap,
                            title=f"attribution (call {sc.get('policy_call_idx', idx)})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    save_new_figure(fig, out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
