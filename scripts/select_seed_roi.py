#!/usr/bin/env python3
"""Interactively select a CT seed and in-plane ROI, then enter the 3D slice range.

Keyboard:
  up/down (or left/right) = navigate axial slice
  s = select seed (single left click)
  r = draw rectangular XY ROI (left-button drag)
  q = finish and close

Displayed x-axis is canonical voxel i; y-axis is canonical voxel j.
This script loads CT only; it does not load or display GT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from nibabel.orientations import apply_orientation
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from src.common.io_spatial import load_case


def find_nii(folder: Path, case_id: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        candidate = folder / f"{case_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find {case_id}.nii[.gz] in {folder}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--case-id", default="spleen_2")
    parser.add_argument("--slice-index", type=int, default=None,
                        help="starting canonical axial slice k; default middle slice")
    parser.add_argument("--output-json", type=Path,
                        default=Path("outputs/traditional/spleen_2_seed_roi.json"))
    parser.add_argument("--window", type=float, nargs=2, default=(-57.0, 164.0),
                        metavar=("LOW_HU", "HIGH_HU"))
    args = parser.parse_args()

    image_path = find_nii(args.data_root / "imagesTr", args.case_id)
    case = load_case(image_path, None)
    ct_ras = apply_orientation(
        np.asarray(case["ct_original"], dtype=np.float32),
        np.asarray(case["to_ras_ornt"], dtype=np.float64),
    )
    nx, ny, nz = ct_ras.shape
    initial_z = nz // 2 if args.slice_index is None else int(args.slice_index)
    if not 0 <= initial_z < nz:
        raise ValueError(f"slice-index must be in [0, {nz - 1}], got {initial_z}")

    selected = {"seed_ij": None, "roi_ij": None, "mode": "seed"}
    z_state = {"value": initial_z}
    fig, ax = plt.subplots(figsize=(9, 8))
    low, high = args.window
    image_artist = ax.imshow(
        ct_ras[:, :, z_state["value"]].T,
        cmap="gray", vmin=low, vmax=high, origin="lower"
    )
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xlabel("canonical voxel i")
    ax.set_ylabel("canonical voxel j")
    title_template = (
        f"{args.case_id} | axial k={{k}}/{nz-1}\n"
        "Up/Down: change slice | S: select seed | R: draw ROI | Q: finish"
    )
    ax.set_title(title_template.format(k=z_state["value"]))
    marker = {"artist": None}
    selector_holder = {"selector": None}

    def clear_selection():
        selected["seed_ij"] = None
        selected["roi_ij"] = None
        if marker["artist"] is not None:
            marker["artist"].remove()
            marker["artist"] = None
        selector_obj = selector_holder["selector"]
        if selector_obj is not None:
            try:
                selector_obj.extents = (0, 0, 0, 0)
            except Exception:
                pass

    def on_click(event):
        if selected["mode"] != "seed" or event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        i = min(max(int(round(event.xdata)), 0), nx - 1)
        j = min(max(int(round(event.ydata)), 0), ny - 1)
        selected["seed_ij"] = (i, j)
        if marker["artist"] is not None:
            marker["artist"].remove()
        marker["artist"] = ax.plot(
            i, j, marker="+", color="yellow", markersize=14, mew=2
        )[0]
        fig.canvas.draw_idle()
        print(f"Selected seed (i,j,k)=({i},{j},{z_state['value']})")

    def on_rectangle(eclick, erelease):
        if selected["mode"] != "roi":
            return
        if (eclick.xdata is None or eclick.ydata is None
                or erelease.xdata is None or erelease.ydata is None):
            return
        i0 = max(0, int(np.floor(min(eclick.xdata, erelease.xdata))))
        i1 = min(nx, int(np.ceil(max(eclick.xdata, erelease.xdata))) + 1)
        j0 = max(0, int(np.floor(min(eclick.ydata, erelease.ydata))))
        j1 = min(ny, int(np.ceil(max(eclick.ydata, erelease.ydata))) + 1)
        if i1 <= i0 or j1 <= j0:
            print("ROI selection was empty; draw a larger rectangle.")
            return
        selected["roi_ij"] = (i0, i1, j0, j1)
        print(f"Selected in-plane ROI: i=[{i0},{i1}), j=[{j0},{j1})")
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, on_rectangle, useblit=True, button=[1], minspanx=5, minspany=5,
        spancoords="pixels", interactive=True,
        props={"facecolor": "none", "edgecolor": "yellow", "linewidth": 1.5},
    )
    selector_holder["selector"] = selector
    selector.set_active(False)

    def on_key(event):
        if event.key in ("s", "S"):
            selected["mode"] = "seed"
            selector.set_active(False)
            print("Mode: seed. Left-click once inside the spleen.")
        elif event.key in ("r", "R"):
            selected["mode"] = "roi"
            selector.set_active(True)
            print("Mode: ROI. Left-button drag a rectangle around the spleen and nearby margin.")
        elif event.key in ("up", "right"):
            z_state["value"] = min(nz - 1, z_state["value"] + 1)
            clear_selection()
            image_artist.set_data(ct_ras[:, :, z_state["value"]].T)
            ax.set_title(title_template.format(k=z_state["value"]))
            fig.canvas.draw_idle()
            print(f"Showing axial slice k={z_state['value']}; reselect seed and ROI.")
        elif event.key in ("down", "left"):
            z_state["value"] = max(0, z_state["value"] - 1)
            clear_selection()
            image_artist.set_data(ct_ras[:, :, z_state["value"]].T)
            ax.set_title(title_template.format(k=z_state["value"]))
            fig.canvas.draw_idle()
            print(f"Showing axial slice k={z_state['value']}; reselect seed and ROI.")
        elif event.key in ("q", "Q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Interactive CT-only selection opened.")
    print("Use Up/Down to find spleen; press S and click seed; press R and drag ROI; Q to finish.")
    plt.show()

    if selected["seed_ij"] is None:
        raise RuntimeError("No seed selected. Rerun the script and press S before clicking.")
    if selected["roi_ij"] is None:
        raise RuntimeError("No ROI selected. Rerun the script and press R to draw a rectangle.")

    z = int(z_state["value"])
    i, j = selected["seed_ij"]
    i0, i1, j0, j1 = selected["roi_ij"]
    if not (i0 <= i < i1 and j0 <= j < j1):
        raise RuntimeError(
            f"Seed (i={i}, j={j}) is outside the selected ROI. "
            "Rerun and include the seed in ROI."
        )

    default_k0 = max(0, z - 15)
    default_k1 = min(nz, z + 16)
    entered = input(
        f"Enter z-range k0 k1, half-open, 0 <= k0 < k1 <= {nz} "
        f"[default {default_k0} {default_k1}]: "
    ).strip()
    if entered:
        parts = entered.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError("Enter exactly two integers, e.g. 25 65")
        k0, k1 = (int(v) for v in parts)
    else:
        k0, k1 = default_k0, default_k1
    if not (0 <= k0 < k1 <= nz):
        raise ValueError(f"Invalid z range [{k0},{k1}) for {nz} slices")
    if not k0 <= z < k1:
        raise ValueError(f"Selected slice k={z} must be inside z range [{k0},{k1})")

    payload = {
        "schema_version": "1.0",
        "case_id": args.case_id,
        "image_filename": image_path.name,
        "coordinate_system": "closest-RAS canonical voxel indices; not physical millimeters",
        "seed_ijk": [int(i), int(j), int(z)],
        "roi_ijk_half_open": [int(i0), int(i1), int(j0), int(j1), int(k0), int(k1)],
        "display_slice_k": int(z),
        "display_hu_window": [float(low), float(high)],
        "selection_source": "manual selection from CT only; GT not loaded or displayed",
        "note": "Semi-automatic method; record manual seed and ROI when reporting results.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved selection: {args.output_json}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
