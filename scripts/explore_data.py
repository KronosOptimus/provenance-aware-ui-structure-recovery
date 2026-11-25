"""Data analysis: measurements + report figures from the node table.

Run:  python scripts/explore_data.py --config configs/explore_data.yaml
Writes results/explore_data.json (every number quoted in the report) and results/figures/data_*.png.
Reads only data/processed/*.parquet plus a few screenshots for the example figure.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.metrics import f1_score

from uistruct.plotting import GREY, MINT, MINT_D, PEACH, PEACH_D, SKY, SKY_D, use_report_style

LABELS = {-1: "ROOT", 0: "BACKGROUND", 1: "IMAGE", 2: "PICTOGRAM", 3: "BUTTON", 4: "TEXT", 5: "LABEL",
          6: "TEXT_INPUT", 7: "MAP", 8: "CHECK_BOX", 9: "SWITCH", 10: "PAGER_INDICATOR", 11: "SLIDER",
          12: "RADIO_BUTTON", 13: "SPINNER", 14: "PROGRESS_BAR", 15: "ADVERTISEMENT", 16: "DRAWER",
          17: "NAVIGATION_BAR", 18: "TOOLBAR", 19: "LIST_ITEM", 20: "CARD_VIEW", 21: "CONTAINER",
          22: "DATE_PICKER", 23: "NUMBER_STEPPER"}
SPLITS = ["train", "dev", "test"]
# Report figure style (shared soft palette). Valid = sky, BACKGROUND (invalid) = peach.
use_report_style()


def pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def quant(x):
    x = np.asarray(x)
    return {"mean": round(float(x.mean()), 2), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def clay_depth(lab):
    """Depth in CLAY's tree (follow clay_parent_idx). lab is sorted by (screen_id, node_idx)."""
    key = lab.screen_id.to_numpy().astype(np.int64) * 10_000 + lab.node_idx.to_numpy()
    pkey = lab.screen_id.to_numpy().astype(np.int64) * 10_000 + lab.clay_parent_idx.to_numpy()
    pos = np.searchsorted(key, pkey)
    has_parent = lab.clay_parent_idx.to_numpy() >= 0
    pos[~has_parent] = -1
    depth = np.zeros(len(lab), dtype=np.int16)
    for i in range(len(lab)):  # pre-order: a parent always precedes its children
        p = pos[i]
        if p >= 0:
            depth[i] = depth[p] + 1
    return depth, pos


def main(cfg):
    t0 = time.time()
    fig_dir = Path(cfg["fig_dir"]); fig_dir.mkdir(parents=True, exist_ok=True)
    out = {"config": cfg, "env": {"python": platform.python_version(), "pandas": pd.__version__,
                                  "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    s = pd.read_parquet(f"{cfg['data_dir']}/screens.parquet")
    n = pd.read_parquet(f"{cfg['data_dir']}/nodes.parquet", columns=[
        "screen_id", "node_idx", "parent_idx", "clay_parent_idx", "depth", "n_children", "label",
        "is_labelled", "x0", "y0", "x1", "y1", "nx0", "ny0", "nx1", "ny1", "box_ok", "class_name", "clickable", "focusable",
        "visible_to_user", "enabled", "scrollable_v", "scrollable_h", "text"])
    n = n.sort_values(["screen_id", "node_idx"], kind="stable").reset_index(drop=True)
    n["split"] = n.screen_id.map(s.set_index("screen_id").split)
    n["w"] = (n.nx1 - n.nx0).clip(lower=0); n["h"] = (n.ny1 - n.ny0).clip(lower=0)
    n["area"] = n.w * n.h
    lab = n[n.is_labelled].reset_index(drop=True)
    lab["name"] = lab.label.map(LABELS)

    # ---- A. class distribution per split ----
    A = {}
    cnt = lab.groupby(["split", "name"]).size().unstack(0).fillna(0).astype(int)[SPLITS]
    share = cnt / cnt.sum() * 100
    A["counts"] = cnt.to_dict()
    A["max_share_gap_between_splits_pct_points"] = round(float((share.max(1) - share.min(1)).max()), 3)
    tot = cnt.sum(1).sort_values(ascending=False)
    A["imbalance_ratio_largest_to_smallest"] = round(float(tot.iloc[0] / tot.iloc[-1]), 1)
    A["classes_under_0.5pct"] = [k for k, v in (tot / tot.sum() * 100).items() if v < 0.5]
    A["test_classes_under_100_nodes"] = {k: int(v) for k, v in cnt["test"].items() if v < 100}
    out["A_class_distribution"] = A

    # ---- B. density (elements per screen), Screen Parsing bins ----
    B = {}
    bins = [0, 16, 32, 48, 64, np.inf]; names = ["0-16", "16-32", "32-48", "48-64", "64+"]
    s["bin"] = pd.cut(s.n_labelled, bins, right=False, labels=names)
    B["labelled_per_screen"] = quant(s.n_labelled)
    B["raw_nodes_per_screen"] = quant(s.n_nodes)
    B["labelled_share_of_raw_nodes_pct"] = pct(int(s.n_labelled.sum()), int(s.n_nodes.sum()))
    B["screens_per_bin"] = {sp: s[s.split == sp].bin.value_counts().reindex(names).to_dict() for sp in SPLITS}
    B["screens_with_lt64_labelled_pct"] = pct(int((s.n_labelled < 64).sum()), len(s))
    vis_raw = n[n.visible_to_user & n.box_ok].groupby("screen_id").size()
    B["visible_boxok_raw_nodes_per_screen"] = quant(vis_raw.reindex(s.screen_id).fillna(0))
    out["B_density"] = B

    # ---- C. tree structure: raw vs CLAY tree ----
    C = {}
    cd, ppos = clay_depth(lab)
    lab["clay_depth"] = cd
    C["raw_depth_all_nodes"] = quant(n.depth)
    C["raw_depth_labelled_nodes"] = quant(lab.depth)
    C["clay_tree_depth_labelled_nodes"] = quant(lab.clay_depth)
    C["max_depth_per_screen_raw"] = quant(n.groupby("screen_id").depth.max())
    C["max_depth_per_screen_clay"] = quant(lab.groupby("screen_id").clay_depth.max())
    kids = pd.Series(ppos[ppos >= 0]).value_counts()
    nkids = np.zeros(len(lab), dtype=np.int32); nkids[kids.index.to_numpy()] = kids.to_numpy()
    lab["clay_children"] = nkids
    internal = lab[lab.clay_children > 0]
    C["clay_internal_nodes_pct_of_labelled"] = pct(len(internal), len(lab))
    C["clay_children_per_internal_node"] = quant(internal.clay_children)
    C["internal_share_by_class_pct"] = (lab.groupby("name").clay_children.apply(lambda x: (x > 0).mean() * 100)
                                        .round(2).sort_values(ascending=False).to_dict())
    # How many raw wrapper levels CLAY's tree skips between a labelled node and its CLAY parent.
    has_p = ppos >= 0
    skipped = (lab.depth.to_numpy()[has_p] - lab.depth.to_numpy()[ppos[has_p]] - 1)
    C["raw_levels_skipped_per_clay_edge"] = quant(skipped)
    C["clay_edges_skipping_ge1_level_pct"] = pct(int((skipped >= 1).sum()), int(has_p.sum()))
    # Does the CLAY parent's box contain the child's box?  (structure is spatial containment?)
    px0, py0, px1, py1 = (lab[c].to_numpy()[ppos[has_p]] for c in ["nx0", "ny0", "nx1", "ny1"])
    cx0, cy0, cx1, cy1 = (lab[c].to_numpy()[has_p] for c in ["nx0", "ny0", "nx1", "ny1"])
    eps = 1e-4
    contained = (cx0 >= px0 - eps) & (cy0 >= py0 - eps) & (cx1 <= px1 + eps) & (cy1 <= py1 + eps)
    C["clay_child_box_inside_parent_box_pct"] = pct(int(contained.sum()), int(has_p.sum()))
    out["C_tree"] = C

    # ---- D. metadata noise: CLAY BACKGROUND (invalid) nodes ----
    D = {}
    nonroot = lab[lab.label >= 0]
    bg = nonroot.label == 0
    per = nonroot.groupby("screen_id").label.apply(lambda x: (x == 0).mean())
    D["screens_with_ge1_background"] = int((per > 0).sum())
    D["screens_with_ge1_background_pct"] = pct(int((per > 0).sum()), len(s))
    D["background_share_per_screen_given_ge1"] = quant(per[per > 0])
    D["screens_background_ge50pct"] = int((per >= 0.5).sum())
    D["background_by_split_pct_of_labelled_nonroot"] = {sp: pct(int(((nonroot.split == sp) & bg).sum()),
                                                             int((nonroot.split == sp).sum())) for sp in SPLITS}
    flags = ["visible_to_user", "box_ok", "clickable", "focusable", "enabled"]
    D["flag_rate_pct_background_vs_valid"] = {f: {"background": pct(int(nonroot[bg][f].sum()), int(bg.sum())),
                                                  "valid": pct(int(nonroot[~bg][f].sum()), int((~bg).sum()))}
                                              for f in flags}
    inframe = (nonroot.nx0 >= -1e-4) & (nonroot.ny0 >= -1e-4) & (nonroot.nx1 <= 1 + 1e-4) & (nonroot.ny1 <= 1 + 1e-4)
    D["box_fully_inside_frame_pct"] = {"background": pct(int(inframe[bg].sum()), int(bg.sum())),
                                       "valid": pct(int(inframe[~bg].sum()), int((~bg).sum()))}
    D["median_area_background_vs_valid"] = {"background": round(float(nonroot[bg].area.median()), 4),
                                            "valid": round(float(nonroot[~bg].area.median()), 4)}
    D["background_top_class_names_pct"] = (nonroot[bg].class_name.str.split(".").str[-1].value_counts(normalize=True)
                                           .head(8).mul(100).round(2).to_dict())
    pkg = s.activity_name.str.split("/").str[0]
    app_bg = (per.reindex(s.screen_id).fillna(0).to_numpy() > 0)
    app = pd.DataFrame({"pkg": pkg.to_numpy(), "bg": app_bg}).groupby("pkg").bg.mean()
    D["apps"] = len(app)
    D["apps_with_bg_on_every_screen"] = int((app == 1).sum())
    D["apps_with_no_bg_screen"] = int((app == 0).sum())
    out["D_metadata_noise"] = D

    # ---- E. what CLAY leaves unlabelled ----
    E = {}
    un = n[~n.is_labelled]
    E["unlabelled_nodes"] = len(un)
    E["unlabelled_pct_of_raw"] = pct(len(un), len(n))
    E["unlabelled_visible_boxok_pct"] = pct(int((un.visible_to_user & un.box_ok).sum()), len(un))
    vu = un[un.visible_to_user & un.box_ok]
    E["unlabelled_visible_boxok_top_class_names_pct"] = (vu.class_name.str.split(".").str[-1]
                                                         .value_counts(normalize=True).head(10).mul(100).round(2).to_dict())
    E["unlabelled_visible_boxok_with_text_pct"] = pct(int(vu.text.fillna("").str.len().gt(0).sum()), len(vu))
    # The box overlay check showed labelled text inside an unlabelled clickable wrapper (a button CLAY did not box).
    n_pos = n.groupby("screen_id").cumcount()  # == node_idx position within screen when sorted
    assert (n_pos.to_numpy() == n.node_idx.to_numpy()).all()
    offs = np.r_[0, np.cumsum(n.groupby("screen_id").size().to_numpy())[:-1]]
    soff = np.repeat(offs, n.groupby("screen_id").size().to_numpy())
    par = n.parent_idx.to_numpy()
    pidx = np.where(par >= 0, soff + par, -1)
    plab = np.where(pidx >= 0, n.is_labelled.to_numpy()[np.maximum(pidx, 0)], True)
    pclick = np.where(pidx >= 0, n.clickable.to_numpy()[np.maximum(pidx, 0)], False)
    pvis = np.where(pidx >= 0, n.visible_to_user.to_numpy()[np.maximum(pidx, 0)], False)
    is_text = (n.label == 4).to_numpy()
    wrapped = is_text & ~plab & pclick & pvis
    E["labelled_TEXT_with_unlabelled_clickable_visible_parent"] = int(wrapped.sum())
    E["labelled_TEXT_with_unlabelled_clickable_visible_parent_pct_of_TEXT"] = pct(int(wrapped.sum()), int(is_text.sum()))
    E["screens_with_such_TEXT"] = int(n.screen_id[wrapped].nunique())
    E["screens_with_such_TEXT_pct"] = pct(int(n.screen_id[wrapped].nunique()), len(s))
    # CLAY pre-filtered RICO before labelling (paper §3: duplicate boxes keep one node, tiny/invisible removed).
    B4 = n[["x0", "y0", "x1", "y1"]].to_numpy()
    pi0 = np.maximum(pidx, 0)
    same = np.abs(B4[wrapped] - B4[pi0[wrapped]]).max(1) <= 2
    E["wrapped_TEXT_parent_box_equals_child_box_pct"] = pct(int(same.sum()), int(wrapped.sum()))
    bkey = lambda d: (d.screen_id.astype(str) + ":" + d.x0.astype(str) + "," + d.y0.astype(str) + ","
                      + d.x1.astype(str) + "," + d.y1.astype(str))
    vu_full = n[~n.is_labelled & n.visible_to_user & n.box_ok]
    dup = bkey(vu_full).isin(set(bkey(n[n.is_labelled])))
    E["unlabelled_visible_boxok_same_box_as_labelled_node_pct"] = pct(int(dup.sum()), len(vu_full))
    out["E_unlabelled"] = E

    # ---- F. how much the metadata alone says: class_name -> label lookup ----
    F = {}
    lab["cls"] = lab.class_name.str.split(".").str[-1].fillna("")
    nr = lab[lab.label >= 0]
    tr, te = nr[nr.split == "train"], nr[nr.split == "test"]
    lut = tr.groupby("cls").label.agg(lambda x: x.value_counts().idxmax())
    fallback = int(tr.label.value_counts().idxmax())
    pred = te.cls.map(lut).fillna(fallback).astype(int)
    F["test_macro_f1"] = round(float(f1_score(te.label, pred, average="macro")), 4)
    F["test_micro_f1"] = round(float(f1_score(te.label, pred, average="micro")), 4)
    F["test_class_name_unseen_in_train_pct"] = pct(int((~te.cls.isin(lut.index)).sum()), len(te))
    F["distinct_class_names_train"] = int(tr.cls.nunique())
    pc = f1_score(te.label, pred, average=None, labels=sorted(te.label.unique()))
    F["per_class_f1"] = {LABELS[k]: round(float(v), 3) for k, v in zip(sorted(te.label.unique()), pc)}
    F["note"] = ("Analysis only, not a model configuration: label = most frequent CLAY label for the Android "
                 "class name (last dotted part) on train; unseen names get the train majority label. "
                 "Non-root labelled nodes, 24 classes incl. BACKGROUND (ROOT excluded).")
    out["F_class_name_lookup"] = F

    # ---- G. geometry by class ----
    G = {}
    G["median_area_by_class"] = nr.groupby("name").area.median().round(5).sort_values().to_dict()
    G["median_cy_by_class"] = ((nr.ny0 + nr.ny1) / 2).groupby(nr.name).median().round(3).to_dict()
    out["G_geometry"] = G

    # ---- H. leakage: apps across splits ----
    H = {}
    sp_pkg = pd.DataFrame({"split": s.split, "pkg": pkg})
    sets = {sp: set(sp_pkg[sp_pkg.split == sp].pkg) for sp in SPLITS}
    H["apps_per_split"] = {sp: len(v) for sp, v in sets.items()}
    H["apps_in_train_and_test"] = len(sets["train"] & sets["test"])
    H["test_screens_from_apps_seen_in_train_pct"] = pct(int(sp_pkg[(sp_pkg.split == "test")].pkg.isin(sets["train"]).sum()),
                                                       int((sp_pkg.split == "test").sum()))
    out["H_split_leakage"] = H

    # ================= figures =================
    order = tot.index.tolist()
    # Fig 1: class distribution (log)
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    vals = tot.to_numpy()
    cols = [PEACH if k == "BACKGROUND" else (GREY if k == "ROOT" else SKY) for k in order]
    edges = [PEACH_D if k == "BACKGROUND" else ("#7a7975" if k == "ROOT" else SKY_D) for k in order]
    ax.bar(range(len(order)), vals, color=cols, edgecolor=edges, lw=0.6)
    ax.set_yscale("log"); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Labelled nodes (log scale)")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{100 * v / vals.sum():.1f}" if v / vals.sum() >= 0.001 else f"{100 * v / vals.sum():.2f}",
                ha="center", fontsize=5.5)
    ax.set_title("CLAY labels over all splits (% of 1,427,913 nodes above bars; BACKGROUND = invalid node)", fontsize=8)
    fig.savefig(fig_dir / "data_class_distribution.png"); plt.close(fig)

    # Fig 2: density
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5))
    axs[0].hist(s.n_nodes.clip(upper=400), bins=80, color=MINT, label="all RICO nodes")
    axs[0].hist(s.n_labelled.clip(upper=400), bins=80, color=SKY_D, alpha=0.75, label="CLAY-labelled nodes")
    axs[0].set_xlabel("Nodes per screen (clipped at 400)"); axs[0].set_ylabel("Screens"); axs[0].legend(frameon=False, fontsize=7)
    x = np.arange(len(names)); wdt = 0.27
    for k, (sp, c) in enumerate(zip(SPLITS, [(SKY, SKY_D), (PEACH, PEACH_D), (MINT, MINT_D)])):
        v = s[s.split == sp].bin.value_counts(normalize=True).reindex(names).to_numpy() * 100
        axs[1].bar(x + (k - 1) * wdt, v, wdt, color=c[0], edgecolor=c[1], lw=0.6, label=sp)
    axs[1].set_xticks(x); axs[1].set_xticklabels(names); axs[1].set_xlabel("Labelled nodes per screen (Screen Parsing bins)")
    axs[1].set_ylabel("% of split's screens"); axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "data_density.png"); plt.close(fig)

    # Fig 3: depth raw vs CLAY tree (labelled nodes)
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    mx = int(max(lab.depth.max(), lab.clay_depth.max()))
    b = np.arange(mx + 2) - 0.5
    ax.hist(lab.depth, bins=b, color=MINT, edgecolor=MINT_D, lw=0.4, label="raw RICO hierarchy")
    ax.hist(lab.clay_depth, bins=b, histtype="step", color=SKY_D, lw=1.6, label="CLAY tree")
    ax.set_xlabel("Depth of labelled node"); ax.set_ylabel("Nodes"); ax.legend(frameon=False, fontsize=7)
    fig.savefig(fig_dir / "data_depth.png"); plt.close(fig)

    # Fig 4: metadata noise
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"width_ratios": [1.1, 1]})
    axs[0].hist(per[per > 0] * 100, bins=40, color=PEACH, edgecolor=PEACH_D, lw=0.4)
    axs[0].set_xlabel("% of screen's labelled nodes that are BACKGROUND"); axs[0].set_ylabel("Screens")
    axs[0].set_title(f"{D['screens_with_ge1_background']:,} screens ({D['screens_with_ge1_background_pct']}%) have ≥1", fontsize=8)
    fl = flags + ["in frame"]
    bgv = [D["flag_rate_pct_background_vs_valid"][f]["background"] for f in flags] + [D["box_fully_inside_frame_pct"]["background"]]
    vav = [D["flag_rate_pct_background_vs_valid"][f]["valid"] for f in flags] + [D["box_fully_inside_frame_pct"]["valid"]]
    y = np.arange(len(fl))
    axs[1].barh(y - 0.2, vav, 0.4, color=SKY, edgecolor=SKY_D, lw=0.6, label="valid")
    axs[1].barh(y + 0.2, bgv, 0.4, color=PEACH, edgecolor=PEACH_D, lw=0.6, label="BACKGROUND")
    axs[1].set_yticks(y); axs[1].set_yticklabels([f.replace("_", " ") for f in fl], fontsize=7)
    axs[1].set_xlabel("% of nodes with the flag")
    axs[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2)
    axs[1].set_title("Metadata flags barely separate them", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "data_metadata_noise.png"); plt.close(fig)

    # Fig 5: geometry overlap by class (area, log)
    cls_order = [k for k in order if k != "ROOT"]
    data = [np.log10(nr[nr.name == k].area.clip(lower=1e-6).to_numpy()) for k in cls_order]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True, medianprops={"color": PEACH_D, "lw": 1.4},
               boxprops={"facecolor": SKY, "edgecolor": SKY_D}, whiskerprops={"color": GREY}, capprops={"color": GREY})
    ax.set_xticks(range(1, len(cls_order) + 1)); ax.set_xticklabels(cls_order, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("log10 box area (fraction of screen)")
    fig.savefig(fig_dir / "data_area_by_class.png"); plt.close(fig)

    # Fig 6: example screen, valid vs BACKGROUND boxes (deterministic pick)
    cand = per[(per > 0.2) & (per < 0.6)].index
    cand = s[s.screen_id.isin(cand) & (s.split == "train") & (s.img_w == 1080) & s.n_labelled.between(15, 35)].screen_id
    sid = int(cand.sort_values().iloc[cfg["example_index"]])
    im = Image.open(f"{cfg['rico_dir']}/{sid}.jpg").convert("RGB"); W, Hh = im.size
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 4.1))
    sub = lab[(lab.screen_id == sid) & (lab.label >= 0)]
    for ax, (title, mask, col) in zip(axs, [("Valid labelled nodes", sub.label > 0, SKY_D),
                                           ("BACKGROUND (invalid) nodes", sub.label == 0, PEACH_D)]):
        ax.imshow(im); ax.set_axis_off(); ax.set_title(title, fontsize=8)
        for r in sub[mask].itertuples():
            ax.add_patch(plt.Rectangle((r.nx0 * W, r.ny0 * Hh), (r.nx1 - r.nx0) * W, (r.ny1 - r.ny0) * Hh,
                                       fill=False, lw=1.3, ec=col))
    fig.tight_layout(); fig.savefig(fig_dir / "data_example_screen.png"); plt.close(fig)
    out["example_screen_id"] = sid

    out["seconds"] = round(time.time() - t0, 1)
    Path(cfg["results"]).write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: v for k, v in out.items() if k != "config"}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/explore_data.yaml")
    main(yaml.safe_load(open(ap.parse_args().config)))
