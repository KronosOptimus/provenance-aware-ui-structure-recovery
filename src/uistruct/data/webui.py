"""WebUI (Wu et al., CHI 2023) pages for the Android-to-web test.

The dataset is published on Hugging Face as one zip archive per split, cut into ~4.3 GB parts
(`<name>.zip.001`, `.002`, ...). The parts are a plain byte split of one zip file, so the archive's
central directory can be read with HTTP range requests and single members fetched without downloading
the whole split (the test split is 17 GB; the pages used here are a few hundred MB).

Per page and simulated device the crawler stored (WebUI repository, crawler/process.js):
  <device>-axtree.json.gz    Chrome's full accessibility tree (Accessibility.getFullAXTree)
  <device>-bb.json.gz        bounding box per DOM node (Puppeteer boundingBox), keyed by backendDOMNodeId
  <device>-viewport.json.gz  whether each DOM node intersects the viewport, keyed by backendDOMNodeId
  <device>-screenshot.webp   the viewport screenshot
"""
import gzip
import io
import json
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = "https://huggingface.co/datasets/biglab/{repo}/resolve/main/{name}"
SPLITS = {"test": ("webui-test", "test_split_webui", 5), "val": ("webui-val", "val_split_webui", 3),
          "train": ("webui-7k", "train_split_web7k", 2)}
DEVICES = {"phone": "iPhone-13 Pro", "desktop": "default_1280-720"}
FILES = ["axtree.json.gz", "bb.json.gz", "viewport.json.gz", "screenshot.webp"]


def _open(req, tries=5):
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


class RemoteZip:
    """Read-only view of a multi-part zip on Hugging Face (parts concatenated byte for byte)."""

    def __init__(self, split):
        repo, stem, n = SPLITS[split]
        self.urls = [HF.format(repo=repo, name=f"{stem}.zip.{i:03d}") for i in range(1, n + 1)]
        sizes = []
        for u in self.urls:
            with urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=120) as r:
                sizes.append(int(r.headers["Content-Length"]))
        self.offs = [0]
        for s in sizes:
            self.offs.append(self.offs[-1] + s)
        self.total = self.offs[-1]
        z = zipfile.ZipFile(io.BufferedReader(_Virtual(self), buffer_size=1 << 16))
        self.info = {i.filename: i for i in z.infolist()}

    def read_range(self, a, b):
        """Bytes [a, b) of the concatenated archive."""
        out = []
        for i, u in enumerate(self.urls):
            lo, hi = self.offs[i], self.offs[i + 1]
            s, e = max(a, lo), min(b, hi)
            if s < e:
                out.append(_open(urllib.request.Request(u, headers={"Range": f"bytes={s - lo}-{e - lo - 1}"})))
        return b"".join(out)

    def member(self, name):
        """One file's bytes: local header + data in one request (slack covers the local extra field)."""
        zi = self.info[name]
        n = len(zi.filename.encode("utf-8"))
        buf = self.read_range(zi.header_offset, min(zi.header_offset + 30 + n + zi.compress_size + 1024, self.total))
        sig, *_, nlen, elen = struct.unpack("<IHHHHHIIIHH", buf[:30])
        assert sig == 0x04034B50, f"bad local header for {name}"
        start = 30 + nlen + elen
        if start + zi.compress_size > len(buf):
            buf = self.read_range(zi.header_offset, zi.header_offset + start + zi.compress_size)
        data = buf[start:start + zi.compress_size]
        if zi.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        assert len(data) == zi.file_size and zlib.crc32(data) == zi.CRC, f"crc mismatch for {name}"
        return data


class _Virtual(io.RawIOBase):
    def __init__(self, rz):
        self.rz, self.p = rz, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, o, w=0):
        self.p = o if w == 0 else (self.p + o if w == 1 else self.rz.total + o)
        return self.p

    def tell(self):
        return self.p

    def readinto(self, b):
        n = min(len(b), self.rz.total - self.p)
        if n <= 0:
            return 0
        d = self.rz.read_range(self.p, self.p + n)
        b[:len(d)] = d
        self.p += len(d)
        return len(d)


def page_ids(rz):
    """Page id -> {device: {file: ZipInfo}} for every page in the archive."""
    out = {}
    for name, zi in rz.info.items():
        parts = name.split("/")
        if len(parts) != 3 or not parts[2]:
            continue
        for dkey, dev in DEVICES.items():
            if parts[2].startswith(dev + "-"):
                out.setdefault(parts[1], {}).setdefault(dkey, {})[parts[2][len(dev) + 1:]] = zi
    return out


def eligible(pages):
    """Pages where both devices have every needed file and a non-empty screenshot and tree."""
    ok = []
    for pid, devs in pages.items():
        if all(d in devs and all(f in devs[d] and devs[d][f].file_size > 0 for f in FILES) for d in DEVICES):
            ok.append(pid)
    return sorted(ok)


def fetch_pages(rz, pids, out_dir, workers=16, log=print):
    """Download the needed files of the given pages to out_dir/<page>/<device>-<file>. Skips files present."""
    out_dir = Path(out_dir)
    jobs = []
    for pid in pids:
        for dkey in DEVICES:
            for f in FILES:
                dst = out_dir / pid / f"{dkey}-{f}"
                if not dst.exists():
                    jobs.append((pid, dkey, f, dst))
    stem = next(iter(rz.info)).split("/")[0]
    t0, done, nbytes = time.time(), 0, 0

    def one(j):
        pid, dkey, f, dst = j
        data = rz.member(f"{stem}/{pid}/{DEVICES[dkey]}-{f}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dst)
        return len(data)

    with ThreadPoolExecutor(workers) as ex:
        for n in ex.map(one, jobs):
            done += 1
            nbytes += n
            if done % 400 == 0 or done == len(jobs):
                log(f"  {done}/{len(jobs)} files, {nbytes / 1e6:.0f} MB, {time.time() - t0:.0f}s")
    return {"files": len(jobs), "bytes": nbytes, "sec": round(time.time() - t0, 1)}


def read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- page -> the project's node format
VIEWPORT = {"phone": (390, 844, 3), "desktop": (1280, 720, 1)}   # CSS width, height, device pixel ratio
MIN_INSIDE = 0.5            # an element is on screen when at least half its box area is inside the viewport
CROP_FRAME = {"phone": (540, 960), "desktop": (960, 540)}        # crops.SIZE, turned for the landscape viewport


def _num(v):
    return float(v) if v is not None else 0.0


def _prop(node, name):
    for p in node.get("properties", []):
        if p.get("name") == name:
            return p.get("value", {}).get("value")
    return None


def page_tables(page_dir, device, page_num, flag_fill):
    """One page on one device -> (g, raw, info) in the shapes the provenance builder takes.

    raw (the page's own metadata, like the Android view hierarchy): every accessibility node that Chrome does
      not mark ignored and that has a box with positive area; a node's parent is its nearest such ancestor.
      Columns as data/processed/nodes.parquet (pre-order node_idx, parent_idx, box, depth, child count,
      class name, flags) + label -1 / is_labelled False (no human labels exist on the web).
    g (the true tree): elements as WebUI defines them (paper section 3.2; generate_dataset_web.py): every leaf of
      raw, moved up through parents that have exactly one child (the element takes the top-most box);
      kept when at least MIN_INSIDE of its area is in the viewport. Containers = the distinct sets of on-screen
      elements under raw nodes; sets of one element and the set of all elements (the page itself) are not
      containers, as the virtual root is not one. A container's parent is the smallest container holding it.
    Class names have no web counterpart and are left empty; flags: clickable and focusable = the node's
      `focusable` property, enabled = not `disabled`, selected = `selected`, visible_to_user = box overlaps
      the viewport; long_clickable, scrollable_v/h and draw (no web counterpart) = flag_fill (Android train mode).
    """
    import numpy as np
    import pandas as pd
    from uistruct.data.features import FLAGS
    W, H, _ = VIEWPORT[device]
    d = Path(page_dir)
    ax = read_json_gz(d / f"{device}-axtree.json.gz")["nodes"]
    bb = read_json_gz(d / f"{device}-bb.json.gz")
    byid = {n["nodeId"]: n for n in ax}

    def box(n):
        b = bb.get(str(n.get("backendDOMNodeId"))) if n.get("backendDOMNodeId") is not None else None
        if not b or _num(b.get("width")) <= 0 or _num(b.get("height")) <= 0:
            return None
        x, y = _num(b["x"]), _num(b["y"])
        return (x / W, y / H, (x + _num(b["width"])) / W, (y + _num(b["height"])) / H)

    roots = [n for n in ax if "parentId" not in n or n["parentId"] not in byid]
    kept, order, par, boxes = {}, [], {}, {}
    for r in roots:                         # pre-order over the full tree; ignored / boxless nodes are skipped
        stack = [(r["nodeId"], None)]
        while stack:
            nid, kp = stack.pop()
            n = byid.get(nid)
            if n is None or nid in kept:
                continue
            b = None if n.get("ignored") else box(n)
            if b is not None:
                kept[nid] = len(order)
                order.append(nid)
                par[nid] = kp
                boxes[nid] = b
                kp = nid
            for c in reversed(n.get("childIds", [])):
                stack.append((c, kp))
    if not order:
        return None, None, {"reason": "no boxed node"}
    children = {}
    for nid in order:
        if par[nid] is not None:
            children.setdefault(par[nid], []).append(nid)
    depth = {}
    for nid in order:
        depth[nid] = 0 if par[nid] is None else depth[par[nid]] + 1

    def overlap(b):
        return min(b[2], 1) > max(b[0], 0) and min(b[3], 1) > max(b[1], 0)

    rows = []
    for nid in order:
        n, b = byid[nid], boxes[nid]
        foc = bool(_prop(n, "focusable"))
        fl = {f: flag_fill[f] for f in FLAGS}
        fl.update(clickable=foc, focusable=foc, enabled=not bool(_prop(n, "disabled")),
                  selected=bool(_prop(n, "selected")), visible_to_user=overlap(b))
        rows.append({"screen_id": page_num, "node_idx": kept[nid],
                     "parent_idx": -1 if par[nid] is None else kept[par[nid]], "box_ok": overlap(b),
                     "nx0": b[0], "ny0": b[1], "nx1": b[2], "ny1": b[3], "depth": depth[nid],
                     "n_children": len(children.get(nid, [])), "class_name": "", **fl,
                     "role": (n.get("role") or {}).get("value", ""), "label": -1, "is_labelled": False})
    raw = pd.DataFrame(rows)
    for c in ["nx0", "ny0", "nx1", "ny1"]:
        raw[c] = raw[c].astype(np.float32)

    # elements: leaves moved up through single-child parents (never into the root)
    elems = set()
    for nid in order:
        if nid in children:
            continue
        e = nid
        while par[e] is not None and par[par[e]] is not None and len(children[par[e]]) == 1:
            e = par[e]
        elems.add(e)

    def inside(b):
        a = (b[2] - b[0]) * (b[3] - b[1])
        iw = max(0.0, min(b[2], 1) - max(b[0], 0))
        ih = max(0.0, min(b[3], 1) - max(b[1], 0))
        return a > 0 and iw * ih / a >= MIN_INSIDE
    on = sorted((e for e in elems if inside(boxes[e])), key=kept.get)
    info = {"raw_nodes": len(order), "elements_all": len(elems), "elements_on_screen": len(on)}
    if len(on) < 2:
        return None, None, {**info, "reason": "fewer than 2 elements on screen"}
    onset = set(on)
    under = {}
    for e in on:
        a = par[e]
        while a is not None:
            under.setdefault(a, set()).add(e)
            a = par[a]
    sets = {}
    for nid in order:                       # pre-order: the outermost node of equal sets comes first and keeps it
        if nid in under and nid not in onset:
            s = frozenset(under[nid])
            if 1 < len(s) < len(on) and s not in sets:
                sets[s] = nid
    cont = list(sets.items())

    def smallest_holder(test, size):
        best = None
        for s, nid in cont:
            if len(s) > size and test(s) and (best is None or len(s) < len(best[0])):
                best = (s, nid)
        return -1 if best is None else kept[best[1]]
    gpar = {kept[e]: smallest_holder(lambda s, e=e: e in s, 1) for e in on}
    for s, nid in cont:
        gpar[kept[nid]] = smallest_holder(lambda s2, s=s: s <= s2, len(s))
    g = raw.set_index("node_idx").loc[sorted(gpar)].reset_index()
    g = g[["screen_id", "node_idx", "nx0", "ny0", "nx1", "ny1"]].copy()
    g["gt_parent"] = g.node_idx.map(gpar).astype(np.int64)
    g["label"] = 1                          # no type truth on the web; never an input, type head not scored
    info.update(true_containers=len(cont))
    return g, raw, info


def web_crops(path, boxes, device, size, margin=8):
    """As crops.screen_crops (box + margin on a resized screenshot, squashed to a square), but the screenshot is
    resized to CROP_FRAME[device], so a landscape desktop page is not blanked (crops.py blanks landscape images,
    whose Android boxes did not align)."""
    import numpy as np
    from PIL import Image
    from uistruct.data.crops import crop_box
    out = np.zeros((len(boxes), size, size, 3), np.uint8)
    fw, fh = CROP_FRAME[device]
    im = Image.open(path).convert("RGB").resize((fw, fh), Image.BILINEAR)
    for i, b in enumerate(boxes):
        pb = crop_box(b, margin, (fw, fh))
        if pb is None:
            continue
        out[i] = np.asarray(im.crop(pb).resize((size, size), Image.BILINEAR))
    return out
