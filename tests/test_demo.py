"""Demo pipeline (src/uistruct/demo.py): same result as the provenance test run, and a well-formed tree.
Needs the model files in data/models/ and the processed tables (skipped otherwise)."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDS = [ROOT / "data/models/fusion_full_s0/model.pt", ROOT / "data/models/element_types_cnn/model3_lr2.pt",
         ROOT / "data/processed/nodes.parquet"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDS), reason="model files or tables not on this machine")


@pytest.fixture(scope="module")
def demo():
    from uistruct.demo import Demo
    return Demo(root=ROOT, workers=4)


def test_matches_test_run(demo):
    from uistruct.demo import SYSTEM
    df = pd.read_csv(ROOT / "results/runs/colab_trust_score/trust_score_test_run1/per_screen.csv.gz",
                     usecols=["screen_id", "system", "level", "truth", "container_f1", "n_removed"])
    df = df[df.truth == "raw"]
    for sid in (218, 3680):
        for lv in ("real", "heavy", "absent"):
            r = demo.run(sid, lv)
            sysname = "fusion_full_s0" if lv == "absent" else SYSTEM
            t = df[(df.screen_id == sid) & (df.level == lv) & (df.system == sysname)].iloc[0]
            assert abs(r["answer"]["container_f1"] - t.container_f1) < 1e-3
            assert r["summary"]["removed"] == (0 if lv == "absent" else t.n_removed)


def test_tree_well_formed(demo):
    r = demo.run(2362, "heavy")
    cont = {c["id"]: c for c in r["containers"]}
    for c in r["containers"]:
        assert c["parent"] is None or c["parent"] in cont
        assert c["source"] in ("metadata", "inferred")
    for e in r["elements"]:
        if e["removed"]:
            assert e["parent"] is None and e["trust"] < r["t_e"]
        else:
            assert e["parent"] is None or (e["parent"] in cont and e["id"] in cont[e["parent"]]["members"])
    a = demo.run(2362, "absent")
    assert not a["has_meta"] and a["summary"]["read_from_metadata"] == 0
    assert all(e["trust"] is None and not e["removed"] for e in a["elements"])
