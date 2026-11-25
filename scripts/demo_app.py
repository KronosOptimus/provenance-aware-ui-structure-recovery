"""Local demo app: pick a held-out test screen and a metadata level, see the recovered tree, the trust score of
every element and, for every container, whether it was read from the metadata or inferred by the model.

  python scripts/demo_app.py            then open http://127.0.0.1:8050 (opened automatically)

Python standard library web server, no extra packages. Model files are expected in data/models/ (copied from
the training run's output folder). The pipeline itself is src/uistruct/demo.py.
"""
import argparse
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import yaml

from uistruct.demo import LEVEL_TEXT, LEVELS, SYSTEM, Demo

PAGE = Path(__file__).with_name("demo_page.html")
TEST_RUN = "results/runs/colab_trust_score/trust_score_test_run1/metrics.json"


def test_set_numbers():
    """Whole-test-set container F1 (raw truth) of the demo system per level, from the test run."""
    s = json.load(open(TEST_RUN))["structure"]
    out = {}
    for lv in LEVELS:
        name = "fusion_full_s0" if lv == "absent" else SYSTEM       # absent: nothing removed (plan section 7)
        out[lv] = round(s[f"{name}|{lv}|raw"]["container_f1"]["mean"], 3)
    return out


def make_handler(demo, cfg, valid_ids, numbers):
    lock = threading.Lock()
    screens = [{**s, "id": int(s["id"])} for s in cfg["screens"]]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/info":
                body = {"screens": screens, "levels": [{"id": lv, "text": LEVEL_TEXT[lv]} for lv in LEVELS],
                        "test_set": numbers, "t_c": demo.t_c, "t_e": demo.t_e, "n_test": len(valid_ids)}
                return self.send(200, json.dumps(body).encode(), "application/json")
            if u.path == "/api/run":
                try:
                    sid, lv = int(q["sid"][0]), q["level"][0]
                except (KeyError, ValueError):
                    return self.send(400, b'{"error": "bad request"}', "application/json")
                if sid not in valid_ids or lv not in LEVELS:
                    return self.send(404, b'{"error": "not a test screen"}', "application/json")
                with lock:                                    # one model, one run at a time
                    r = demo.run(sid, lv)
                return self.send(200, json.dumps(r).encode(), "application/json")
            m = re.fullmatch(r"/img/(\d+)\.jpg", u.path)
            if m and int(m.group(1)) in valid_ids:
                return self.send(200, (demo.img_dir / f"{m.group(1)}.jpg").read_bytes(), "image/jpeg")
            return self.send(404, b"not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_screens.yaml")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    print("Loading the model and the test screens (about 15 seconds)...", flush=True)
    demo = Demo()
    el = pd.read_parquet(demo.data_dir / "elements.parquet", columns=["screen_id", "split_clean", "is_valid"])
    valid_ids = set(el[(el.split_clean == "test") & el.is_valid].screen_id.unique().tolist())   # as load_test_tables
    del el
    demo.load([s["id"] for s in cfg["screens"]])
    for s in cfg["screens"]:                      # warm-up: first run of each demo screen
        demo.run(int(s["id"]), "real")
    url = f"http://{cfg['host']}:{cfg['port']}"
    srv = ThreadingHTTPServer((cfg["host"], cfg["port"]), make_handler(demo, cfg, valid_ids, test_set_numbers()))
    print(f"Demo ready: {url}   (press Ctrl+C here to stop)", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
