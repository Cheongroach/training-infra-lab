"""Save one described Grafana panel and summarize only readable data points."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from collect_preflight import CLI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--panel", type=int, required=True)
    args = parser.parse_args()
    url = json.loads((args.folder / "capture-url.json").read_text())["url"]
    parts = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parts.query) if key not in ("from", "to")]
    status = json.loads((args.folder.parent / "status.json").read_text())
    def milliseconds(value):
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    starts = [milliseconds(p["start_at"]) for p in status["pods"] if p.get("start_at")]
    stops = [milliseconds(p["stop_at"]) for p in status["pods"] if p.get("stop_at")]
    query.append(("from", str(min(starts) - 60_000) if starts else "now-30m"))
    query.append(("to", str(max(stops) + 60_000) if status["terminal"] and stops
                  else str(int(time.time() * 1000))))
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    body = {"url": url, "panel_id": args.panel}
    prefix = args.folder / f"panel-{args.panel}"
    prefix.with_suffix(".request.json").write_text(json.dumps(body, indent=2))
    response = subprocess.run(
        [CLI, "--control-plane", "cn", "grafana", "panel-data", "--json", json.dumps(body)],
        capture_output=True, timeout=60,
    )
    prefix.with_suffix(".json").write_bytes(response.stdout)
    prefix.with_suffix(".stderr").write_bytes(response.stderr)
    response.check_returncode()
    series = []

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("dps"), dict):
                points = [[float(stamp), number] for stamp, number in value["dps"].items()
                          if isinstance(number, (float, int)) and not isinstance(number, bool)]
                if points:
                    series.append({"metric": value.get("metric"), "tags": value.get("tags"),
                                   "query_metadata": {key: item for key, item in value.items()
                                                      if key not in ("metric", "tags", "dps")},
                                   "points": points})
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(json.loads(response.stdout))
    summary = {"panel_id": args.panel, "readable_series": len(series),
               "readable_points": sum(len(item["points"]) for item in series),
               "series": series, "numerical_evidence": bool(series),
               "observed_at": datetime.now(timezone.utc).isoformat(),
               "query": body,
               "response_sha256": hashlib.sha256(response.stdout).hexdigest(),
               "window_scope": "Actual Pod start through last terminal stop, with one-minute margins"
                               if status["terminal"] else "Actual Pod start through capture time"}
    prefix.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({key: summary[key] for key in
                      ("panel_id", "readable_series", "readable_points", "numerical_evidence")}))


if __name__ == "__main__":
    main()
