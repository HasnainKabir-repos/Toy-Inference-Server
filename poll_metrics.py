"""
Poll /metrics every `interval` seconds and append to a CSV, so you have a
queue_depth / active_batch_size timeline to plot after each `hey` run --
the CPU/local analog of the Grafana panels you used in Week 1.
 
Usage:
    python poll_metrics.py concurrency_10.csv --interval 0.25
 
Run this in its own terminal, start it just before launching `hey`,
and Ctrl+C it just after `hey` finishes.
"""
import csv
import sys
import time
import requests
METRICS_URL = "http://localhost:8000/metrics"

def main(out_path, interval):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "queue_depth", "active_batch_size", "completed"])
        f.flush()
 
        print(f"Polling {METRICS_URL} every {interval}s -> {out_path}")
        while True:
            try:
                r = requests.get(METRICS_URL, timeout=1)
                m = r.json()
                writer.writerow([
                    time.time(),
                    m["queue_depth"],
                    m["active_batch_size"],
                    m["completed"],
                ])
                f.flush()
            except Exception as e:
                print(f"poll failed: {e}")
            time.sleep(interval)
 
 
if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "metrics_log.csv"
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    main(out_path, interval)