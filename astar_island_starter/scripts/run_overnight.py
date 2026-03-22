import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

#python scripts/run_overnight.py

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "overnight_logs"
LOG_DIR.mkdir(exist_ok=True)

PYTHON = sys.executable

POLL_SECONDS = 3600
QUERIES_PER_ROUND = 50
MODEL_PATH = ROOT / "models" / "prior_model.joblib"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "overnight.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd):
    log("RUN " + " ".join(map(str, cmd)))
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.stdout:
        with open(LOG_DIR / "overnight.log", "a", encoding="utf-8") as f:
            f.write(result.stdout)
    if result.stderr:
        with open(LOG_DIR / "overnight.log", "a", encoding="utf-8") as f:
            f.write(result.stderr)
    return result


def get_rounds():
    import requests

    base = os.environ.get("AINM_BASE_URL", "https://api.ainm.no")
    token = os.environ["AINM_TOKEN"]

    r = requests.get(
        f"{base}/astar-island/rounds",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def get_active_round():
    rounds = get_rounds()
    active = [r for r in rounds if r.get("status") == "active"]
    if not active:
        return None
    active.sort(key=lambda r: r.get("started_at", ""))
    return active[-1]


def already_processed(round_id: str) -> bool:
    marker = LOG_DIR / f"done_{round_id}.json"
    return marker.exists()


def mark_processed(round_id: str, payload: dict):
    marker = LOG_DIR / f"done_{round_id}.json"
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    log("Overnight runner started")
    while True:
        try:
            active = get_active_round()
            if not active:
                log("No active round")
                time.sleep(POLL_SECONDS)
                continue

            round_id = active["id"]
            round_number = active.get("round_number")
            closes_at = active.get("closes_at")

            if already_processed(round_id):
                log(f"Round already processed: #{round_number} {round_id}")
                time.sleep(POLL_SECONDS)
                continue

            log(f"Active round found: #{round_number} {round_id} closes_at={closes_at}")

            cmd = [
                PYTHON,
                "scripts/play_round.py",
                "--model",
                str(MODEL_PATH),
                "--queries",
                str(QUERIES_PER_ROUND),
                "--submit",
            ]
            result = run_cmd(cmd)

            mark_processed(
                round_id,
                {
                    "round_id": round_id,
                    "round_number": round_number,
                    "returncode": result.returncode,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            if result.returncode == 0:
                log(f"Finished round #{round_number} successfully")
            else:
                log(f"Round #{round_number} failed with code {result.returncode}")

        except KeyboardInterrupt:
            log("Stopped by user")
            break
        except Exception as e:
            log(f"ERROR: {e!r}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()