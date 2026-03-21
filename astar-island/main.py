import requests

BASE = "https://api.ainm.no"
WS_URL = open(".ws_url").readline() 
# Option 1: Cookie-based auth
session = requests.Session()
session.cookies.set("access_token", WS_URL)

# Option 2: Bearer token auth
# session = requests.Session()
# session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"

rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)

if active:
    round_id = active["id"]
    print(f"Active round: {active['round_number']}")


detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

width = detail["map_width"]      # 40
height = detail["map_height"]    # 40
seeds = detail["seeds_count"]    # 5
print(f"Round: {width}x{height}, {seeds} seeds")

for i, state in enumerate(detail["initial_states"]):
    grid = state["grid"]           # height x width terrain codes
    settlements = state["settlements"]  # [{x, y, has_port, alive}, ...]
    print(f"Seed {i}: {len(settlements)} settlements")




import numpy as np
 
for seed_idx in range(seeds):
    break
    prediction = np.full((height, width, 6), 1/6)  # uniform baseline
    
    for runs in range(10):
        result = session.post(f"{BASE}/astar-island/simulate", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "viewport_x": 10,
        "viewport_y": 5,
        "viewport_w": 15,
        "viewport_h": 15,
        }).json()
        grid = result["grid"]                # 15x15 terrain after simulation
        settlements = result["settlements"]  # settlements in viewport with full stats
        viewport = result["viewport"]        # {x, y, w, h}

    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    print(f"Seed {seed_idx}: {resp.status_code}")
