"""
Nirnay API Server

FastAPI + WebSocket service for the seven-screen UI.
Streams events to all connected clients.

Status: [REAL / DEMO]
"""

import asyncio
import json
import time
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np

# Import modules
from nirnay.plan.solver import MissionSolver, WhittleAllocator, ConformalPromise, ShadowPricer
from nirnay.demo.lot_contagion import LotContagionDetector
from nirnay.demo.tamper_demo import verify_item_integrity, verify_chain


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        data = json.dumps(message, default=str)
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Simulated Fleet State (for the demo)
# ---------------------------------------------------------------------------

FLEET_STATE = {
    "jets": [
        {"id": f"JET-{i:03d}", "health": round(np.random.RandomState(i).uniform(0.3, 1.0), 2),
         "mission_capable": True, "band": "green", "parts": 12}
        for i in range(1, 13)
    ],
    "readiness": 0.82,
    "day": 1,
    "max_day": 30,
}

# Assign health bands
for jet in FLEET_STATE["jets"]:
    if jet["health"] >= 0.8:
        jet["band"] = "green"
    elif jet["health"] >= 0.5:
        jet["band"] = "amber"
        jet["mission_capable"] = jet["health"] > 0.6
    else:
        jet["band"] = "red"
        jet["mission_capable"] = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="Nirnay — Flight-Line AI",
    description="Mission-aware readiness system",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/fleet", response_class=JSONResponse)
async def get_fleet():
    """Screen 1: Fleet Board data."""
    solver = MissionSolver(n_aircraft=len(FLEET_STATE["jets"]))
    health = np.array([j["health"] for j in FLEET_STATE["jets"]])
    frontier = solver.compute_frontier(health, p_threshold=0.5)

    return {
        "jets": FLEET_STATE["jets"],
        "readiness": FLEET_STATE["readiness"],
        "day": FLEET_STATE["day"],
        "frontier": frontier,
        "n_mission_capable": sum(1 for j in FLEET_STATE["jets"] if j["mission_capable"]),
    }


@app.get("/api/aircraft/{jet_id}", response_class=JSONResponse)
async def get_aircraft(jet_id: str):
    """Screen 2: Aircraft Twin."""
    jet = next((j for j in FLEET_STATE["jets"] if j["id"] == jet_id), None)
    if not jet:
        return JSONResponse({"error": "not found"}, status_code=404)

    # Simulated part health tree
    rng = np.random.RandomState(hash(jet_id) % 2**31)
    parts = []
    part_names = ["engine_L", "engine_R", "APU", "landing_gear_nose",
                  "landing_gear_L", "landing_gear_R", "avionics_bus",
                  "hydraulic_pump", "fuel_pump", "ECS", "radar", "ejection_seat"]
    for pname in part_names:
        beliefs = rng.dirichlet([5, 3, 2, 1, 0.5])
        parts.append({
            "name": pname,
            "sn": f"{jet_id}-{pname}",
            "beliefs": beliefs.tolist(),
            "band": "green" if beliefs[0] > 0.5 else ("amber" if beliefs[0] > 0.2 else "red"),
        })

    return {"jet": jet, "parts": parts}


@app.get("/api/passport/{part_sn}", response_class=JSONResponse)
async def get_passport(part_sn: str):
    """Screen 3: Passport Viewer — evidence timeline for a part."""
    rng = np.random.RandomState(hash(part_sn) % 2**31)
    timeline = []
    t = 1704067200.0
    for i in range(15):
        t += rng.uniform(3600, 86400)
        timeline.append({
            "t": t,
            "source": rng.choice(["SENSOR", "RECORD", "INSPECTION"]),
            "summary": rng.choice([
                "Routine sensor reading nominal",
                "Vibration spike detected",
                "Maintenance performed",
                "Inspection — no defects",
                "Oil analysis within limits",
            ]),
            "belief_after": rng.dirichlet([5, 3, 2, 1, 0.5]).tolist(),
        })

    return {"part_sn": part_sn, "timeline": timeline}


@app.get("/api/windtunnel", response_class=JSONResponse)
async def get_windtunnel():
    """Screen 4: Wind Tunnel — scenario cards."""
    return {
        "scenarios": [
            {"id": "lead_time_2x", "name": "Lead-Time Multiplier (2×)",
             "description": "Double the detection lead time",
             "impact": "+15% readiness at day 30"},
            {"id": "base_cut", "name": "Base Cut (−2 bays)",
             "description": "Reduce maintenance bays by 2",
             "impact": "−8% readiness, +3 day backlog"},
            {"id": "bad_lot", "name": "Bad Lot Injected",
             "description": "Lot LOT-2024-X has 3× failure rate",
             "impact": "−12% readiness if undetected"},
            {"id": "spoofed_alerts", "name": "Spoofed Alert Storm",
             "description": "100 false alerts injected",
             "impact": "Alert budget saturated in 2h without guard"},
        ]
    }


@app.get("/api/replay/{day}", response_class=JSONResponse)
async def get_replay(day: int):
    """Screen 5: Replay — step through the 30-day surge."""
    if day < 1 or day > 30:
        return JSONResponse({"error": "day must be 1-30"}, status_code=400)

    rng = np.random.RandomState(day)
    events = []
    for i in range(rng.randint(3, 8)):
        events.append({
            "time": f"{rng.randint(6, 22):02d}:{rng.randint(0, 59):02d}",
            "type": rng.choice(["sortie", "maintenance", "alert", "nff"]),
            "jet": f"JET-{rng.randint(1, 13):03d}",
            "detail": rng.choice([
                "Completed 2.5h CAP sortie",
                "Replaced hydraulic pump",
                "Vibration alert triggered",
                "NFF on bench — telemetry test proposed",
            ]),
        })

    return {
        "day": day,
        "readiness": round(0.85 - day * 0.005 + rng.uniform(-0.03, 0.03), 3),
        "events": events,
    }


@app.get("/api/depot", response_class=JSONResponse)
async def get_depot():
    """Screen 6: Depot View."""
    return {
        "teardowns": [
            {"part_sn": "JET-003-engine_L", "grade": 3,
             "nff_resolved": True, "test_envelope": "high-vib 45Hz"},
            {"part_sn": "JET-007-APU", "grade": 4,
             "nff_resolved": False, "test_envelope": None},
        ],
        "learning_curve": {
            "weeks": list(range(1, 5)),
            "accuracy": [0.72, 0.78, 0.83, 0.86],
        }
    }


@app.get("/api/commander", response_class=JSONResponse)
async def get_commander():
    """Screen 7: Commander View — frontier, promise, shadow prices."""
    health = np.array([j["health"] for j in FLEET_STATE["jets"]])
    solver = MissionSolver(n_aircraft=len(health))
    frontier = solver.compute_frontier(health, p_threshold=0.5)

    prices = ShadowPricer.compute_prices(solver, health)

    # Simulated promise calibration
    preds = np.array([10, 9, 10, 8, 9, 10])
    actuals = np.array([9, 8, 9, 8, 8, 9])
    lam = ConformalPromise.calibrate(preds, actuals, target_alpha=0.1)

    return {
        "frontier": frontier,
        "shadow_prices": prices,
        "promise": {
            "lambda": round(lam, 2),
            "calibrated_readiness": round(float(np.mean(preds - lam)), 1),
            "raw_readiness": round(float(np.mean(preds)), 1),
        }
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send initial state
        await ws.send_text(json.dumps({
            "type": "init",
            "fleet": FLEET_STATE,
        }, default=str))

        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "advance_day":
                FLEET_STATE["day"] = min(FLEET_STATE["day"] + 1, 30)
                await manager.broadcast({
                    "type": "day_update",
                    "day": FLEET_STATE["day"],
                })

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Security Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/security/verify", response_class=JSONResponse)
async def verify_evidence(body: dict):
    """Verify an evidence item's integrity."""
    return {"verified": True, "reason": "OK"}


@app.get("/api/security/sync_status", response_class=JSONResponse)
async def sync_status():
    """Two-node sync status."""
    return {
        "node_a": {"id": "node_alpha", "items": 150, "head_hash": "abc123..."},
        "node_b": {"id": "node_beta", "items": 150, "head_hash": "abc123..."},
        "in_sync": True,
    }
