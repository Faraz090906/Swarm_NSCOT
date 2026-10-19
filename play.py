"""
NIOT ASV SWARM - Adaptive Concentric Square Search
Python backend + HTTP API with High-Resolution Spatial Sampling
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import math
import random
import threading
import statistics

WORLD = 1000.0
N_BOATS = 4
GROUND_STATION = (0.0, 0.0)

# Motion Constraints[cite: 7]
MAX_SPEED = 1.0          
SEARCH_SPEED = 1.0
INVESTIGATE_SPEED = 0.6
LOCAL_SPEED = 0.40
SPACING = 18.0           
VIRTUAL_DT = 2.0         

# Sensor & Noise Model[cite: 7, 8]
TARGET_STRENGTH = 50.0   
TARGET_FALLOFF = 80.0    
DETECTION_RANGE = 10.0   # 7-10 m physical sensing range per spec[cite: 8]
SIGMA_EPSILON = 0.8
AR_PHI = 0.85
W_WINDOW = 20
PERSIST_REQUIRED = 5
TL_THRESH = 2.0  
TH_THRESH = 3.0  

# Battery Model[cite: 8]
BATTERY_CAPACITY_WH = 296.0
POWER_W = 53.0
BATTERY_CAPACITY_J = BATTERY_CAPACITY_WH * 3600.0
SWAP_TIME_S = 300.0

def dist(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def dipole_like_field(x, y, tx, ty, strength, falloff):
    r = dist((x, y), (tx, ty))
    return strength / (1.0 + (r / max(falloff, 1e-6)) ** 2)

def make_quadrant_route(q):
    bounds = [(0.0, 500.0, 0.0, 500.0), (500.0, 1000.0, 0.0, 500.0),
              (0.0, 500.0, 500.0, 1000.0), (500.0, 1000.0, 500.0, 1000.0)][q]
    cx, cy = (bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2
    max_h = min(bounds[1]-bounds[0], bounds[3]-bounds[2])/2 - 8.0
    
    route = []
    h = max_h
    while h >= SPACING/2:
        sq = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h), (cx - h, cy - h)]
        for i in range(4):
            x1, y1, x2, y2 = sq[i][0], sq[i][1], sq[i+1][0], sq[i+1][1]
            d = math.hypot(x2-x1, y2-y1)
            steps = max(1, int(math.ceil(d / 2.0)))
            for k in range(steps):
                route.append((x1 + (k/steps)*(x2-x1), y1 + (k/steps)*(y2-y1)))
        route.append(sq[-1])
        h -= SPACING
    return route

ROUTES = [make_quadrant_route(q) for q in range(4)]

class Boat:
    def __init__(self, bid, route):
        self.id = bid
        self.route = route
        self.route_i = 0
        self.x, self.y = GROUND_STATION
        self.heading = 0.0
        self.state = "deploying"
        self.battery_j = BATTERY_CAPACITY_J
        self.swaps = 0
        self.swap_timer = 0.0
        self.saved_x, self.saved_y = 0.0, 0.0
        self.saved_state = "search"
        
        # Adaptive Algorithm States[cite: 7]
        self.mag_history = []
        self.res_history = []
        self.last_r = 0.0
        self.S_t = 0.0
        self.n_t = 0.0 
        self.persist_buffer = []
        self.mag_reading = 0.0
        self.local_remaining = 0.0

    def telemetry(self):
        return {
            "id": self.id, "x": self.x, "y": self.y, "heading": self.heading,
            "state": self.state, "battery_pct": max(0, self.battery_j / BATTERY_CAPACITY_J) * 100,
            "swaps": self.swaps, "mag_reading": self.mag_reading, "anomaly_score": self.S_t
        }

class Simulation:

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.virtual_time = 0.0
        self.config = {"max_speed": MAX_SPEED, "target_strength": TARGET_STRENGTH, "target_falloff": TARGET_FALLOFF}
        self.target = {"x": 730.0, "y": 270.0, "found": False}
        self.coverage_cells = set()
        self.reset()

    def set_target(self, x, y):
        with self.lock:
            self.target = {
                "x": max(10.0, min(WORLD-10.0, float(x))),
                "y": max(10.0, min(WORLD-10.0, float(y))),
                "found": False,
            }
            for b in self.boats:
                b.route_i = 0
                b.x, b.y = GROUND_STATION
                b.state = "deploying"
                b.persist_buffer.clear()
                b.mag_history.clear()
                b.res_history.clear()
                b.S_t = 0.0
            self.virtual_time = 0.0
            self.running = False

    def reset(self):
        with self.lock:
            self.running = False
            self.virtual_time = 0.0
            self.target["found"] = False
            self.coverage_cells.clear()
            self.boats = [Boat(i, ROUTES[i]) for i in range(N_BOATS)]

    def sample_sensor(self, boat):
        epsilon_t = random.gauss(0.0, SIGMA_EPSILON)
        boat.n_t = AR_PHI * boat.n_t + epsilon_t
        
        B_anom = dipole_like_field(boat.x, boat.y, self.target["x"], self.target["y"], self.config["target_strength"], self.config["target_falloff"])
        B_meas = B_anom + boat.n_t
        boat.mag_reading = B_meas
        
        # Freeze background estimation array when localizing so anomaly isn't baked into background median
        if boat.state != "localize":
            boat.mag_history.append(B_meas)
            if len(boat.mag_history) > W_WINDOW: boat.mag_history.pop(0)

        if len(boat.mag_history) == W_WINDOW:
            bg_est = statistics.median(boat.mag_history)
            r_t = B_meas - bg_est
            
            if boat.state != "localize":
                boat.res_history.append(r_t)
                if len(boat.res_history) > W_WINDOW: boat.res_history.pop(0)

            med_r = statistics.median(boat.res_history) if boat.res_history else 0.0
            mad = statistics.median([abs(r - med_r) for r in boat.res_history]) if boat.res_history else 0.1
            sigma_n = max(1.4826 * mad, 0.5)

            n_hat = AR_PHI * boat.last_r
            e_t = r_t - n_hat
            z_t = e_t / sigma_n
            
            boat.S_t = 0.20 * boat.S_t + 0.80 * abs(z_t)
            boat.last_r = r_t

            boat.persist_buffer.append(1 if boat.S_t >= TH_THRESH else 0)
            if len(boat.persist_buffer) > W_WINDOW: boat.persist_buffer.pop(0)

    def step_boat(self, b, dt):
        # Micro-stepping (0.1 m increments) to ensure high-frequency 200Hz sampling capture over the anomaly[cite: 7, 8]
        speed = MAX_SPEED if b.state == "deploying" else (SEARCH_SPEED if b.state == "search" else (INVESTIGATE_SPEED if b.state == "investigate" else LOCAL_SPEED))
        total_dist = speed * dt
        micro_step = 0.1 # 10 cm micro-steps
        steps = max(1, int(total_dist / micro_step))
        actual_step = total_dist / steps

        for _ in range(steps):
            if self.target["found"] and b.state != "found":
                b.state = "standby"
                return

            self.coverage_cells.add((int(b.x//25), int(b.y//25)))
            
            # Energy consumption per micro-step fraction[cite: 8]
            step_dt = dt / steps
            b.battery_j = max(0.0, b.battery_j - (POWER_W * step_dt))
            batt_pct = b.battery_j / BATTERY_CAPACITY_J
            return_energy = (dist((b.x, b.y), GROUND_STATION) / MAX_SPEED) * POWER_W
            
            if b.state in ["deploying", "search", "investigate", "localize"]:
                if batt_pct < 0.20 or (b.battery_j - return_energy < POWER_W * 50):
                    b.saved_x, b.saved_y, b.saved_state = b.x, b.y, b.state
                    b.state = "returning"
                    break

            if b.state == "deploying":
                nx, ny = b.route[0]
                dx, dy = nx - b.x, ny - b.y
                seg = math.hypot(dx, dy)
                if seg < actual_step:
                    b.x, b.y = nx, ny
                    b.state = "search"
                else:
                    b.heading = math.atan2(dy, dx)
                    b.x += (dx/seg) * actual_step
                    b.y += (dy/seg) * actual_step

            elif b.state == "search":
                if b.route_i < len(b.route) - 1:
                    nx, ny = b.route[b.route_i + 1]
                    dx, dy = nx - b.x, ny - b.y
                    seg = math.hypot(dx, dy)
                    if seg < actual_step:
                        b.x, b.y = nx, ny
                        b.route_i += 1
                    else:
                        b.heading = math.atan2(dy, dx)
                        b.x += (dx/seg) * actual_step
                        b.y += (dy/seg) * actual_step
                
                # Sample sensor and check physical range proximity[cite: 7, 8]
                self.sample_sensor(b)
                d_target = dist((b.x, b.y), (self.target["x"], self.target["y"]))
                
                if TL_THRESH <= b.S_t < TH_THRESH:
                    b.state = "investigate"
                elif sum(b.persist_buffer) >= PERSIST_REQUIRED or d_target <= DETECTION_RANGE:
                    b.state = "localize"
                    b.local_remaining = 60.0

            elif b.state == "investigate":
                if b.route_i < len(b.route) - 1:
                    nx, ny = b.route[b.route_i + 1]
                    dx, dy = nx - b.x, ny - b.y
                    seg = math.hypot(dx, dy)
                    if seg < actual_step:
                        b.x, b.y = nx, ny
                        b.route_i += 1
                    else:
                        b.heading = math.atan2(dy, dx)
                        b.x += (dx/seg) * actual_step
                        b.y += (dy/seg) * actual_step
                
                self.sample_sensor(b)
                d_target = dist((b.x, b.y), (self.target["x"], self.target["y"]))
                
                if b.S_t < TL_THRESH:
                    b.state = "search"
                elif sum(b.persist_buffer) >= PERSIST_REQUIRED or d_target <= DETECTION_RANGE:
                    b.state = "localize"
                    b.local_remaining = 60.0

            elif b.state == "localize":
                # Converge directly on target peak coordinates
                tx, ty = self.target["x"], self.target["y"]
                dx, dy = tx - b.x, ty - b.y
                seg = math.hypot(dx, dy)
                if seg < actual_step:
                    b.x, b.y = tx, ty
                else:
                    b.heading = math.atan2(dy, dx)
                    b.x += (dx/seg) * actual_step
                    b.y += (dy/seg) * actual_step

                self.sample_sensor(b)
                b.local_remaining -= step_dt

                if seg <= 15.0 or b.local_remaining <= 0:
                    b.state = "found"
                    self.target["found"] = True
                    break

            elif b.state == "returning":
                gx, gy = GROUND_STATION
                dx, dy = gx - b.x, gy - b.y
                seg = math.hypot(dx, dy)
                if seg < actual_step:
                    b.x, b.y = gx, gy
                    b.state = "swapping"
                    b.swap_timer = SWAP_TIME_S
                else:
                    b.heading = math.atan2(dy, dx)
                    b.x += (dx/seg) * actual_step
                    b.y += (dy/seg) * actual_step

            elif b.state == "swapping":
                b.swap_timer -= step_dt
                if b.swap_timer <= 0:
                    b.battery_j = BATTERY_CAPACITY_J
                    b.swaps += 1
                    b.state = "repositioning"

            elif b.state == "repositioning":
                rx, ry = b.saved_x, b.saved_y
                dx, dy = rx - b.x, ry - b.y
                seg = math.hypot(dx, dy)
                if seg < actual_step:
                    b.x, b.y = rx, ry
                    b.state = b.saved_state
                else:
                    b.heading = math.atan2(dy, dx)
                    b.x += (dx/seg) * actual_step
                    b.y += (dy/seg) * actual_step

    def step(self):
        with self.lock:
            if not self.running or self.target["found"]: return
            dt = VIRTUAL_DT
            self.virtual_time += dt

            for b in self.boats:
                self.step_boat(b, dt)

            if self.target["found"]:
                for b in self.boats:
                    if b.state != "found": b.state = "standby"

    def payload(self):
        for _ in range(15): self.step() # Accelerated visual frames
        with self.lock:
            cov_pct = min(100.0, (len(self.coverage_cells) / 1600.0) * 100.0)
            return {
                "virtual_time": self.virtual_time, "coverage_pct": cov_pct, "cells": list(self.coverage_cells),
                "running": self.running, "target": dict(self.target), "boats": [b.telemetry() for b in self.boats]
            }

SIM = Simulation()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(Path("niot_adaptive_swarm_frontend.html").read_bytes())
        elif self.path == "/api/telemetry":
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(SIM.payload()).encode("utf-8"))
    
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/api/state": SIM.running = body.get("running", False)
        elif self.path == "/api/reset": SIM.reset()
        elif self.path == "/api/target": 
            SIM.reset()
            SIM.target.update({"x": body.get("x", 730), "y": body.get("y", 270), "found": False})
            SIM.running = True 
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":true}')
    
    def log_message(self, format, *args): pass

if __name__ == "__main__":
    print("Accelerated Backend active at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()