import os
import json
import math
import random
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# =====================================================================
# 1. PHYSICAL WORLD & PROPULSION CONSTANTS
# =====================================================================
WORLD_SIZE = 1000.0       # 1x1 km open lake surface area
WATER_DEPTH = 1.5         # Depth parameter d for dipole equation (meters)
GPS_RTK_ERROR = 0.05      # Simulated RTK standard deviation (5 cm accuracy)
MAGNETOMETER_NOISE = 1.5  # Ambient thermal magnetometer noise (nT)
PORT = 8080               # Local HTTP Telemetry Port

# =====================================================================
# 2. PLUGGABLE SWARM STRATEGIES (THE ALGORITHM SANDBOX)
# =====================================================================
class SwarmStrategies:
    """
    MODULAR STRATEGY REGISTER
    --------------------------
    Add or modify strategies here. Every strategy must return a dict mapping
    boat_id to an execution command dict: {'v': linear_velocity, 'omega': angular_velocity}.
    """
    @staticmethod
    def random_walk(boats, target_config, dt):
        commands = {}
        turn_rate_rad = math.radians(target_config['turn_rate_deg'])
        for b in boats:
            heading_error = random.uniform(-turn_rate_rad, turn_rate_rad)
            commands[b['id']] = {'v': target_config['max_speed'], 'omega': heading_error / dt}
        return commands

    @staticmethod
    def concentric_spiral(boats, target_config, dt):
        commands = {}
        num_boats = len(boats)
        
        # TIGHT LANE SPACING: 18m rows guarantee overlapping coverage with 10m sensors
        ladder_distance = 18.0  
        center_x = WORLD_SIZE / 2.0
        center_y = WORLD_SIZE / 2.0
        
        # 1. PURE BLIND SENSOR AGGREGATION (Zero Target Coordinate Leaks)
        g_best_x, g_best_y, g_best_mag = None, None, 0.0
        for ob in boats:
            if ob.get('p_best_mag', 0) > g_best_mag:
                g_best_mag = ob['p_best_mag']
                g_best_x = ob['p_best_x']
                g_best_y = ob['p_best_y']

        # Dynamic anomaly threshold checked directly against config settings
        anomaly_threshold = target_config['target_strength']

        for i, b in enumerate(boats):
            # 2. CONCENTRIC SQUARE SPIRAL WAYPOINT PATTERN GENERATION
            if 'wps' not in b or not b['wps'] or b.get('reset_wps_flag', False):
                b['wps'] = []
                b['reset_wps_flag'] = False
                
                # Area-based concentric square boundary calculations
                s_inner = WORLD_SIZE * math.sqrt(i / num_boats)
                s_outer = WORLD_SIZE * math.sqrt((i + 1) / num_boats)
                
                h_in = s_inner / 2.0
                h_out = s_outer / 2.0
                
                # Start radius avoids overlap with inner neighbor
                r = h_in + (ladder_distance / 2.0)
                if i == 0: 
                    r = ladder_distance / 2.0
                
                # Outward square spiral generator
                while r <= h_out:
                    # Bottom-Left
                    b['wps'].append([center_x - r, center_y - r])
                    # Bottom-Right
                    b['wps'].append([center_x + r, center_y - r])
                    # Top-Right
                    b['wps'].append([center_x + r, center_y + r])
                    # Top-Left
                    b['wps'].append([center_x - r, center_y + r])
                    # Return to Bottom-Left to close the ring before stepping out
                    b['wps'].append([center_x - r, center_y - r])
                    
                    r += ladder_distance
                
                # Ensure the outermost boundary is perfectly covered if division leaves a gap
                if r - ladder_distance < h_out - 1.0:
                    r = h_out
                    b['wps'].append([center_x - r, center_y - r])
                    b['wps'].append([center_x + r, center_y - r])
                    b['wps'].append([center_x + r, center_y + r])
                    b['wps'].append([center_x - r, center_y + r])
                    b['wps'].append([center_x - r, center_y - r])

            # 3. BEHAVIORAL CONTROL STATE DECIDER
            if g_best_mag >= anomaly_threshold and g_best_x is not None:
                b['state'] = 'converging'
                dx = g_best_x - b['x']
                dy = g_best_y - b['y']
                dist = math.hypot(dx, dy)
                
                # SENSOR-DRIVEN BRAKING RULE: Apply hard brakes if inside accuracy window
                if dist <= 3.5:
                    commands[b['id']] = {'v': 0.0, 'omega': 0.0}
                    continue
            else:
                b['state'] = 'searching'
                if not b['wps']:
                    commands[b['id']] = {'v': 0.0, 'omega': 0.0}
                    continue
                    
                wp = b['wps'][0]
                dx = wp[0] - b['x']
                dy = wp[1] - b['y']
                dist = math.hypot(dx, dy)
                
                # Advance tracking focus sequentially if within interception footprint
                if dist < 20.0:
                    b['wps'].pop(0)
                    if not b['wps']:
                        commands[b['id']] = {'v': 0.0, 'omega': 0.0}
                        continue
                    wp = b['wps'][0]
                    dx = wp[0] - b['x']
                    dy = wp[1] - b['y']
                    dist = math.hypot(dx, dy)

            # 4. INTER-VEHICLE COLLISION AVOIDANCE FIELD
            repulsion_x, repulsion_y = 0.0, 0.0
            bubble_radius = 22.0
            for ob in boats:
                if ob['id'] == b['id']: 
                    continue
                rx = b['x'] - ob['x']
                ry = b['y'] - ob['y']
                p_dist = math.hypot(rx, ry)
                if 0.0 < p_dist < bubble_radius:
                    repulsion_x += (rx / p_dist) * (bubble_radius - p_dist)
                    repulsion_y += (ry / p_dist) * (bubble_radius - p_dist)

            # 5. VELOCITY TRACKING & KINEMATIC MOMENTUM FILTERING
            max_speed = target_config['max_speed']
            target_vx = (dx / dist if dist > 0.1 else 0.0) * max_speed + repulsion_x * 0.6
            target_vy = (dy / dist if dist > 0.1 else 0.0) * max_speed + repulsion_y * 0.6
            
            if 'vx_filter' not in b: 
                b['vx_filter'], b['vy_filter'] = 0.0, 0.0
                
            # Filter blend (Smooths out sudden changes when maneuvering around corners)
            alpha = 0.35
            b['vx_filter'] += (target_vx - b['vx_filter']) * alpha
            b['vy_filter'] += (target_vy - b['vy_filter']) * alpha
            
            # Map Cartesian guidance onto steering commands
            filtered_heading = math.atan2(b['vy_filter'], b['vx_filter'])
            heading_err = (filtered_heading - b['heading'] + math.pi) % (math.pi * 2) - math.pi
            
            v_cmd = math.hypot(b['vx_filter'], b['vy_filter'])
            if v_cmd > max_speed: 
                v_cmd = max_speed
                
            steer_limit = math.radians(target_config['turn_rate_deg'])
            omega_cmd = max(-steer_limit, min(steer_limit, heading_err * 2.2))
            
            commands[b['id']] = {'v': v_cmd, 'omega': omega_cmd / dt}
            
        return commands

    @staticmethod
    def particle_swarm_optimization(boats, target_config, dt):
        commands = {}
        g_best_x, g_best_y, g_best_mag = 500.0, 500.0, 0.0
        for b in boats:
            if b.get('p_best_mag', 0) > g_best_mag:
                g_best_mag, g_best_x, g_best_y = b['p_best_mag'], b['p_best_x'], b['p_best_y']
        w, c1, c2 = 0.5, 1.2, 1.8
        for b in boats:
            if 'vx' not in b: b['vx'], b['vy'] = 0.0, 0.0
            r1, r2 = random.random(), random.random()
            b['vx'] = w * b['vx'] + c1 * r1 * (b.get('p_best_x', b['x']) - b['x']) + c2 * r2 * (g_best_x - b['x'])
            b['vy'] = w * b['vy'] + c1 * r2 * (b.get('p_best_y', b['y']) - b['y']) + c2 * r2 * (g_best_y - b['y'])
            v_mag = math.hypot(b['vx'], b['vy'])
            max_v = target_config['max_speed'] * 10.0
            if v_mag > max_v:
                b['vx'], b['vy'] = (b['vx'] / v_mag) * max_v, (b['vy'] / v_mag) * max_v
            desired_heading = math.atan2(b['vy'], b['vx'])
            heading_err = (desired_heading - b['heading'] + math.pi) % (math.pi * 2) - math.pi
            commands[b['id']] = {
                'v': min(target_config['max_speed'], math.hypot(b['vx'], b['vy']) / 10.0),
                'omega': max(-1.2, min(1.2, heading_err * 2.5))
            }
        return commands

# =====================================================================
# 3. INTER-PROCESS HARDWARE ENGINE / BACKEND TELEMETRY PIPELINE
# =====================================================================
class ASVSimulationEngine:
    def __init__(self):
        self.tick_count = 0
        self.paused = True
        self.strategy_name = 'random'
        self.config = {
            'num_boats': 3, 'max_speed': 1.0, 'turn_rate_deg': 15.0,
            'target_strength': 50.0, 'target_falloff': 10.0
        }
        self.target = {'x': 600.0, 'y': 650.0, 'found': False}
        self.boats = []
        self.reset_simulation()

    def reset_simulation(self):
        self.tick_count = 0
        self.target['found'] = False
        self.boats = []
        n = self.config['num_boats']
        for i in range(n):
            # Restores the clustered bottom-left starting layout (spaced 6m apart)
            start_x = 50.0 + (i * 6.0)
            start_y = 50.0
            
            self.boats.append({
                'id': i,
                'x': start_x, 
                'y': start_y,
                'heading': math.radians(45.0), # Set initial heading to 45 degrees pointing inward
                'speed': 0.0, 
                'mag_reading': 0.0,
                'p_best_mag': 0.0, 
                'p_best_x': start_x, 
                'p_best_y': start_y, 
                'state': 'searching',
                'wps': [], 
                'reset_wps_flag': True
            })

    def compute_magnetic_dipole(self, x, y):
        distance = math.hypot(x - self.target['x'], y - self.target['y'])
        if distance < 1.0: true_field = self.config['target_strength']
        else:
            falloff_cubed = self.config['target_falloff'] ** 3
            depth_factor = (WATER_DEPTH ** 2 / (distance ** 2 + WATER_DEPTH ** 2)) ** 1.5
            true_field = self.config['target_strength'] * falloff_cubed * depth_factor
        noise = np.random.normal(0, MAGNETOMETER_NOISE)
        return min(self.config['target_strength'] * 2.5, max(0.0, true_field + noise))

    def step_simulation(self):
        if self.paused: return
        dt = 0.05
        self.tick_count += 1
        
        # Route logic to new concentric spiral algorithm
        if self.strategy_name == 'spiral': 
            cmds = SwarmStrategies.concentric_spiral(self.boats, self.config, dt)
        elif self.strategy_name == 'pso': 
            cmds = SwarmStrategies.particle_swarm_optimization(self.boats, self.config, dt)
        else: 
            cmds = SwarmStrategies.random_walk(self.boats, self.config, dt)
            
        for b in self.boats:
            cmd = cmds.get(b['id'], {'v': 0.0, 'omega': 0.0})
            b['heading'] = (b['heading'] + cmd['omega'] * dt + math.pi) % (math.pi * 2) - math.pi
            b['x'] += math.cos(b['heading']) * cmd['v'] * dt * 15.0
            b['y'] += math.sin(b['heading']) * cmd['v'] * dt * 15.0
            margin = 15.0
            if b['x'] < margin or b['x'] > WORLD_SIZE - margin:
                b['heading'] = math.pi - b['heading']
                b['x'] = np.clip(b['x'], margin, WORLD_SIZE - margin)
            if b['y'] < margin or b['y'] > WORLD_SIZE - margin:
                b['heading'] = -b['heading']
                b['y'] = np.clip(b['y'], margin, WORLD_SIZE - margin)
            gps_x, gps_y = b['x'] + np.random.normal(0, GPS_RTK_ERROR), b['y'] + np.random.normal(0, GPS_RTK_ERROR)
            b['mag_reading'] = self.compute_magnetic_dipole(gps_x, gps_y)
            if b['mag_reading'] > b.get('p_best_mag', 0):
                b['p_best_mag'], b['p_best_x'], b['p_best_y'] = b['mag_reading'], gps_x, gps_y
            if math.hypot(b['x'] - self.target['x'], b['y'] - self.target['y']) < 9.0:
                self.target['found'], b['state'] = True, 'found'

    def get_payload(self):
        self.step_simulation()
        return {'tick': self.tick_count, 'target': self.target, 'boats': self.boats, 'paused': self.paused}

# Global backend instance
engine = ASVSimulationEngine()

# =====================================================================
# 4. EMBEDDED REALTIME UI RENDERING SUB-LAYER (HTML/JS PIPELINE)
# =====================================================================
HTML_INTERFACE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ASV Swarm Simulation Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');
  :root {
    --bg: #0a0f1a; --surface: #0f1724; --panel: #131e2e; --border: #1e2d42;
    --accent: #00d4ff; --green: #00ff9d; --amber: #ffb830; --red: #ff4d6a;
    --purple: #a855f7; --text: #e2eaf4; --muted: #5a7a96; --dim: #2a3d52; --water1: #0d2137;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  header { display: flex; align-items: center; justify-content: space-between; padding: 0 20px; height: 48px; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .logo { display: flex; align-items: center; gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 500; color: var(--accent); }
  .logo-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .header-stats { display: flex; gap: 24px; font-family: 'JetBrains Mono', monospace; font-size: 11px; }
  .hstat { display: flex; flex-direction: column; align-items: flex-end; gap: 1px; }
  .hstat-val { color: var(--accent); font-weight: 500; font-size: 13px; }
  .hstat-lbl { color: var(--muted); font-size: 10px; text-transform: uppercase; }
  .status-badge { display: flex; align-items: center; gap: 6px; font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--green); background: rgba(0,255,157,0.08); border: 1px solid rgba(0,255,157,0.2); border-radius: 4px; padding: 3px 10px; }
  .badge-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }
  .main { display: grid; grid-template-columns: 240px 1fr 280px; flex: 1; overflow: hidden; }
  .panel { background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  .panel-right { border-right: none; border-left: 1px solid var(--border); }
  .panel-section { padding: 14px 16px; border-bottom: 1px solid var(--border); }
  .panel-label { font-size: 9px; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin-bottom: 10px; }
  .ctrl-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
  .ctrl-lbl { font-size: 12px; opacity: 0.8; }
  .ctrl-val { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: var(--accent); background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.15); border-radius: 3px; padding: 2px 8px; min-width: 45px; text-align: center; }
  input[type=range] { width: 100%; height: 4px; background: var(--dim); border-radius: 2px; outline: none; margin: 4px 0 12px; cursor: pointer; -webkit-appearance: none; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 6px var(--accent); }
  .toggle-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .toggle-lbl { font-size: 12px; opacity: 0.8; }
  .toggle { position: relative; width: 36px; height: 18px; cursor: pointer; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle-slider { position: absolute; inset: 0; background: var(--dim); border-radius: 9px; transition: 0.2s; border: 1px solid var(--border); }
  .toggle-slider:before { position: absolute; content: ''; width: 12px; height: 12px; left: 2px; top: 2px; background: var(--muted); border-radius: 50%; transition: 0.2s; }
  .toggle input:checked + .toggle-slider { background: rgba(0,212,255,0.2); border-color: var(--accent); }
  .toggle input:checked + .toggle-slider:before { transform: translateX(18px); background: var(--accent); }
  .btn-group { display: flex; gap: 6px; margin-bottom: 8px; }
  .btn { flex: 1; padding: 8px 10px; font-size: 11px; font-family: 'JetBrains Mono', monospace; border: 1px solid var(--border); background: transparent; color: var(--text); border-radius: 4px; cursor: pointer; transition: all 0.15s; text-align: center; }
  .btn:hover { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,0.05); }
  .btn.active { background: rgba(0,212,255,0.12); border-color: var(--accent); color: var(--accent); }
  .btn.danger { border-color: var(--red); color: var(--red); }
  .btn.danger:hover { background: rgba(255,77,106,0.1); }
  .btn.success { border-color: var(--green); color: var(--green); }
  .btn.success:hover { background: rgba(0,255,157,0.08); }
  .canvas-wrap { position: relative; flex: 1; background: var(--water1); display: flex; align-items: center; justify-content: center; overflow: hidden; }
  canvas#lake { background: #071424; box-shadow: inset 0 0 40px rgba(0,0,0,0.8); }
  .hud-overlay { position: absolute; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: rgba(0,212,255,0.6); pointer-events: none; }
  .telem-scroll { overflow-y: auto; flex: 1; }
  .boat-card { padding: 12px 16px; border-bottom: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; }
  .boat-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
  .boat-name { display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 600; }
  .boat-dot { width: 8px; height: 8px; border-radius: 50%; }
  .telem-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 11px; }
  .telem-k { color: var(--muted); font-size: 9px; text-transform: uppercase; }
  .mag-bar-track { height: 4px; background: var(--dim); border-radius: 2px; margin-top: 4px; overflow: hidden; }
  .mag-bar-fill { height: 100%; width: 0%; background: var(--accent); transition: width 0.1s; }
  .log-wrap { height: 140px; background: var(--surface); border-top: 1px solid var(--border); padding: 10px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 10px; }
  .log-row { margin-bottom: 4px; color: var(--muted); }
  .log-row.success { color: var(--green); }
  .log-row.warn { color: var(--amber); }
  .cmd-section { padding: 14px 16px; border-top: 1px solid var(--border); background: rgba(168,85,247,0.02); }
  .algo-badge { display: flex; align-items: center; font-size: 10px; color: var(--purple); font-family: 'JetBrains Mono', monospace; margin-bottom: 8px; background: rgba(168,85,247,0.08); padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(168,85,247,0.2); }
  .found-overlay { position: absolute; inset: 0; background: radial-gradient(circle, rgba(0,255,157,0.12) 0%, transparent 70%); opacity: 0; pointer-events: none; transition: opacity 0.4s; }
  .found-overlay.active { opacity: 1; }
</style>
</head>
<body>
<header>
  <div class="logo"><div class="logo-dot"></div>NIOT ASV SWARM CORE ENGINE</div>
  <div class="header-stats">
    <div class="hstat"><span class="hstat-val" id="stat-time">00:00</span><span class="hstat-lbl">Elapsed Time</span></div>
    <div class="hstat"><span class="hstat-val" id="stat-peak">0.0 nT</span><span class="hstat-lbl">Max Anomaly</span></div>
    <div class="hstat"><span class="hstat-val" id="stat-coverage">0.0%</span><span class="hstat-lbl">Area Scanned</span></div>
  </div>
  <div class="status-badge"><div class="badge-dot"></div><span id="stat-mode">INITIALIZED</span></div>
</header>
<div class="main">
  <div class="panel">
    <div class="panel-section">
      <div class="panel-label">Swarm Infrastructure</div>
      <div class="ctrl-row"><span class="ctrl-lbl">Fleet Size</span><span class="ctrl-val" id="lbl-boats">3</span></div>
      <input type="range" id="sl-boats" min="3" max="5" value="3">
      <div class="ctrl-row"><span class="ctrl-lbl">Max Velocity</span><span class="ctrl-val" id="lbl-speed">1.0 m/s</span></div>
      <input type="range" id="sl-speed" min="0.5" max="3.0" step="0.1" value="1.0">
      <div class="ctrl-row"><span class="ctrl-lbl">Steer Threshold</span><span class="ctrl-val" id="lbl-turn">15°/s</span></div>
      <input type="range" id="sl-turn" min="5" max="45" value="15">
    </div>
    <div class="panel-section">
      <div class="panel-label">Telemetry Filters</div>
      <div class="toggle-row"><span class="toggle-lbl">Blind Search Mode</span><label class="toggle"><input type="checkbox" id="tg-blind" checked><span class="toggle-slider"></span></label></div>
      <div class="toggle-row"><span class="toggle-lbl">Magnetic Heatmap</span><label class="toggle"><input type="checkbox" id="tg-mag" checked><span class="toggle-slider"></span></label></div>
      <div class="toggle-row"><span class="toggle-lbl">Field Contours</span><label class="toggle"><input type="checkbox" id="tg-contour" checked><span class="toggle-slider"></span></label></div>
      <div class="toggle-row"><span class="toggle-lbl">RF Comm Interlinks</span><label class="toggle"><input type="checkbox" id="tg-comms" checked><span class="toggle-slider"></span></label></div>
      <div class="toggle-row"><span class="toggle-lbl">Coverage Matrix</span><label class="toggle"><input type="checkbox" id="tg-coverage" checked><span class="toggle-slider"></span></label></div>
    </div>
    <div class="panel-section">
      <div class="panel-label">Submerged Dipole Configuration</div>
      <div class="ctrl-row"><span class="ctrl-lbl">Core Source</span><span class="ctrl-val" id="lbl-strength">50 nT</span></div>
      <input type="range" id="sl-strength" min="20" max="150" value="50">
      <div class="ctrl-row"><span class="ctrl-lbl">Falloff Radius</span><span class="ctrl-val" id="lbl-falloff">80 m</span></div>
      <input type="range" id="sl-falloff" min="0" max="120" value="80">
    </div>
    <div class="panel-section" style="flex:1">
      <div class="panel-label">Mission Control</div>
      <div class="btn-group">
        <button class="btn success" id="btn-start">EXECUTE</button>
        <button class="btn danger" id="btn-reset">RESET</button>
      </div>
      <button class="btn" id="btn-random-target" style="width:100%">🎲 RANDOMIZE TARGET LOCATION</button>
    </div>
  </div>
  <div class="canvas-wrap" id="canvas-container">
    <canvas id="lake"></canvas>
    <div class="hud-overlay" style="top:14px; left:14px;">
      <div>SURFACE AREA: 1000m × 1000m</div>
      <div id="hud-mode">ROUTING: RANDOM_WALK</div>
    </div>
    <div class="hud-overlay" style="bottom:14px; right:14px; text-align:right;">
      <div id="hud-coords">CURSOR: ---m, ---m</div>
    </div>
    <div class="found-overlay" id="flash-overlay"></div>
  </div>
  <div class="panel panel-right">
    <div class="panel-section"><div class="panel-label">Dynamic Node Array</div></div>
    <div class="telem-scroll" id="telemetry-feed"></div>
    <div class="cmd-section">
      <div class="panel-label">Swarm Control Router</div>
      <div class="algo-badge"><span>◈ STRATEGY: </span><span id="lbl-active-algo" style="margin-left:4px; font-weight:bold;">RANDOM_WALK</span></div>
      <div class="btn-group">
        <button class="btn active" onclick="setStrategy('random')">RANDOM</button>
        <button class="btn" onclick="setStrategy('spiral')">SPIRAL</button>
        <button class="btn" onclick="setStrategy('pso')">PSO COOP</button>
      </div>
    </div>
    <div class="log-wrap" id="system-log"></div>
  </div>
</div>
<script>
  const canvas = document.getElementById('lake');
  const ctx = canvas.getContext('2d');
  const colors = ['#00d4ff','#00ff9d','#ffb830','#a855f7','#ff4d6a'];
  let scale = 1.0;
  let coverageMatrix = new Uint8Array(40 * 40);

  function resizeCanvas() {
    const parent = document.getElementById('canvas-container');
    const dim = Math.min(parent.clientWidth, parent.clientHeight) - 32;
    canvas.width = dim; canvas.height = dim;
    scale = dim / 1000.0;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  function appendLog(type, text) {
    const container = document.getElementById('system-log');
    const stamp = new Date().toTimeString().split(' ')[0];
    container.innerHTML = `<div class="log-row ${type}">[${stamp}] ${text}</div>` + container.innerHTML;
  }

  function pushConfig() {
    fetch('/api/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        num_boats: parseInt(document.getElementById('sl-boats').value),
        max_speed: parseFloat(document.getElementById('sl-speed').value),
        turn_rate_deg: parseFloat(document.getElementById('sl-turn').value),
        target_strength: parseFloat(document.getElementById('sl-strength').value),
        target_falloff: parseFloat(document.getElementById('sl-falloff').value)
      })
    });
  }

  ['sl-boats','sl-speed','sl-turn','sl-strength','sl-falloff'].forEach(id => {
    document.getElementById(id).addEventListener('input', (e) => {
      let suffix = id.includes('speed') ? ' m/s' : id.includes('turn') ? '°/s' : id.includes('falloff') ? ' m' : id.includes('strength') ? ' nT' : '';
      document.getElementById(id.replace('sl-', 'lbl-')).innerText = e.target.value + suffix;
      pushConfig();
    });
  });

  document.getElementById('btn-start').addEventListener('click', () => {
    let active = document.getElementById('btn-start').innerText === 'EXECUTE';
    document.getElementById('btn-start').innerText = active ? 'SUSPEND' : 'EXECUTE';
    document.getElementById('btn-start').classList.toggle('success', !active);
    document.getElementById('stat-mode').innerText = active ? 'ENGAGED' : 'SUSPENDED';
    fetch('/api/state', {method: 'POST', body: active ? 'running' : 'paused'});
    appendLog('info', active ? 'Swarm navigation loop active.' : 'Swarm navigation loop paused.');
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    fetch('/api/reset', {method: 'POST'});
    coverageMatrix.fill(0);
    appendLog('warn', 'Swarm hardware array state re-indexed.');
  });

  document.getElementById('btn-random-target').addEventListener('click', () => {
    // Generate undisclosed random positions cleanly bounded within the lake margins (100m to 900m)
    const rx = 100.0 + Math.random() * 800.0;
    const ry = 100.0 + Math.random() * 800.0;
    
    fetch('/api/target', {
      method: 'POST', 
      headers: {'Content-Type': 'application/json'}, 
      body: JSON.stringify({x: rx, y: ry})
    });
    
    appendLog('success', `Submerged anomaly dropped at random undisclosed coordinates!`);
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const wx = ((e.clientX - rect.left) / scale).toFixed(0);
    const wy = (1000.0 - ((e.clientY - rect.top) / scale)).toFixed(0);
    document.getElementById('hud-coords').innerText = `CURSOR: ${wx}m, ${wy}m`;
  });

  function setStrategy(key) {
    fetch('/api/strategy', {method: 'POST', body: key});
    document.getElementById('lbl-active-algo').innerText = key.toUpperCase() + '_ROUTE';
    document.getElementById('hud-mode').innerText = 'ROUTING: ' + key.toUpperCase();
    document.querySelectorAll('.cmd-section .btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    appendLog('info', `Swarm strategy profile swapped to [${key.toUpperCase()}].`);
  }

  function drawEngine(data) {
    ctx.clearRect(0,0, canvas.width, canvas.height);
    const S = canvas.width;
    ctx.strokeStyle = 'rgba(30, 45, 66, 0.4)';
    ctx.lineWidth = 1;
    for(let i=100; i<1000; i+=100) {
      ctx.beginPath(); ctx.moveTo(i*scale, 0); ctx.lineTo(i*scale, S); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i*scale); ctx.lineTo(S, i*scale); ctx.stroke();
    }
    data.boats.forEach(b => {
      let cx = Math.floor(b.x / 25.0), cy = Math.floor(b.y / 25.0);
      if(cx>=0 && cx<40 && cy>=0 && cy<40) coverageMatrix[cy*40 + cx] = 1;
    });
    if(document.getElementById('tg-coverage').checked) {
      ctx.fillStyle = 'rgba(0, 255, 157, 0.04)';
      for(let y=0; y<40; y++) {
        for(let x=0; x<40; x++) {
          if(coverageMatrix[y*40 + x]) ctx.fillRect(x*25*scale, S - (y*25*scale) - 25*scale, 25*scale, 25*scale);
        }
      }
    }

    // CHECK BLIND SEARCH RULES
    const hideTargetData = document.getElementById('tg-blind').checked && !data.target.found;

    if(document.getElementById('tg-mag').checked && !hideTargetData) {
      let rad = parseFloat(document.getElementById('sl-falloff').value) * scale;
      let grad = ctx.createRadialGradient(data.target.x*scale, S - (data.target.y*scale), 0, data.target.x*scale, S - (data.target.y*scale), rad*1.8);
      grad.addColorStop(0, 'rgba(0, 212, 255, 0.3)');
      grad.addColorStop(0.3, 'rgba(168, 85, 247, 0.08)');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(data.target.x*scale, S - (data.target.y*scale), rad*1.8, 0, Math.PI*2); ctx.fill();
    }
    if(document.getElementById('tg-contour').checked && !hideTargetData) {
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      [0.4, 0.8, 1.3].forEach(f => {
        ctx.beginPath(); ctx.arc(data.target.x*scale, S - (data.target.y*scale), parseFloat(document.getElementById('sl-falloff').value)*f*scale, 0, Math.PI*2); ctx.stroke();
      });
      ctx.setLineDash([]);
    }
    if(document.getElementById('tg-comms').checked) {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.25)';
      ctx.lineWidth = 1;
      for(let i=0; i<data.boats.length; i++) {
        for(let j=i+1; j<data.boats.length; j++) {
          if (Math.hypot(data.boats[i].x - data.boats[j].x, data.boats[i].y - data.boats[j].y) < 320.0) {
            ctx.beginPath(); ctx.moveTo(data.boats[i].x * scale, S - (data.boats[i].y * scale)); ctx.lineTo(data.boats[j].x * scale, S - (data.boats[j].y * scale)); ctx.stroke();
          }
        }
      }
    }

    if (!hideTargetData) {
      let tx = data.target.x * scale, ty = S - (data.target.y * scale);
      ctx.fillStyle = data.target.found ? 'var(--green)' : 'var(--amber)';
      ctx.beginPath(); ctx.arc(tx, ty, 4, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = ctx.fillStyle;
      ctx.beginPath(); ctx.arc(tx, ty, 10 + Math.sin(data.tick*0.1)*3, 0, Math.PI*2); ctx.stroke();
    }

    data.boats.forEach(b => {
      let bx = b.x * scale, by = S - (b.y * scale), col = colors[b.id % colors.length];
      ctx.save(); ctx.translate(bx, by); ctx.rotate(-b.heading - Math.PI/2);
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.moveTo(0, 8); ctx.lineTo(-5, -6); ctx.lineTo(5, -6); ctx.closePath(); ctx.fill();
      ctx.restore();
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = '9px JetBrains Mono';
      ctx.fillText(`ASV-${b.id+1}`, bx + 8, by - 4);
    });
    let feedHTML = '', highestMag = 0;
    data.boats.forEach(b => {
      if (b.mag_reading > highestMag) highestMag = b.mag_reading;
      let col = colors[b.id % colors.length];
      let maxCap = parseFloat(document.getElementById('sl-strength').value);
      let percentage = Math.min(100, (b.mag_reading / maxCap) * 100);
      feedHTML += `<div class="boat-card">
          <div class="boat-card-header">
            <div class="boat-name"><div class="boat-dot" style="background:${col}"></div>ASV-0${b.id+1}</div>
            <div style="font-size:10px; color:${b.state==='found'?'var(--green)':'var(--muted)'}">${b.state.toUpperCase()}</div>
          </div>
          <div class="telem-grid">
            <div><span class="telem-k">Pose X:</span> ${b.x.toFixed(1)}m</div><div><span class="telem-k">Pose Y:</span> ${b.y.toFixed(1)}m</div>
            <div><span class="telem-k">Heading:</span> ${((b.heading*180/Math.PI)%360).toFixed(0)}°</div><div><span class="telem-k">Sensor:</span> ${b.mag_reading.toFixed(1)} nT</div>
          </div>
          <div class="mag-bar-track"><div class="mag-bar-fill" style="width:${percentage}%; background:${col}"></div></div>
        </div>`;
    });
    document.getElementById('telemetry-feed').innerHTML = feedHTML;
    document.getElementById('stat-peak').innerText = highestMag.toFixed(1) + ' nT';
    document.getElementById('stat-coverage').innerText = ((coverageMatrix.reduce((a,v)=>a+v,0)/1600)*100).toFixed(1) + '%';
    document.getElementById('stat-time').innerText = `${String(Math.floor(data.tick/1200)).padStart(2,'0')}:${String(Math.floor((data.tick/20)%60)).padStart(2,'0')}`;
    let overlay = document.getElementById('flash-overlay');
    if (data.target.found) {
      overlay.classList.add('active');
      document.getElementById('stat-mode').innerText = 'TARGET LOCATED';
      document.getElementById('stat-mode').parentElement.style.color = 'var(--green)';
    } else { overlay.classList.remove('active'); }
  }

  function queryLoop() {
    fetch('/api/telemetry').then(r => r.json()).then(data => drawEngine(data)).catch(e => {});
  }

  pushConfig();
  appendLog('success', 'Python HTTP core dynamic kernel connected.');
  setInterval(queryLoop, 50);
</script>
</body>
</html>
"""

# =====================================================================
# 5. LIGHTWEIGHT HTTP TELEMETRY SERVER & BROWSER LAUNCHER
# =====================================================================
class GCSRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Mute terminal HTTP spam to keep command line clean

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_INTERFACE.encode('utf-8'))
        elif self.path == '/api/telemetry':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(engine.get_payload()).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length > 0 else ""
        
        if self.path == '/api/config':
            data = json.loads(body)
            re_spawn = int(data['num_boats']) != engine.config['num_boats']
            engine.config.update(data)
            if re_spawn: engine.reset_simulation()
        elif self.path == '/api/target':
            data = json.loads(body)
            engine.target['x'], engine.target['y'] = float(data['x']), float(data['y'])
            engine.target['found'] = False
            for b in engine.boats: b['state'], b['p_best_mag'] = 'searching', 0.0
        elif self.path == '/api/state':
            engine.paused = (body == 'paused')
        elif self.path == '/api/reset':
            engine.reset_simulation()
        elif self.path == '/api/strategy':
            engine.strategy_name = body
            
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', PORT), GCSRequestHandler)
    url = f'http://127.0.0.1:{PORT}'
    print(f"=== NIOT NSCOT OBJECTIVE 2 GCS SERVER ACTIVE ===")
    print(f"-> Local Telemetry Bridge: {url}")
    print(f"-> Auto-launching system web browser...")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] GCS Telemetry Server Closed.")
        server.server_close()