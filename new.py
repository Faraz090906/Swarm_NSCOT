import os
import json
import math
import random
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# =====================================================================
# 1. PHYSICAL WORLD & PROPULSION CONSTANTS (CENTERED AT 0,0)
# =====================================================================
WORLD_HALF = 500.0        # Boundary offset from origin (-500m to +500m, 1km total grid)
WATER_DEPTH = 1.5         # Depth parameter d for dipole equation (meters)
GPS_RTK_ERROR = 0.05      # Simulated RTK standard deviation (5 cm accuracy)
MAGNETOMETER_NOISE = 1.5  # Ambient thermal magnetometer noise (nT)
PORT = 8080               # Local HTTP Telemetry Port
NUM_BOATS = 4             # Strictly fixed fleet size of 4 ASVs

# =====================================================================
# 2. TRUE 4-QUADRANT CONCENTRIC EXPANDING SQUARE SWARM STRATEGY
# =====================================================================
class SwarmStrategies:
    """
    STRICT 4-QUADRANT CONCENTRIC SEARCH SWARM
    ----------------------------------------
    All ASVs deploy from origin (0,0) and expand outward into their respective quadrants:
      - ASV 0: Quadrant I   (+X, +Y) [North-East]
      - ASV 1: Quadrant II  (-X, +Y) [North-West]
      - ASV 2: Quadrant III (-X, -Y) [South-West]
      - ASV 3: Quadrant IV  (+X, -Y) [South-East]
    High-confidence thresholding prevents premature collapse before completing search loops.
    """
    @staticmethod
    def concentric_swarm_search(boats, target_config, dt):
        commands = {}
        
        # Quadrant sign multipliers for the 4 ASVs (Index 0 to 3)
        quadrant_signs = [
            (1.0, 1.0),   # ASV 0: NE (+X, +Y)
            (-1.0, 1.0),  # ASV 1: NW (-X, +Y)
            (-1.0, -1.0), # ASV 2: SW (-X, -Y)
            (1.0, -1.0)   # ASV 3: SE (+X, -Y)
        ]
        
        # 1. GLOBAL SWARM SENSOR AGGREGATION & HIGH-CONFIDENCE THRESHOLD
        g_best_x, g_best_y, g_best_mag = None, None, 0.0
        for ob in boats:
            if ob.get('p_best_mag', 0) > g_best_mag:
                g_best_mag = ob['p_best_mag']
                g_best_x = ob['p_best_x']
                g_best_y = ob['p_best_y']

        # Strict threshold (requires 85% of peak strength) to prevent premature collapsing
        anomaly_threshold = target_config['target_strength'] * 0.85

        for i, b in enumerate(boats):
            x_sign, y_sign = quadrant_signs[i]

            # 2. GENERATE QUADRANT-ISOLATED EXPANDING SQUARE WAYPOINTS FROM (0,0)
            if 'wps' not in b or not b['wps'] or b.get('reset_wps_flag', False):
                b['wps'] = []
                b['reset_wps_flag'] = False
                
                curr_x, curr_y = 0.0, 0.0
                b['wps'].append([curr_x, curr_y])
                
                step = 30.0
                # Construct expanding square legs scaled strictly into the assigned quadrant
                for r in range(1, 16):
                    d = r * step
                    b['wps'].append([x_sign * d, y_sign * 0.0])
                    b['wps'].append([x_sign * d, y_sign * d])
                    b['wps'].append([x_sign * 0.0, y_sign * d])
                    b['wps'].append([x_sign * d, y_sign * d])

            # 3. BEHAVIORAL CONTROL STATE DECIDER
            if g_best_mag >= anomaly_threshold and g_best_x is not None:
                b['state'] = 'swarm_converging'
                dx = g_best_x - b['x']
                dy = g_best_y - b['y']
                dist = math.hypot(dx, dy)
                
                # SENSOR-DRIVEN BRAKING RULE: Hold position precisely within 3.5m detection range
                if dist <= 3.5:
                    commands[b['id']] = {'v': 0.0, 'omega': 0.0}
                    continue
            else:
                b['state'] = 'concentric_searching'
                if not b['wps']:
                    commands[b['id']] = {'v': 0.0, 'omega': 0.0}
                    continue
                    
                wp = b['wps'][0]
                dx = wp[0] - b['x']
                dy = wp[1] - b['y']
                dist = math.hypot(dx, dy)
                
                if dist < 12.0:
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
            bubble_radius = 20.0
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
                
            alpha = 0.35
            b['vx_filter'] += (target_vx - b['vx_filter']) * alpha
            b['vy_filter'] += (target_vy - b['vy_filter']) * alpha
            
            filtered_heading = math.atan2(b['vy_filter'], b['vx_filter'])
            heading_err = (filtered_heading - b['heading'] + math.pi) % (math.pi * 2) - math.pi
            
            v_cmd = math.hypot(b['vx_filter'], b['vy_filter'])
            if v_cmd > max_speed: 
                v_cmd = max_speed
                
            steer_limit = math.radians(target_config['turn_rate_deg'])
            omega_cmd = max(-steer_limit, min(steer_limit, heading_err * 2.2))
            
            commands[b['id']] = {'v': v_cmd, 'omega': omega_cmd / dt}
            
        return commands

# =====================================================================
# 3. INTER-PROCESS HARDWARE ENGINE / BACKEND TELEMETRY PIPELINE
# =====================================================================
class ASVSimulationEngine:
    def __init__(self):
        self.tick_count = 0
        self.paused = True
        self.config = {
            'num_boats': NUM_BOATS, 'max_speed': 1.0, 'turn_rate_deg': 15.0,
            'target_strength': 50.0, 'target_falloff': 80.0
        }
        self.target = {'x': 200.0, 'y': 250.0, 'found': False}
        self.boats = []
        self.reset_simulation()

    def reset_simulation(self):
        self.tick_count = 0
        self.target['found'] = False
        self.boats = []
        n = self.config['num_boats']
        
        # Unified initial deployment dock point at exact origin (0.0, 0.0)
        common_start_x = 0.0
        common_start_y = 0.0
        
        for i in range(n):
            self.boats.append({
                'id': i, 'x': common_start_x, 'y': common_start_y,
                'heading': math.radians(90.0), 'speed': 0.0, 
                'mag_reading': 0.0, 'p_best_mag': 0.0, 
                'p_best_x': common_start_x, 'p_best_y': common_start_y, 
                'state': 'concentric_searching', 'wps': [], 'reset_wps_flag': True
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
        
        cmds = SwarmStrategies.concentric_swarm_search(self.boats, self.config, dt)
            
        for b in self.boats:
            cmd = cmds.get(b['id'], {'v': 0.0, 'omega': 0.0})
            b['heading'] = (b['heading'] + cmd['omega'] * dt + math.pi) % (math.pi * 2) - math.pi
            b['x'] += math.cos(b['heading']) * cmd['v'] * dt * 15.0
            b['y'] += math.sin(b['heading']) * cmd['v'] * dt * 15.0
            
            # Boundary clipping for coordinate space [-500, 500]
            margin = 15.0
            limit = WORLD_HALF - margin
            if b['x'] < -limit or b['x'] > limit:
                b['heading'] = math.pi - b['heading']
                b['x'] = np.clip(b['x'], -limit, limit)
            if b['y'] < -limit or b['y'] > limit:
                b['heading'] = -b['heading']
                b['y'] = np.clip(b['y'], -limit, limit)
                
            gps_x, gps_y = b['x'] + np.random.normal(0, GPS_RTK_ERROR), b['y'] + np.random.normal(0, GPS_RTK_ERROR)
            b['mag_reading'] = self.compute_magnetic_dipole(gps_x, gps_y)
            if b['mag_reading'] > b.get('p_best_mag', 0):
                b['p_best_mag'], b['p_best_x'], b['p_best_y'] = b['mag_reading'], gps_x, gps_y
            
            if math.hypot(b['x'] - self.target['x'], b['y'] - self.target['y']) < 3.5:
                self.target['found'], b['state'] = True, 'target_intercepted'

    def get_payload(self):
        self.step_simulation()
        actual_hours = (self.tick_count * 0.05) / 3600.0
        return {
            'tick': self.tick_count, 
            'actual_hours': actual_hours,
            'target': self.target, 
            'boats': self.boats, 
            'paused': self.paused
        }

engine = ASVSimulationEngine()

# =====================================================================
# 4. EMBEDDED REALTIME UI RENDERING SUB-LAYER (HTML/JS PIPELINE)
# =====================================================================
HTML_INTERFACE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ASV 4-Fleet 4-Quadrant Concentric Swarm Simulation Dashboard</title>
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
  <div class="logo"><div class="logo-dot"></div>NIOT ASV 4-QUADRANT CONCENTRIC SWARM</div>
  <div class="header-stats">
    <div class="hstat"><span class="hstat-val" id="stat-time">0.000 hrs</span><span class="hstat-lbl">Mission Time (hrs)</span></div>
    <div class="hstat"><span class="hstat-val" id="stat-peak">0.0 nT</span><span class="hstat-lbl">Max Anomaly</span></div>
    <div class="hstat"><span class="hstat-val" id="stat-coverage">0.0%</span><span class="hstat-lbl">Area Scanned</span></div>
  </div>
  <div class="status-badge"><div class="badge-dot"></div><span id="stat-mode">INITIALIZED</span></div>
</header>
<div class="main">
  <div class="panel">
    <div class="panel-section">
      <div class="panel-label">Swarm Fleet Configuration</div>
      <div class="ctrl-row"><span class="ctrl-lbl">Fixed Fleet Size</span><span class="ctrl-val">4 ASVs</span></div>
      <div class="ctrl-row" style="margin-top:8px;"><span class="ctrl-lbl">Max Velocity</span><span class="ctrl-val" id="lbl-speed">1.0 m/s</span></div>
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
      <div>SURFACE AREA: 1000m × 1000m (-500m to +500m Origin Center)</div>
      <div id="hud-mode">ROUTING: 4_QUADRANT_CONCENTRIC_SQUARE</div>
    </div>
    <div class="hud-overlay" style="bottom:14px; right:14px; text-align:right;">
      <div id="hud-coords">CURSOR: ---m, ---m</div>
    </div>
    <div class="found-overlay" id="flash-overlay"></div>
  </div>
  <div class="panel panel-right">
    <div class="panel-section"><div class="panel-label">Dynamic Node Array (4 Quadrants)</div></div>
    <div class="telem-scroll" id="telemetry-feed"></div>
    <div class="cmd-section">
      <div class="panel-label">Swarm Control Router</div>
      <div class="algo-badge"><span>◈ ACTIVE: </span><span style="margin-left:4px; font-weight:bold;">4-QUADRANT EXPANDING SQUARE</span></div>
    </div>
    <div class="log-wrap" id="system-log"></div>
  </div>
</div>
<script>
  const canvas = document.getElementById('lake');
  const ctx = canvas.getContext('2d');
  const colors = ['#00d4ff','#00ff9d','#ffb830','#a855f7'];
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
        num_boats: 4,
        max_speed: parseFloat(document.getElementById('sl-speed').value),
        turn_rate_deg: parseFloat(document.getElementById('sl-turn').value),
        target_strength: parseFloat(document.getElementById('sl-strength').value),
        target_falloff: parseFloat(document.getElementById('sl-falloff').value)
      })
    });
  }

  ['sl-speed','sl-turn','sl-strength','sl-falloff'].forEach(id => {
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
    appendLog('info', active ? '4-Quadrant swarm navigation loop active.' : 'Navigation loop paused.');
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    fetch('/api/reset', {method: 'POST'});
    coverageMatrix.fill(0);
    appendLog('warn', '4-ASV Quadrant array re-indexed to dock at (0,0).');
  });

  document.getElementById('btn-random-target').addEventListener('click', () => {
    const rx = (Math.random() > 0.5 ? 1 : -1) * (50.0 + Math.random() * 400.0);
    const ry = (Math.random() > 0.5 ? 1 : -1) * (50.0 + Math.random() * 400.0);
    fetch('/api/target', {
      method: 'POST', 
      headers: {'Content-Type': 'application/json'}, 
      body: JSON.stringify({x: rx, y: ry})
    });
    appendLog('success', `Submerged anomaly dropped at random target coordinates (${rx.toFixed(0)}, ${ry.toFixed(0)})!`);
  });

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const wx = (((e.clientX - rect.left) / scale) - 500.0).toFixed(0);
    const wy = (500.0 - ((e.clientY - rect.top) / scale)).toFixed(0);
    document.getElementById('hud-coords').innerText = `CURSOR: ${wx}m, ${wy}m`;
  });

  function toCanvasCoords(x, y) {
    return {
      x: (x + 500.0) * scale,
      y: canvas.height - ((y + 500.0) * scale)
    };
  }

  function drawEngine(data) {
    ctx.clearRect(0,0, canvas.width, canvas.height);
    const S = canvas.width;
    
    // Grid lines
    ctx.strokeStyle = 'rgba(30, 45, 66, 0.4)';
    ctx.lineWidth = 1;
    for(let i=0; i<=1000; i+=100) {
      let p = i * scale;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, S); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(S, p); ctx.stroke();
    }
    
    // Center Origin Axes (0,0) splitting quadrants
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(500 * scale, 0); ctx.lineTo(500 * scale, S); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, 500 * scale); ctx.lineTo(S, 500 * scale); ctx.stroke();

    data.boats.forEach(b => {
      let cx = Math.floor((b.x + 500.0) / 25.0), cy = Math.floor((b.y + 500.0) / 25.0);
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

    const hideTargetData = document.getElementById('tg-blind').checked && !data.target.found;

    if(document.getElementById('tg-mag').checked && !hideTargetData) {
      let rad = parseFloat(document.getElementById('sl-falloff').value) * scale;
      let t_pos = toCanvasCoords(data.target.x, data.target.y);
      let grad = ctx.createRadialGradient(t_pos.x, t_pos.y, 0, t_pos.x, t_pos.y, rad*1.8);
      grad.addColorStop(0, 'rgba(0, 212, 255, 0.3)');
      grad.addColorStop(0.3, 'rgba(168, 85, 247, 0.08)');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(t_pos.x, t_pos.y, rad*1.8, 0, Math.PI*2); ctx.fill();
    }
    
    if(document.getElementById('tg-contour').checked && !hideTargetData) {
      let t_pos = toCanvasCoords(data.target.x, data.target.y);
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.15)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      [0.4, 0.8, 1.3].forEach(f => {
        ctx.beginPath(); ctx.arc(t_pos.x, t_pos.y, parseFloat(document.getElementById('sl-falloff').value)*f*scale, 0, Math.PI*2); ctx.stroke();
      });
      ctx.setLineDash([]);
    }
    
    if(document.getElementById('tg-comms').checked) {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.25)';
      ctx.lineWidth = 1;
      for(let i=0; i<data.boats.length; i++) {
        for(let j=i+1; j<data.boats.length; j++) {
          if (Math.hypot(data.boats[i].x - data.boats[j].x, data.boats[i].y - data.boats[j].y) < 350.0) {
            let p1 = toCanvasCoords(data.boats[i].x, data.boats[i].y);
            let p2 = toCanvasCoords(data.boats[j].x, data.boats[j].y);
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
          }
        }
      }
    }

    if (!hideTargetData) {
      let t_pos = toCanvasCoords(data.target.x, data.target.y);
      ctx.fillStyle = data.target.found ? 'var(--green)' : 'var(--amber)';
      ctx.beginPath(); ctx.arc(t_pos.x, t_pos.y, 4, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = ctx.fillStyle;
      ctx.beginPath(); ctx.arc(t_pos.x, t_pos.y, 10 + Math.sin(data.tick*0.1)*3, 0, Math.PI*2); ctx.stroke();
    }

    data.boats.forEach(b => {
      let b_pos = toCanvasCoords(b.x, b.y);
      let col = colors[b.id % colors.length];
      ctx.save(); ctx.translate(b_pos.x, b_pos.y); ctx.rotate(-b.heading - Math.PI/2);
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.moveTo(0, 8); ctx.lineTo(-5, -6); ctx.lineTo(5, -6); ctx.closePath(); ctx.fill();
      ctx.restore();
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = '9px JetBrains Mono';
      ctx.fillText(`ASV-${b.id+1}`, b_pos.x + 8, b_pos.y - 4);
    });

    let feedHTML = '', highestMag = 0;
    data.boats.forEach(b => {
      if (b.mag_reading > highestMag) highestMag = b.mag_reading;
      let col = colors[b.id % colors.length];
      let maxCap = parseFloat(document.getElementById('sl-strength').value);
      let percentage = Math.min(100, (b.mag_reading / maxCap) * 100);
      let qName = b.id === 0 ? 'Q1 (NE)' : b.id === 1 ? 'Q2 (NW)' : b.id === 2 ? 'Q3 (SW)' : 'Q4 (SE)';
      feedHTML += `<div class="boat-card">
          <div class="boat-card-header">
            <div class="boat-name"><div class="boat-dot" style="background:${col}"></div>ASV-0${b.id+1} [${qName}]</div>
            <div style="font-size:10px; color:${b.state==='target_intercepted'?'var(--green)':'var(--muted)'}">${b.state.toUpperCase()}</div>
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
    document.getElementById('stat-time').innerText = data.actual_hours.toFixed(3) + ' hrs';
    
    let overlay = document.getElementById('flash-overlay');
    if (data.target.found) {
      overlay.classList.add('active');
      document.getElementById('stat-mode').innerText = 'TARGET LOCATED';
      document.getElementById('stat-mode').parentElement.style.color = 'var(--green)';
    } else { 
      overlay.classList.remove('active'); 
    }
  }

  function queryLoop() {
    fetch('/api/telemetry').then(r => r.json()).then(data => drawEngine(data)).catch(e => {});
  }

  pushConfig();
  appendLog('success', '4-Quadrant Concentric Swarm kernel connected.');
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
        pass

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
            engine.config.update(data)
            engine.config['num_boats'] = 4
        elif self.path == '/api/target':
            data = json.loads(body)
            engine.target['x'], engine.target['y'] = float(data['x']), float(data['y'])
            engine.target['found'] = False
            for b in engine.boats: b['state'], b['p_best_mag'] = 'concentric_searching', 0.0
        elif self.path == '/api/state':
            engine.paused = (body == 'paused')
        elif self.path == '/api/reset':
            engine.reset_simulation()
            
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', PORT), GCSRequestHandler)
    url = f'http://127.0.0.1:{PORT}'
    print(f"=== NIOT 4-QUADRANT CONCENTRIC SWARM GCS SERVER ACTIVE ===")
    print(f"-> Local Telemetry Bridge: {url}")
    print(f"-> Auto-launching system web browser...")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] GCS Telemetry Server Closed.")
        server.server_close()