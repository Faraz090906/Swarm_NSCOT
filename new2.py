import math
import random
import numpy as np

# =====================================================================
# PHYSICAL & HARDWARE-CONSTRAINED PARAMETERS
# =====================================================================
WORLD_SIZE = 1000.0       # 1km x 1km lake surface area
WATER_DEPTH = 1.5         # Dipole depth parameter (meters)
MAX_SPEED = 1.0           # Strict ASV velocity limit (m/s)
NUM_BOATS = 4             # Fixed 4 ASVs
GPS_RTK_ERROR = 0.05      # RTK standard deviation (5 cm)
MAGNETOMETER_NOISE = 0.8  # Low-level ambient sensor noise floor
DT = 0.5                  # Simulation time step (seconds)

class ThreeStageEmpiricalSimulation:
    def __init__(self):
        self.tick = 0
        self.target = {'x': 620.0, 'y': 580.0, 'found': False}
        self.boats = []
        self.init_fleet()

    def init_fleet(self):
        # 4-Quadrant initial deployment centers for Stage 1 (Concentric Search)
        quadrants = [
            (250.0, 250.0),   # ASV-1: Bottom-Left
            (750.0, 250.0),   # ASV-2: Bottom-Right
            (250.0, 750.0),   # ASV-3: Top-Left
            (750.0, 750.0)    # ASV-4: Top-Right
        ]
        for i in range(NUM_BOATS):
            sx, sy = quadrants[i]
            self.boats.append({
                'id': i,
                'x': sx, 'y': sy,
                'heading': random.uniform(0, 2 * math.pi),
                'mag_raw': 0.0,
                'mag_filtered': 0.0,
                'p_best_mag': 0.0,
                'p_best_x': sx,
                'p_best_y': sy,
                'state': 'STAGE_1_CONCENTRIC',
                'concentric_phase': 0.0,
                'concentric_radius': 15.0,
                'center_x': sx,
                'center_y': sy,
                'vx_filter': 0.0,
                'vy_filter': 0.0
            })

    def process_sensor_pipeline(self, x, y, boat):
        distance = math.hypot(x - self.target['x'], y - self.target['y'])
        if distance < 1.0:
            true_field = 50.0
        else:
            falloff_cubed = 80.0 ** 3
            depth_factor = (WATER_DEPTH ** 2 / (distance ** 2 + WATER_DEPTH ** 2)) ** 1.5
            true_field = 50.0 * falloff_cubed * depth_factor
        
        noise = np.random.normal(0, MAGNETOMETER_NOISE)
        raw_reading = max(0.0, true_field + noise)
        boat['mag_raw'] = raw_reading
        
        # EMA Denoising Filter (alpha = 0.30)
        alpha = 0.30
        boat['mag_filtered'] = alpha * raw_reading + (1.0 - alpha) * boat.get('mag_filtered', raw_reading)
        return boat['mag_filtered']

    def update_swarm_strategy(self):
        g_best_x, g_best_y, g_best_mag = None, None, 0.0
        for b in self.boats:
            if b['p_best_mag'] > g_best_mag:
                g_best_mag = b['p_best_mag']
                g_best_x = b['p_best_x']
                g_best_y = b['p_best_y']

        # Corrected threshold multiplier (0.15) to successfully trigger state transition
        anomaly_threshold = 50.0 * 0.15

        for b in self.boats:
            if g_best_mag >= anomaly_threshold and g_best_x is not None:
                dist_to_g = math.hypot(g_best_x - b['x'], g_best_y - b['y'])
                if dist_to_g <= 15.0:
                    b['state'] = 'STAGE_3_DPSO'
                else:
                    b['state'] = 'STAGE_2_GRADIENT'
            else:
                b['state'] = 'STAGE_1_CONCENTRIC'

            if b['state'] == 'STAGE_3_DPSO':
                w, c1, c2 = 0.4, 1.0, 1.8
                r1, r2 = random.random(), random.random()
                target_x = b['x'] + w * b['vx_filter'] + c1 * r1 * (b['p_best_x'] - b['x']) + c2 * r2 * (g_best_x - b['x'])
                target_y = b['y'] + w * b['vy_filter'] + c1 * r1 * (b['p_best_y'] - b['y']) + c2 * r2 * (g_best_y - b['y'])
                target_heading = math.atan2(target_y - b['y'], target_x - b['x'])

            elif b['state'] == 'STAGE_2_GRADIENT':
                dx = g_best_x - b['x']
                dy = g_best_y - b['y']
                target_heading = math.atan2(dy, dx)

            else:
                # Stage 1: Concentric Expanding Search Pattern
                b['concentric_phase'] += 0.4 * DT
                if b['concentric_phase'] > 2 * math.pi:
                    b['concentric_phase'] = 0.0
                    b['concentric_radius'] += 25.0

                target_x = b['center_x'] + b['concentric_radius'] * math.cos(b['concentric_phase'])
                target_y = b['center_y'] + b['concentric_radius'] * math.sin(b['concentric_phase'])
                dx = target_x - b['x']
                dy = target_y - b['y']
                target_heading = math.atan2(dy, dx)

            base_vx = math.cos(target_heading) * MAX_SPEED
            base_vy = math.sin(target_heading) * MAX_SPEED
            
            filter_alpha = 0.3
            b['vx_filter'] += (base_vx - b['vx_filter']) * filter_alpha
            b['vy_filter'] += (base_vy - b['vy_filter']) * filter_alpha
            
            b['heading'] = math.atan2(b['vy_filter'], b['vx_filter'])
            actual_v = min(MAX_SPEED, math.hypot(b['vx_filter'], b['vy_filter']))
            b['x'] += math.cos(b['heading']) * actual_v * DT
            b['y'] += math.sin(b['heading']) * actual_v * DT

            margin = 10.0
            b['x'] = np.clip(b['x'], margin, WORLD_SIZE - margin)
            b['y'] = np.clip(b['y'], margin, WORLD_SIZE - margin)

            gps_x = b['x'] + np.random.normal(0, GPS_RTK_ERROR)
            gps_y = b['y'] + np.random.normal(0, GPS_RTK_ERROR)
            filtered_mag = self.process_sensor_pipeline(gps_x, gps_y, b)

            if filtered_mag > b['p_best_mag']:
                b['p_best_mag'] = filtered_mag
                b['p_best_x'] = gps_x
                b['p_best_y'] = gps_y

            if math.hypot(b['x'] - self.target['x'], b['y'] - self.target['y']) < 3.0:
                self.target['found'] = True

    def run_simulation(self, max_ticks=36000):
        print(f"=== 3-STAGE CONCENTRIC EMPIRICAL SIMULATION ===")
        print(f"Parameters: 4 ASVs | Max Speed: {MAX_SPEED} m/s | Target at ({self.target['x']}, {self.target['y']})")
        
        while self.tick < max_ticks and not self.target['found']:
            self.tick += 1
            self.update_swarm_strategy()
            
            if self.tick % 200 == 0:
                print(f"[Tick {self.tick}] Mission Time: {self.tick * DT / 60:.1f} min | Target Found: {self.target['found']}")
                for b in self.boats:
                    print(f"  -> ASV-{b['id']+1} | State: {b['state']} | Pos: ({b['x']:.1f}, {b['y']:.1f}) | Filtered Mag: {b['mag_filtered']:.2f} nT")

        if self.target['found']:
            print(f"\n[SUCCESS] Target localized at tick {self.tick} ({self.tick * DT / 60:.2f} minutes).")
        else:
            print(f"\n[TIMEOUT] Simulation reached time limit without final convergence.")

if __name__ == '__main__':
    sim = ThreeStageEmpiricalSimulation()
    sim.run_simulation()