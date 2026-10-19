"""
500-Trial Benchmark Runner
Imports the core Simulation engine and logic directly from niot_adaptive_swarm_backend.py
"""

import random
import statistics
import json
import time

# Import core simulation classes and constants from the backend module
from play import Simulation, ROUTES, N_BOATS, GROUND_STATION

def run_single_trial(trial_id, tx, ty):
    sim = Simulation()
    # Inject randomized target coordinates
    sim.set_target(tx, ty)
    sim.running = True
    
    start_real_time = time.time()
    max_steps = 20000  # Safety ceiling to prevent infinite loops
    steps = 0
    
    # Run the backend step loop until the target is found or max steps reached
    while not sim.target["found"] and sim.running and steps < max_steps:
        sim.step()
        steps += 1
        
    real_duration = time.time() - start_real_time
    virtual_time = sim.virtual_time
    
    detected = 1 if sim.target["found"] else 0
    total_swaps = sum(b.swaps for b in sim.boats)
    # Total energy consumed across all 4 boats in Wh
    total_energy_wh = sum((296.0 - (b.battery_j / 3600.0)) for b in sim.boats)
    
    return {
        "trial": trial_id,
        "detected": detected,
        "virtual_time_h": virtual_time / 3600.0,
        "real_execution_time_s": real_duration,
        "battery_swaps": total_swaps,
        "energy_wh": total_energy_wh,
        "steps": steps
    }

def main():
    rng = random.Random(20260914)
    n_trials = 500
    results = []
    
    print(f"Initializing 500-trial batch benchmark using niot_adaptive_swarm_backend engine...")
    batch_start = time.time()
    
    for i in range(1, n_trials + 1):
        # Random target placement within 100m - 900m lake boundaries
        tx = rng.uniform(100, 900)
        ty = rng.uniform(100, 900)
        res = run_single_trial(i, tx, ty)
        results.append(res)
        
        if i % 50 == 0:
            print(f"Progress: {i}/{n_trials} trials completed...")
            
    total_batch_time = time.time() - batch_start
    
    detected_results = [r for r in results if r["detected"] == 1]
    times_h = [r["virtual_time_h"] for r in detected_results]
    real_times = [r["real_execution_time_s"] for r in results]
    swaps = [r["battery_swaps"] for r in results]
    energies = [r["energy_wh"] for r in results]
    
    def pct(arr, p):
        if not arr: return 0.0
        s = sorted(arr)
        k = (len(s)-1)*p/100.0
        f, c = int(k), int(k)+1
        if c >= len(s): c = len(s)-1
        return s[f] if f==c else s[f] + (s[c]-s[f])*(k-f)

    summary = {
        "total_trials": n_trials,
        "detected_trials": len(detected_results),
        "detection_success_rate_pct": (len(detected_results)/n_trials)*100.0,
        "mean_mission_time_h": statistics.mean(times_h) if times_h else 0,
        "median_mission_time_h": statistics.median(times_h) if times_h else 0,
        "p90_mission_time_h": pct(times_h, 90),
        "p95_mission_time_h": pct(times_h, 95),
        "max_mission_time_h": max(times_h) if times_h else 0,
        "mean_battery_swaps": statistics.mean(swaps),
        "max_battery_swaps": max(swaps),
        "mean_energy_wh": statistics.mean(energies),
        "max_energy_wh": max(energies),
        "pct_completed_within_5h": (sum(1 for t in times_h if t <= 5.0)/len(detected_results))*100.0 if detected_results else 0,
        "total_benchmark_real_duration_s": total_batch_time,
        "mean_real_execution_time_per_trial_ms": (statistics.mean(real_times)) * 1000.0
    }
    
    print("\n==========================================")
    print("      BACKEND 500-TRIAL BENCHMARK RESULTS")
    print("==========================================")
    print(json.dumps(summary, indent=2))
    
    with open("backend_500_trial_benchmark_report.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()