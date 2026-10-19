import sys
import math
import random
from ursina import *
from ursina.shaders import lit_with_shadows_shader

# =====================================================================
# COMMAND LINE ARGUMENT FOR NUMBER OF BOATS
# =====================================================================
num_boats = 3  # Default if no argument is given
if len(sys.argv) > 1:
    try:
        num_boats = int(sys.argv[1])
        if(num_boats < 3 or num_boats > 5):
            print("Number of boats must be between 3 and 5. Defaulting to 3 boats.")
            num_boats = 3
    except ValueError:
        print("Invalid argument. Defaulting to 3 boats.")

# Global simulation state control
sim_running = False

# ---------------------------------------------------------------------
# 1. GENERATE ROW-WISE SEARCH PATTERN (Lawnmower Strategy)
# ---------------------------------------------------------------------
def generate_sweep_waypoints(boat_id, total_boats):
    waypoints = []
    zone_height = 100.0 / total_boats
    z_min = boat_id * zone_height
    z_max = (boat_id + 1) * zone_height
    
    sweep_spacing = 15.0
    current_z = z_min + 5.0
    moving_right = True
    
    # Drive to their designated starting lane
    waypoints.append(Vec3(0, 0, current_z))
    
    while current_z < z_max:
        if moving_right:
            waypoints.append(Vec3(100, 0, current_z))
        else:
            waypoints.append(Vec3(0, 0, current_z))
            
        current_z += sweep_spacing
        moving_right = not moving_right
        
        if current_z < z_max:
            if not moving_right:
                waypoints.append(Vec3(100, 0, current_z))
            else:
                waypoints.append(Vec3(0, 0, current_z))
                
    return waypoints

# ---------------------------------------------------------------------
# 2. COORDINATED SWARM ALGORITHM
# ---------------------------------------------------------------------
def swarm_coordination_algorithm(boat_id, current_gps, current_reading, fleet_memory, my_waypoints, max_speed, target_pos):
    SIGNAL_THRESHOLD = 15.0  
    
    if current_reading > fleet_memory["global_best_reading"]:
        fleet_memory["global_best_reading"] = current_reading
        fleet_memory["global_best_pos"] = Vec3(current_gps.x, 0, current_gps.z)

    # Calculate distance to the physical target
    diff_target = current_gps - target_pos
    diff_target.y = 0
    dist_to_target = diff_target.length()

    # GLOBAL BRAKING RULE: If within 3 units of the target, drop desired velocity to 0
    if dist_to_target <= 3.1:
        desired_velocity = Vec3(0, 0, 0)
        
    # BEHAVIOR A: Target Found -> Swarm!
    elif fleet_memory["global_best_reading"] >= SIGNAL_THRESHOLD:
        dir_to_target = fleet_memory["global_best_pos"] - current_gps
        dir_to_target.y = 0
        if dir_to_target.length() > 0:
            desired_velocity = dir_to_target.normalized() * max_speed
        else:
            desired_velocity = Vec3(0, 0, 0)
            
    # BEHAVIOR B: Target Unknown -> Sweep Assigned Area
    else:
        if len(my_waypoints) == 0:
            return Vec3(0, 0, 0) 
            
        target_wp = my_waypoints[0]
        dir_to_wp = target_wp - current_gps
        dir_to_wp.y = 0
        
        if dir_to_wp.length() < 3.0: 
            my_waypoints.pop(0) 
            if len(my_waypoints) == 0: return Vec3(0, 0, 0)
            target_wp = my_waypoints[0]
            dir_to_wp = target_wp - current_gps
            dir_to_wp.y = 0
            
        desired_velocity = dir_to_wp.normalized() * max_speed

    # COLLISION AVOIDANCE (FLEET MUTUAL AVOIDANCE)
    repulsion = Vec3(0, 0, 0)
    for other_pos in fleet_memory["all_positions"]:
        if other_pos == current_gps: continue
        diff = current_gps - other_pos
        diff.y = 0
        dist = diff.length()
        if 0 < dist < 8.0:  
            repulsion += (diff / dist) * (8.0 - dist)
            
    # ANTI-COLLISION WITH TARGET (Enforces the strict 3-radius boundary ring)
    if dist_to_target < 3.0:
        if dist_to_target > 0:
            repulsion += (diff_target / dist_to_target) * (3.0 - dist_to_target) * 15.0
            
    final_vector = desired_velocity + (repulsion * 3.0)
    if final_vector.length() > max_speed:
        final_vector = final_vector.normalized() * max_speed
        
    return final_vector

def get_magnetic_reading(asv_pos, target_pos):
    dist = (target_pos - asv_pos).length()
    return 25000.0 / ((dist ** 3) + 1.0)

# ---------------------------------------------------------------------
# 3. 3D ENVIRONMENT & ENGINE SETUP
# ---------------------------------------------------------------------
app = Ursina()
window.title = "Coordinated ASV Swarm Environment"

# Directional Light for actual shadows
sun = DirectionalLight(shadows=True)
sun.look_at(Vec3(-1, -1, -0.5))

# Upgraded Shader: Calculates 3D surface normals to simulate shadows on the water
water_shader = Shader(language=Shader.GLSL,
vertex='''
#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform float p3d_Time;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uvs;
out float wave_height;
out vec3 world_pos;
void main() {
    vec4 v = p3d_Vertex;
    v.y += sin(v.x * 1.5 + p3d_Time * 2.0) * 0.3;
    v.y += cos(v.z * 1.2 + p3d_Time * 1.8) * 0.3;
    wave_height = v.y;
    world_pos = (p3d_ModelMatrix * v).xyz;
    gl_Position = p3d_ModelViewProjectionMatrix * v;
    uvs = p3d_MultiTexCoord0;
}
''',
fragment='''
#version 140
in vec2 uvs;
in float wave_height;
in vec3 world_pos;
out vec4 fragColor;
void main() {
    vec3 dx = dFdx(world_pos);
    vec3 dy = dFdy(world_pos);
    vec3 normal = normalize(cross(dx, dy));
    
    vec3 light_dir = normalize(vec3(-1.0, 1.0, -0.5));
    float diffuse = max(dot(normal, light_dir), 0.0);
    
    vec3 deep = vec3(0.0, 0.15, 0.45);
    vec3 peak = vec3(0.0, 0.6, 0.8);
    vec3 base_color = mix(deep, peak, wave_height * 1.5 + 0.5);
    
    vec3 final_color = base_color * (diffuse * 0.5 + 0.5);
    
    fragColor = vec4(final_color, 0.85);
}
'''
)

water = Entity(model=Plane(subdivisions=(60, 60)), scale=(100, 1, 100), position=(50, 0, 50), shader=water_shader, double_sided=True, transparent=True)

# Target
target_pos = Vec3(random.uniform(20, 80), 0, random.uniform(20, 80))
target_marker = Entity(model='sphere', color=color.red, scale=2.5, position=(target_pos.x, 0, target_pos.z), shader=lit_with_shadows_shader)

fleet_memory = {
    "global_best_reading": -1.0,
    "global_best_pos": Vec3(0, 0, 0),
    "all_positions": []
}

class AutonomousASV(Entity):
    def __init__(self, boat_id, num_total):
        # Spaced out starting grid strictly at bottom-left corner of the 100x100 map
        start_x = (boat_id % 4) * 5.0
        start_z = (boat_id // 4) * 5.0
        
        super().__init__(model='cube', color=color.orange, scale=(1.2, 0.6, 2.5), position=(start_x, 0, start_z), shader=lit_with_shadows_shader)
        self.boat_id = boat_id
        self.max_speed = 10.0
        self.waypoints = generate_sweep_waypoints(boat_id, num_total)
        
        # Smooth orientation dampener to eliminate rapid yaw spinning during proximity adjustments
        self.look_direction = Vec3(0, 0, 1)
        
        self.cabin = Entity(parent=self, model='cube', color=color.white, scale=(0.7, 1.0, 0.4), position=(0, 0.8, -0.2), shader=lit_with_shadows_shader)
        
    def update(self):
        # Keep physical wave bobbing active even when simulation is paused
        self.y = (math.sin(self.x * 1.5 + time.time() * 2.0) * 0.3 + 
                  math.cos(self.z * 1.2 + time.time() * 1.8) * 0.3)
                  
        # Guard clause: Stop algorithmic execution if the GUI button hasn't been clicked
        if not sim_running:
            return
            
        current_gps = self.position
        current_reading = get_magnetic_reading(current_gps, target_marker.position)
        
        # Passed target_marker.position into the swarm coordination loop
        vel = swarm_coordination_algorithm(
            self.boat_id, current_gps, current_reading, fleet_memory, self.waypoints, self.max_speed, target_marker.position
        )
        
        self.position += vel * time.dt
        
        # Clamped exactly to 0 and 100 so they stay on the visible map
        self.x = clamp(self.x, 0, 100)
        self.z = clamp(self.z, 0, 100)
        
        if vel.length() > 0.1:
            target_dir = vel.normalized()
            # Damps out instantaneous collision vector snaps, creating a realistic steering look
            self.look_direction += (target_dir - self.look_direction) * time.dt * 5.0
            self.look_at(self.position + self.look_direction)
            self.rotation_x = 0
            self.rotation_z = 0

fleet = [AutonomousASV(i, num_boats) for i in range(num_boats)]

# ---------------------------------------------------------------------
# 4. CAMERA & LOOP
# ---------------------------------------------------------------------
camera.position = (50, 120, -100)
camera.look_at(Vec3(50, -25, 70))
Sky()

def update():
    fleet_memory["all_positions"] = [b.position for b in fleet]
    
    target_marker.y = (math.sin(target_marker.x * 1.5 + time.time() * 2.0) * 0.3 + 
                       math.cos(target_marker.z * 1.2 + time.time() * 1.8) * 0.3)
    
    speed = 50 * time.dt 
    if held_keys['up arrow']: camera.position += camera.forward * speed
    if held_keys['down arrow']: camera.position += camera.back * speed
    if held_keys['a']: camera.position += camera.left * speed
    if held_keys['d']: camera.position += camera.right * speed
    if held_keys['w']: camera.position += camera.up * speed
    if held_keys['s']: camera.position += camera.down * speed
    
    if mouse.right:
        camera.rotation_y += mouse.velocity[0] * 150
        camera.rotation_x -= mouse.velocity[1] * 150

# UI Informational Text
Text(text=f'Fleet: {num_boats} ASVs | Status: Ready\nRight-Click + Drag: Look | WASD: Move | Arrow Keys: Zoom', 
     position=(-0.75, 0.45), scale=1.1)

# ---------------------------------------------------------------------
# 5. GUI CONTROLS
# ---------------------------------------------------------------------
def trigger_simulation():
    global sim_running
    sim_running = True
    start_button.disable() # Hide button once running

start_button = Button(
    text='Start Simulation',
    color=color.azure,
    scale=(0.25, 0.06),
    position=(0, -0.4),
    on_click=trigger_simulation
)

app.run()