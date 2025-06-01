import numpy as np
import matplotlib.pyplot as plt

# === 基本パラメータ ===
T = 100
dt = 6  # 時間刻み（時間単位）
x_min, x_max = -55, 110
y_min, y_max = -60, 60
grid_x = np.arange(x_min, x_max + 1)
grid_y = np.arange(y_min, y_max + 1)
X, Y = np.meshgrid(grid_x, grid_y)

# === 出発点と目的地 ===
start = np.array([81, -34], dtype=float)
goal = np.array([-10, 40], dtype=float)
Vmax = 1.0

# === エネルギーパラメータ ===
initial_energy = 100
C_basal = 1
C_fly = 3
C_float = 0.5
C_forage = 4
C_gain = 17

# === 行動モード確率 ===
mode_probs = np.array([6, 2, 2]) / 10

# === 風の初期化（緯度対称な東西風）===
wind_field = np.zeros((len(grid_y), len(grid_x), 2))
for i, lat in enumerate(grid_y):
    if np.abs(lat) > 30:
        wind_field[i, :, 0] = 5 * np.cos((np.abs(lat) - 30) / 30 * np.pi) * np.sign(lat)
    else:
        wind_field[i, :, 0] = -3 * np.sin(np.abs(lat) / 30 * np.pi)
    wind_field[i, :, 1] = 0

# # 可視化
# skip = 5
# X_plot = X[::skip, ::skip]
# Y_plot = Y[::skip, ::skip]
# U_wind = wind_field[::skip, ::skip, 0]
# V_wind = wind_field[::skip, ::skip, 1]

# plt.figure(figsize=(12, 6))
# plt.quiver(X_plot, Y_plot, U_wind, V_wind, scale=50, color='blue')
# plt.title("Zonal Wind Field")
# plt.xlabel("Longitude (x)")
# plt.ylabel("Latitude (y)")
# plt.grid(True)
# plt.axis("equal")
# plt.show()

# === 海流の初期化（全域にゆるやかな循環）===
current_field = np.zeros((len(grid_y), len(grid_x), 2))
for j, lon in enumerate(grid_x):
    for i, lat in enumerate(grid_y):
        u = 1.0 * np.sin(2 * np.pi * (lat - y_min) / (y_max - y_min))
        v = 1.0 * np.cos(2 * np.pi * (lon - x_min) / (x_max - x_min))
        current_field[i, j, 0] = u
        current_field[i, j, 1] = v

# U_current = current_field[::skip, ::skip, 0]
# V_current = current_field[::skip, ::skip, 1]

# plt.figure(figsize=(12, 6))
# plt.quiver(X_plot, Y_plot, U_current, V_current, scale=10, color='green')
# plt.title("Surface Ocean Current Field")
# plt.xlabel("Longitude (x)")
# plt.ylabel("Latitude (y)")
# plt.grid(True)
# plt.axis("equal")
# plt.show()

# シミュレーション
num_runs = 20
trajectories = []
energy_tracks = []

for run in range(num_runs):
    energy = initial_energy
    r = start.copy()
    path = [r.copy()]
    e_track = [energy]

    total_distance = np.linalg.norm(goal - start)
    delta_r_avg = total_distance / T

    for t in range(T):
        if energy < 20:
            mode = np.random.choice(["float", "forage"])
        else:
            mode = np.random.choice(["fly", "float", "forage"], p=mode_probs)

        ix = int(np.clip(r[0] - x_min, 0, len(grid_x) - 1))
        iy = int(np.clip(r[1] - y_min, 0, len(grid_y) - 1))
        wind_vec = wind_field[iy, ix]
        current_vec = current_field[iy, ix]

        if mode == "fly":
            d_goal = goal - r
            dist_to_goal = np.linalg.norm(d_goal)
            direction = d_goal / dist_to_goal if dist_to_goal > 0 else np.zeros(2)
            # 到達すべき理想距離
            ideal_dist = total_distance * (t / T)
            # 現在地から目的地への実際の残り距離
            current_dist = np.linalg.norm(goal - r)
            # ラグ：進みすぎていれば0、遅れていれば理想との差
            lag_dist = max(current_dist - (total_distance - ideal_dist), 0)
            # intention の強さ（正規化 + 上限付き）
            intention = min(lag_dist / delta_r_avg, 3)

            direction_noise = np.random.normal(0, 0.1, size=2)
            effort_factor = max(energy / initial_energy, 0.1)
            Vself = (Vmax * intention * direction + direction_noise) * effort_factor

            cos_theta = np.dot(direction, wind_vec) / (np.linalg.norm(direction) * np.linalg.norm(wind_vec)) if np.linalg.norm(wind_vec) > 0 else 0.0
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angle = np.abs(theta) % (2 * np.pi)
            if angle < np.pi / 4 or angle > 7 * np.pi / 4:
                wind_coef = 1.0  # 順風（前からの風）
            elif np.pi / 4 <= angle <= 3 * np.pi / 4 or 5 * np.pi / 4 <= angle <= 7 * np.pi / 4:
                wind_coef = 0.5  # 横風（左/右からの風）
            else:
                wind_coef = 0.2  # 向かい風（後ろからの風）

            Vwind = wind_coef * wind_vec
            Vtotal = (2*Vself + Vwind)/3
            energy -= (C_basal + C_fly)

        elif mode == "float":
            Vtotal = 0.2 * wind_vec + 0.2 * current_vec
            energy -= (C_basal + C_float)

        elif mode == "forage":
            Vtotal = np.random.normal(0, 1.0, size=2)
            energy += (C_gain - C_forage - C_basal)

        r = r + Vtotal * dt
        energy = max(0, energy)
        path.append(r.copy())
        e_track.append(energy)

    trajectories.append(np.array(path))
    energy_tracks.append(e_track)

# === プロット1: 移動軌跡 ===
plt.figure(figsize=(10, 6))
for path in trajectories:
    plt.plot(path[:, 0], path[:, 1], '-o', markersize=2)
plt.scatter([start[0]], [start[1]], color='green', label='Start')
plt.scatter([goal[0]], [goal[1]], color='red', label='Goal')
plt.title("Migration Trajectories")
plt.xlabel("Longitude (x)")
plt.ylabel("Latitude (y)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

# === プロット2: エネルギー推移 ===
plt.figure(figsize=(10, 4))
for track in energy_tracks:
    plt.plot(track)
plt.title("Energy Transition Over Time")
plt.xlabel("Time Step")
plt.ylabel("Energy")
plt.grid(True)
plt.show()