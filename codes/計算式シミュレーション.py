import numpy as np
import matplotlib.pyplot as plt

# === パラメータ定義 ===
T = 20                                # 総ステップ数
start = np.array([0.0, 0.0])           # 初期位置
goal = np.array([200.0, 200.0])        # 目的地
Vmax = 2.0                            # 最大自力速度（仮定）
dt = 5                              # 各ステップの時間間隔

# === 風場の生成（時間と空間に連続的変化） ===
grid_extent = 200
grid_size = 400
wind_field = np.zeros((grid_size, grid_size, T, 2))

# for t in range(T):
#     print(f"Step {t}")
#     for i in range(grid_size):
#         for j in range(grid_size):
#             x = i - grid_extent
#             y = j - grid_extent
#             wind_field[i, j, t, 0] = 1.8*np.cos(np.pi*x*4*t) * np.sin(2 * np.pi * (x / 400 + t / 50))
#             wind_field[i, j, t, 1] = 2.4*np.sin(np.pi*x*t) * np.cos(2 * np.pi * (y / 400 + t / 50))
            
for t in range(T):
    print(f"Step {t}")
    for i in range(grid_size):
        for j in range(grid_size):
            # u, v 成分ともに 0〜5 の範囲で一様分布からランダム生成
            wind_field[i, j, t, 0] = np.random.uniform(-3, 8)  # u 成分（東向き風速）
            wind_field[i, j, t, 1] = np.random.uniform(-2, 7)  # v 成分（北向き風速）


# === f(theta)：風と進行方向の相対角度に応じた係数 ===
def wind_efficiency(theta):
    angle = np.abs(theta) % (2 * np.pi)
    if angle < np.pi / 4 or angle > 7 * np.pi / 4:
        return 1.0
    elif np.pi / 4 <= angle <= 3 * np.pi / 4:
        return 0.5
    else:
        return 0.2

# === 軌跡初期化 ===
trajectory = [start.copy()]
r = start.copy()

total_distance = np.linalg.norm(goal - start)
delta_r_avg = total_distance / T

# === メインループ ===
for t in range(T):
    print(f"Step {t}") 
    d_goal = goal - r
    dist_to_goal = np.linalg.norm(d_goal)
    direction = d_goal / dist_to_goal if dist_to_goal != 0 else np.array([0.0, 0.0])

    ideal_pos = start + (t / T) * (goal - start)
    lag = np.linalg.norm(ideal_pos - r)
    # intention = lag / delta_r_avg
    # Vself = Vmax * intention * direction

    # 意思ベクトルのスケーリング（0.3倍程度に弱める）＋ ランダム性の追加
    intention = lag / delta_r_avg
    direction_noise = np.random.normal(0, 1.0, size=2)  # ガウス分布からノイズ
    Vself = 0.2 * Vmax * intention * direction + 1.5 * direction_noise


    i = int(np.clip(r[0] + grid_extent, 0, grid_size - 1))
    j = int(np.clip(r[1] + grid_extent, 0, grid_size - 1))
    wind_vec = wind_field[i, j, t]

    if np.linalg.norm(wind_vec) != 0 and np.linalg.norm(direction) != 0:
        cos_theta = np.dot(direction, wind_vec) / (np.linalg.norm(direction) * np.linalg.norm(wind_vec))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    else:
        theta = 0.0

    wind_coef = wind_efficiency(theta)
    Vwind = wind_coef * wind_vec

    Vtotal = Vself + Vwind
    r = r + Vtotal * dt
    trajectory.append(r.copy())

trajectory = np.array(trajectory)

plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', markersize=2, label="Path")
plt.scatter([start[0]], [start[1]], color='green', label='Start')
plt.scatter([goal[0]], [goal[1]], color='red', label='Goal')
plt.title("Flight Simulation with Wind Field")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

# === 風場の可視化（t = 0） ===
plt.figure(figsize=(10, 10))
skip = 10  # 矢印の密度（間引き）
x_vals = np.arange(-grid_extent, grid_extent, skip)
y_vals = np.arange(-grid_extent, grid_extent, skip)
X, Y = np.meshgrid(x_vals, y_vals)

U = np.zeros_like(X)
V = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xi = int(X[i, j] + grid_extent)
        yj = int(Y[i, j] + grid_extent)
        U[i, j] = wind_field[xi, yj, 0, 0]
        V[i, j] = wind_field[xi, yj, 0, 1]

plt.quiver(X, Y, U, V, scale=50, width=0.002, color='gray')
plt.title("Wind Field at t = 0")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.show()