import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
T = 200  # ステップ数
num_runs = 10  # アンサンブル本数
initial_energy = 100
C_basal = 1
C_fly = 3
C_float = 0.5
C_forage = 4
C_gain = 17

# モードの確率（合計10で正規化）
mode_probs = [6, 2, 2]  # fly, float, forage の比率
mode_probs = np.array(mode_probs) / sum(mode_probs)

# 結果格納
energy_trajectories = []

for run in range(num_runs):
    print(f"Step {run}")
    energy = initial_energy
    trajectory = [energy]

    for t in range(T):
        if energy < 20:
            mode = np.random.choice(["float", "forage"])
        else:
            mode = np.random.choice(["fly", "float", "forage"], p=mode_probs)

        # エネルギー変化
        if mode == "fly":
            energy -= (C_basal + C_fly)
        elif mode == "float":
            energy -= (C_basal + C_float)
        elif mode == "forage":
            energy += (C_gain - C_forage - C_basal)

        energy = max(0, energy)  # 負の値を防ぐ
        trajectory.append(energy)

    energy_trajectories.append(trajectory)

# 可視化
for traj in energy_trajectories:
    plt.plot(traj)
plt.title("Energy Trajectories (10 Simulations)")
plt.xlabel("Time Step")
plt.ylabel("Energy")
plt.grid(True)
plt.show()
