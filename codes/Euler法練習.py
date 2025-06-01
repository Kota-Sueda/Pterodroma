import numpy as np
import matplotlib.pyplot as plt

# Euler法 ===============================================================

# 初期条件
x, y = 0.0, 0.0  # 初期位置
vx_self, vy_self = 2.0, 1.0  # 鳥の自己速度
vx_wind, vy_wind = 0.5, 1.0  # 風速

# 時間設定
dt = 0.1  # タイムステップ（秒）
T = 10.0  # 合計時間（秒）
steps = int(T / dt)

# 結果記録用リスト
xs, ys = [x], [y]

# Euler法で位置を更新
for _ in range(steps):
    vx_total = vx_self + vx_wind
    vy_total = vy_self + vy_wind
    x += vx_total * dt
    y += vy_total * dt
    xs.append(x)
    ys.append(y)

# 結果を描画
plt.figure(figsize=(6, 6))
plt.plot(xs, ys, marker='o', markersize=2)
plt.title("Euler法による移動軌跡")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.show()


# Runge-Kutta =============================================================

# 初期条件
x, y = 0.0, 0.0  # 初期位置
vx_self, vy_self = 2.0, 1.0  # 自己速度
vx_wind, vy_wind = 1.0, 0.5  # 風速

# 時間設定
dt = 0.1  # タイムステップ
T = 10.0  # 総時間
steps = int(T / dt)

# 結果記録用リスト
xs, ys = [x], [y]

# 速度関数（単純に一定）
def velocity(t, pos):
    vx_total = vx_self + vx_wind
    vy_total = vy_self + vy_wind
    return np.array([vx_total, vy_total])

# Runge-Kutta 4次法で位置を更新
for step in range(steps):
    t = step * dt
    r = np.array([x, y])

    k1 = velocity(t, r)
    k2 = velocity(t + dt/2, r + dt * k1 / 2)
    k3 = velocity(t + dt/2, r + dt * k2 / 2)
    k4 = velocity(t + dt, r + dt * k3)

    r_next = r + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    x, y = r_next
    xs.append(x)
    ys.append(y)

# 結果描画
plt.figure(figsize=(6, 6))
plt.plot(xs, ys, marker='o', markersize=2, label='RK4')
plt.title("Runge-Kutta法による移動軌跡")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()


# Euler法　時間により風ベクトル変化 ==============================================

# 初期位置と自己速度
x, y = 0.0, 0.0
vx_self, vy_self = 2.0, 1.0

# 時間設定
dt = 1
T = 10.0
steps = int(T / dt)

# 結果記録用リスト
xs, ys = [x], [y]

# 時間依存の風ベクトル関数
def wind(t):
    vx_wind = t * np.sin(t)
    vy_wind = 3 * np.cos(t)
    return np.array([vx_wind, vy_wind])

# Euler法で位置更新
for step in range(steps):
    t = step * dt
    vx_total = vx_self + wind(t)[0]
    vy_total = vy_self + wind(t)[1]
    
    x += vx_total * dt
    y += vy_total * dt
    
    xs.append(x)
    ys.append(y)

# 結果を描画
plt.figure(figsize=(6, 6))
plt.plot(xs, ys, marker='o', markersize=2, color='orange', label='Euler with time-varying wind')
plt.title("時間変化する風を含む Euler法による移動")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()


# Runge-Kutta法　時間により風ベクトル変化 =============================================

# 初期位置と自己速度（鳥自身の力による速度）
x, y = 0.0, 0.0
vx_self, vy_self = 2.0, 1.0

# 時間設定
dt = 1
T = 10.0
steps = int(T / dt)

# 結果記録用リスト
xs, ys = [x], [y]

# 時間依存の風ベクトル
def wind(t):
    vx_wind = t * np.sin(t)
    vy_wind = 3 * np.cos(t)
    return np.array([vx_wind, vy_wind])

# 速度ベクトルの合成（自己速度 + 風）
def velocity(t, pos):
    return np.array([vx_self, vy_self]) + wind(t)

# RK4で位置更新
for step in range(steps):
    t = step * dt
    r = np.array([x, y])

    k1 = velocity(t, r)
    k2 = velocity(t + dt/2, r + dt * k1 / 2)
    k3 = velocity(t + dt/2, r + dt * k2 / 2)
    k4 = velocity(t + dt, r + dt * k3)

    r_next = r + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    x, y = r_next
    xs.append(x)
    ys.append(y)

# 結果を描画
plt.figure(figsize=(6, 6))
plt.plot(xs, ys, marker='o', markersize=2, label='RK4 with time-varying wind')
plt.title("時間変化する風を含む Runge-Kutta法による移動")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()

# 時間とともに減少する Vself を含む RK4シミュレーションコード ==================================================

# 初期位置
x, y = 0.0, 0.0

# 初期の自己速度
vx0_self, vy0_self = 2.0, 1.0
alpha = 0.1  # 減衰係数（大きいほど早く減速）

# 時間設定
dt = 1
T = 20.0
steps = int(T / dt)

# 結果記録用リスト
xs, ys = [x], [y]

# 時間依存の風
def wind(t):
    return np.array([t * np.sin(t), 3.5 * np.cos(t)])

# 時間依存の自己速度（エネルギー消費による減速）
def v_self(t):
    factor = np.exp(-alpha * t)
    return np.array([vx0_self, vy0_self]) * factor

# 速度ベクトル（自己速度 + 風）
def velocity(t, pos):
    return v_self(t) + wind(t)

# Runge-Kutta 4次法で位置更新
for step in range(steps):
    t = step * dt
    r = np.array([x, y])

    k1 = velocity(t, r)
    k2 = velocity(t + dt/2, r + dt * k1 / 2)
    k3 = velocity(t + dt/2, r + dt * k2 / 2)
    k4 = velocity(t + dt, r + dt * k3)

    r_next = r + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    x, y = r_next
    xs.append(x)
    ys.append(y)

# 結果描画
plt.figure(figsize=(6, 6))
plt.plot(xs, ys, marker='o', markersize=2)
plt.title("エネルギー消費によって減速する自己速度を持つ移動（RK4）")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.show()
