import numpy as np
import math
import matplotlib.pyplot as plt

deltat = 0.5

# Initial robot state
Px = 0
Py = 0

# Landmarks
l1 = [5, 5]
l2 = [-5, 5]

# Covariances
R = np.array([[0.1, 0], [0, 0.1]])  # process noise
Q = np.array([[0.5, 0], [0, 0.5]])  # measurement noise

# Storage
ground_truth_positions = []
EKF_means = []
EKF_covariances = []

# Initial belief
mu = np.array([0.0, 0.0])
Sigma = np.eye(2)

for t in np.arange(0, 40, deltat):
    # Velocity commands
    if 0 <= t <= 10:
        Vx, Vy = 1, 0
    elif 10 < t <= 20:
        Vx, Vy = 0, -1
    elif 20 < t <= 30:
        Vx, Vy = -1, 0
    elif 30 < t <= 40:
        Vx, Vy = 0, 1

    # True robot motion
    pn = np.random.multivariate_normal([0, 0], R)  # process noise
    Px += Vx * deltat + pn[0]
    Py += Vy * deltat + pn[1]
    ground_truth_positions.append([Px, Py])

    # Predicted state
    predicted_mean = mu + np.array([Vx, Vy]) * deltat
    predicted_cov = Sigma + R

    # Measurements with noise
    mn = np.random.multivariate_normal([0, 0], Q)
    z1 = math.sqrt((Px - l1[0])**2 + (Py - l1[1])**2) + mn[0]
    z2 = math.sqrt((Px - l2[0])**2 + (Py - l2[1])**2) + mn[1]
    z = np.array([z1, z2])

    # Predicted measurements
    pred_r1 = math.sqrt((predicted_mean[0] - l1[0])**2 + (predicted_mean[1] - l1[1])**2)
    pred_r2 = math.sqrt((predicted_mean[0] - l2[0])**2 + (predicted_mean[1] - l2[1])**2)
    pred_z = np.array([pred_r1, pred_r2])

    # Jacobian H
    H = np.array([
        [(predicted_mean[0] - l1[0]) / pred_r1, (predicted_mean[1] - l1[1]) / pred_r1],
        [(predicted_mean[0] - l2[0]) / pred_r2, (predicted_mean[1] - l2[1]) / pred_r2]
    ])

    # Kalman gain
    S = H @ predicted_cov @ H.T + Q
    K = predicted_cov @ H.T @ np.linalg.inv(S)

    # Update
    mu = predicted_mean + K @ (z - pred_z)
    Sigma = (np.eye(2) - K @ H) @ predicted_cov

    # Store
    EKF_means.append(mu.copy())
    EKF_covariances.append(Sigma.copy())

# Convert to arrays
ground_truth_positions = np.array(ground_truth_positions)
EKF_means = np.array(EKF_means)
EKF_covariances = np.array(EKF_covariances)

# 3σ bounds
sigma_x = 3 * np.sqrt(EKF_covariances[:, 0, 0])
sigma_y = 3 * np.sqrt(EKF_covariances[:, 1, 1])

# Plot
plt.figure(figsize=(8, 6))
plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], label='Ground Truth', color='blue')
plt.plot(EKF_means[:, 0], EKF_means[:, 1], label='EKF Mean', color='red')
plt.fill_between(EKF_means[:, 0], EKF_means[:, 1] - sigma_y, EKF_means[:, 1] + sigma_y, color='red', alpha=0.2, label='3σ Y bound')
plt.fill_betweenx(EKF_means[:, 1], EKF_means[:, 0] - sigma_x, EKF_means[:, 0] + sigma_x, color='red', alpha=0.1, label='3σ X bound')
plt.scatter([l1[0], l2[0]], [l1[1], l2[1]], color='green', marker='x', s=100, label='Landmarks')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Localization with EKF (Mean ± 3σ)')
plt.legend()
plt.grid(True)
plt.show()