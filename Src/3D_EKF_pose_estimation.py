import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# PARAMETERS
# ----------------------
deltat = 0.5
Px, Py, Pz = 0, 0, 0   # ground truth initial position
Vx, Vy, Vz = 0, 0, 0

# 3D landmarks
l1 = np.array([5, 5, 5])
l2 = np.array([-5, 5, 10])
l3 = np.array([10, -5, 3])
landmarks = [l1, l2, l3]

# Covariances
R = np.eye(3)*0.01    # process noise
Q = np.eye(3)*0.5    # measurement noise

# Storage
ground_truth_positions = []
EKF_predicted_positions = []
EKF_covariances = []

# ----------------------
# SIMULATION LOOP
# ----------------------
for t in np.arange(0, 40, deltat):
    # Velocity commands (example 3D trajectory)
    if 0 <= t <= 10:
        Vx, Vy, Vz = 1, 0, 0.2
    elif 10 < t <= 20:
        Vx, Vy, Vz = 0, -1, 0
    elif 20 < t <= 30:
        Vx, Vy, Vz = -1, 0, -0.2
    elif 30 < t <= 40:
        Vx, Vy, Vz = 0, 1, 0

    # Add process noise
    pn = np.random.multivariate_normal([0,0,0], R)

    # Update ground truth
    Px += Vx*deltat
    Py += Vy*deltat
    Pz += Vz*deltat
    ground_truth_positions.append([Px, Py, Pz])

    # Predicted state with process noise
    predicted_mean = np.array([Px, Py, Pz]) + pn
    predicted_covariance = R.copy()

    # Measurements (range to each landmark + measurement noise)
    Zk = []
    J = []
    for lm in landmarks:
        mn = np.random.multivariate_normal([0,0,0], Q)
        predicted_range = np.linalg.norm(predicted_mean - lm)
        actual_range = np.linalg.norm(np.array([Px, Py, Pz]) - lm) + mn[0]
        Zk.append(actual_range - predicted_range)
        # Linearized measurement jacobian
        jac = (predicted_mean - lm)/predicted_range
        J.append(jac)
    
    Zk = np.array(Zk)
    J = np.array(J)   # shape: (num_landmarks, 3)

    # Compute Kalman Gain
    A = predicted_covariance @ J.T
    B = J @ predicted_covariance
    C = B @ J.T
    D = np.linalg.inv(C + Q[:len(landmarks), :len(landmarks)])
    Kt = A @ D

    # Update EKF
    posterior_mean = predicted_mean + Kt @ Zk
    posterior_covariance = (np.eye(3) - Kt @ J) @ predicted_covariance

    # Store EKF estimates
    EKF_predicted_positions.append(posterior_mean)
    EKF_covariances.append(posterior_covariance)

ground_truth_positions = np.array(ground_truth_positions)
EKF_predicted_positions = np.array(EKF_predicted_positions)
EKF_covariances = np.array(EKF_covariances)

# ----------------------
# PLOT 3D TRAJECTORY
# ----------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(ground_truth_positions[:,0], ground_truth_positions[:,1], ground_truth_positions[:,2],
        label='Ground Truth', color='blue')
ax.plot(EKF_predicted_positions[:,0], EKF_predicted_positions[:,1], EKF_predicted_positions[:,2],
        label='EKF Estimate', color='red')
ax.scatter(*zip(*landmarks), color='green', marker='x', s=100, label='Landmarks')

# 3-sigma confidence ellipsoids (simplified as spheres)
for mean, cov in zip(EKF_predicted_positions[::5], EKF_covariances[::5]):
    sigma = 3 * np.sqrt(np.diag(cov))
    u, v = np.mgrid[0:2*np.pi:8j, 0:np.pi:4j]
    x = sigma[0]*np.cos(u)*np.sin(v) + mean[0]
    y = sigma[1]*np.sin(u)*np.sin(v) + mean[1]
    z = sigma[2]*np.cos(v) + mean[2]
    ax.plot_wireframe(x, y, z, color='orange', alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Robot Localization with EKF with higer process noise')
ax.legend()
plt.show()