import numpy as np 
import math 
import matplotlib.pyplot as plt 
deltat = 0.5

#state transitiion model 
Px=0 #groud truth 
Py=0  #ground truth 
l1 = [5,5]
l2 = [-5,5]
ground_truth_positions =[]
EKF_predicted_positions = []
for t in np.arange(0,40,0.5):
    if 0 <= t <= 10:
        Vx = 1
        Vy = 0
    elif 10 < t <= 20:
        Vx= 0
        Vy =-1
    elif 20 < t <= 30:
        Vx= -1
        Vy =0
    elif 30 < t <= 40:
        Vx= 0
        Vy = 1
    R =np.array(
        [[0.1,0],
        [0,0.1]])         #process noise 
    Q =np.array(
        [[0.5,0],
        [0,0.5]])         #measurement noise 
    pn1 = np.random.normal(0, np.sqrt(R[0,0]))
    pn2 = np.random.normal(0, np.sqrt(R[1,1]))
    mn1 = np.random.normal(0, np.sqrt(Q[0,0]))
    mn2 = np.random.normal(0, np.sqrt(Q[1,1]))
    Px += Vx*deltat 
    Py += Vy*deltat 
    ground_truth_positions.append([Px,Py])
    predicted_mean = np.array ([Px+pn1, Py+pn2])
    predicted_covariance = R  #because the posterior distribution of the previous state is Identity (I2) 

    #predicted masurements 
    predicted_range1 = math.sqrt(((predicted_mean[0]-l1[0])**2)+(predicted_mean[1]-l1[1])**2 )
    predicted_range2 = math.sqrt(((predicted_mean[0]-l2[0])**2)+(predicted_mean[1]-l2[1])**2 )

    # measurements 
    actual_range1 = math.sqrt(((Px-l1[0])**2)+((Py-l1[1])**2)) +mn1
    actual_range2 = math.sqrt(((Px-l2[0])**2)+((Py-l2[1])**2)) +mn2

    #residual computation 
    Zk = [(actual_range1-predicted_range1),(actual_range2-predicted_range2)]

    #linearized measurement model 
    J = np.array(
        [[(predicted_mean[0]-l1[0]/predicted_range1),(predicted_mean[1]-l1[1]/predicted_range2)],
        [(predicted_mean[0]+l2[0]/predicted_range2), (predicted_mean[1]-l2[1]/predicted_range2)]]

    )
    # compute the Kalman gain 
    A = np.matmul(predicted_covariance,np.transpose(J)) 
    B = np.matmul(J, predicted_covariance)
    C = np.matmul(B,np.transpose(J))
    D = np.linalg.inv(C+Q)

    Kt = np.matmul(A, D)

    # Compute the approximate mean and covariance of the posterior belief 

    posterior_mean = np.array(predicted_mean) + Kt @ np.array(Zk)
    I = np.eye(2)
    posterior_covariance = (I - Kt @ J) @ predicted_covariance
    EKF_predicted_positions.append([posterior_mean[0],posterior_mean[1]])

ground_truth_positions = np.array(ground_truth_positions)
EKF_predicted_positions = np.array(EKF_predicted_positions)

# Plot ground truth and EKF estimates
plt.figure(figsize=(8,6))
plt.plot(ground_truth_positions[:,0], ground_truth_positions[:,1], label='Ground Truth', color='blue')
plt.plot(EKF_predicted_positions[:,0], EKF_predicted_positions[:,1], label='EKF Estimate', color='red')
plt.scatter([5, -5], [5, 5], color='green', marker='x', s=100, label='Landmarks')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Robot Localization with EKF')
plt.legend()
plt.grid(True)
plt.show()
