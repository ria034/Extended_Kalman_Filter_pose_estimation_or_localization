import numpy as np

# PARAMETERS
num_landmarks = 50
x_range = [-20, 20]
y_range = [-20, 20]
z_range = [0, 15]

# GENERATE 3D LANDMARK DATA
landmarks_3d = np.zeros((num_landmarks, 3))
landmarks_3d[:,0] = np.random.uniform(x_range[0], x_range[1], num_landmarks)
landmarks_3d[:,1] = np.random.uniform(y_range[0], y_range[1], num_landmarks)
landmarks_3d[:,2] = np.random.uniform(z_range[0], z_range[1], num_landmarks)

# SAVE TO FILE
np.savetxt("landmarks_3d.csv", landmarks_3d, delimiter=",", header="x,y,z", comments="")

# PRINT SAMPLE
print("3D Landmarks (sample):")
print(landmarks_3d[:5])