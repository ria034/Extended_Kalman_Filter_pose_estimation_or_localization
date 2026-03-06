import numpy as np

# PARAMETERS
num_landmarks = 50
x_range = [-20, 20]
y_range = [-20, 20]

# GENERATE 2D LANDMARK DATA
landmarks_2d = np.zeros((num_landmarks, 2))
landmarks_2d[:,0] = np.random.uniform(x_range[0], x_range[1], num_landmarks)
landmarks_2d[:,1] = np.random.uniform(y_range[0], y_range[1], num_landmarks)

# SAVE TO FILE
np.savetxt("landmarks_2d.csv", landmarks_2d, delimiter=",", header="x,y", comments="")

# PRINT SAMPLE
print("2D Landmarks (sample):")
print(landmarks_2d[:5])