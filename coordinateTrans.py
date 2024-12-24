import numpy as np

def transform_to_vehicle_frame(sensor_data, transform_matrix):
    points = np.array(sensor_data)
    points_homogeneous = np.hstack((points,np.ones((points.shape[0],1))))
    transformed_points = np.dot(transform_matrix, points_homogeneous.T).T
    return transformed_points[:,:3]
