import numpy as np
from sklearn.linear_model import RANSACRegressor
import cv2
from scipy.signal import butter, filtfilt

def preprocess_lidar(data):
    points = np.array([[point.x, point.y, point.z] for point in data])

    # RANSAC
    ransac = RANSACRegressor()
    ransac.fit(points[:,:2], points[:,2])
    inliers = ransac.inlier_mask_
    # floor points
    filtered_points = points[inliers]

    return filtered_points

def preprocess_camera(image):
    image_data = np.array(image.raw_data)
    image_data = image_data.reshape((image.height,image.width,4))[:,:,:3]

    # Edge Detection
    gray = cv2.cvtColor(image_data,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)

    return edges

def low_pass_filter(data, cutoff_frequency=10, sample_rate=500, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b,a,data)
    return filtered_data



