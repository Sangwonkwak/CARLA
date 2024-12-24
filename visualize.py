import matplotlib.pyplot as plt

def visualize_data(camera_image, lidar_points):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Camera Image")
    plt.imshow(camera_image)

    plt.subplot(1,2,2)
    plt.title("LiDAR Point Cloud")
    plt.scatter(lidar_points[:,0], lidar_points[:,1],s=1)
    plt.show()