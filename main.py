import carla
import time
import Sensors


def main():
    world, ego_vehicle, sensors = Sensors.setup_carla()
    sensors[0].listen(Sensors.lidar_callback)
    sensors[1].listen(Sensors.camera_callback)
    sensors[2].listen(Sensors.imu_callback)
    sensors[3].listen(Sensors.gps_callback)
    # wait time for collecting data
    time.sleep(5)




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone')