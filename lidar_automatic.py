#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Automatic control"""

from __future__ import print_function

import argparse
import collections
import glob
import logging
import math
import os
import numpy.random as random
import re
import sys
import weakref
import numpy as np
import kalmanfiltering as km
import utm
from lidar_to_ros2 import CarlaLidarPublisher
import rclpy
import time
# import pyproj
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
import open3d as o3d
from matplotlib import cm
from datetime import datetime


# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

# import carla

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


def lidar_callback(point_cloud, point_list, ros2_publisher):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    ####
    ros2_publisher.process_lidar_data(data)
    ####
    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    # points[:, :1] = -points[:, :1] # 내가 수정

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def generate_lidar_bp(arg, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    
    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))

    return lidar_bp

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('Fail')
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/PythonAPI/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

class Transform_data():
    """
    예측된 차량의 transformer 
    """
    def __init__(self,x=0.0,y=0.0,yaw=0.0,vx=0.0,vy=0.0,steering=0.0):
        self.set_val(x,y,yaw,vx,vy,steering)
    
    def yaw2headvec(self):
        return carla.Vector3D(math.cos(self.yaw), math.sin(self.yaw), 0.0)

    def set_val(self,x,y,yaw,vx,vy,steering):
        self.x = x
        self.y = y
        self.location = carla.Location(x,y,z=0.0)
        self.yaw = yaw # radians
        self.radian_yaw = math.degrees(yaw)
        self.head_vector = self.yaw2headvec() # [x_dir,y_dir,z_dir]
        self.v = carla.Vector3D(vx,vy,0.0)
        self.steering = steering
    
    def get_speed(self):
        return math.sqrt(self.v.x**2 + self.v.y**2) # m/s
    
    def distance(self, location):
        return math.sqrt((location.x-self.x)**2 + (location.y-self.y)**2)
    
    def get_location(self):
        return self.location
    
    def get_forward_vector(self):
        return self.head_vector
    
    def get_right_vector(self):
        R = np.array([[0,-1,0],
                      [1,0,0],
                      [0,0,1]]) # 90도 회전
        
        H = np.array([self.head_vector.x,self.head_vector.y,self.head_vector.z])
        A = R @ H
        return carla.Vector3D(A[0],A[1],A[2])


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, args, ros2_publisher, delta):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        self.delta = delta
        
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        ########
        self.gnss_sensor = None
        self.imu_sensor = None
        self.imu_data_handler = None
        self.lidar_sensor = None
        self.ros2_publisher = ros2_publisher
        self.point_list = o3d.geometry.PointCloud()

        ########
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        # self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        # blueprint = random.choice(blueprint_list)
        blueprint = self.world.get_blueprint_library().find('vehicle.mini.cooper_s')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_index = 20
            spawn_point = spawn_points[spawn_index]
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = generate_lidar_bp(args, blueprint_library, self.delta)
        user_offset = carla.Location(args.x, args.y, args.z)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.player)
        # try:
        #     self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.player)
        # except Exception as e:
        #     print(f"An unexpected error occurred: {e}")
        #     return
        self.lidar_sensor.listen(lambda data: lidar_callback(data, self.point_list, self.ros2_publisher))

        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        ########
        

    def set_player_trans(self, player_transform):
        self._player_transform = player_transform

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.lidar_sensor,
            self.imu_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================

class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.yaw = None
        self.accel = None
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        bp_trans = carla.Transform(carla.Location(x=0.25,y=0.0,z=-0.3)) # 차량의 질량 중심위치

        self.sensor = world.spawn_actor(bp, bp_trans, attach_to=self._parent)
        # print(f'location: {self.sensor.get_transform().location}, rotation: {self.sensor.get_transform().rotation}')
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda imu_data: IMUSensor.imu_callback(weak_self, imu_data))

    def get_val(self):
        return self.yaw, self.accel
    
    @staticmethod
    def imu_callback(weak_self, imu_data):
        self = weak_self()
        if not self:
            return
        
        orientation = imu_data.transform.rotation # degree로 제공된다
        self.yaw = math.radians(orientation.yaw) # degree to radian
        self.local_to_world(orientation,imu_data.accelerometer)

    def local_to_world(self, orientation, accel_data):
        yaw, pitch, roll = math.radians(orientation.yaw), math.radians(orientation.pitch), math.radians(orientation.roll)
        R_z = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
        R_y = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
        R_x = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
        # Combined rotation matrix
        R = R_z @ R_y @ R_x

        temp = R @ np.array([accel_data.x,accel_data.y,accel_data.z])
        self.accel = carla.Vector3D(temp[0],temp[1],temp[2])
        

class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        self.location = None
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        # bp_trans = carla.Transform(carla.Location(x=1.0, z=2.8))
        bp_trans = carla.Transform(carla.Location(x=0.25,y=0.0,z=-0.3)) # 차량의 질량 중심위치
        self.sensor = world.spawn_actor(blueprint, bp_trans, attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    def get_val(self):
        return self.location
    
    def xy_from_gnss(self):
        # 기준점의 UTM 좌표(offset)
        utm_origin_x = 833978.0 if self.lon < 0.0 else 166022.0
        utm_origin_y = 10000000.0 if self.lat < 0.0 else 0.0
        # gnss -> utm
        utm_coords = utm.from_latlon(self.lat,self.lon)
        utm_x, utm_y = utm_coords[0], utm_coords[1]
        # zone_number, zone_letter = utm_coords[2], utm_coords[3]  # UTM 존 정보
        
        # UTM -> CARLA 로컬좌표
        carla_x = utm_x - utm_origin_x if self.lon != 0.0 else 0.0
        carla_y = -utm_y + utm_origin_y if self.lat != 0.0 else 0.0

        self.location = carla.Location(carla_x,carla_y,0.0)
    
    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        # print("GNSS CALLBACK")
        self.lat = event.latitude
        self.lon = event.longitude
        self.xy_from_gnss()
    
    # @staticmethod
    # def cal_v(pre_data, cur_data):
    #     v = np.array([cur_data[0]-pre_data[0],
    #                   cur_data[1]-pre_data[1],
    #                     0.0])
    #     v_scale = math.sqrt(v[0]**2 + v[1]**2)
    #     v *= v_scale
    #     return v


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

class temp_accel():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

def cal_change(pre_val, cur_val, dt):
    return carla.Vector3D((cur_val.x-pre_val.x)/dt, (cur_val.y-pre_val.y)/dt,0.0)

def spin_executor(executor):
    """스레드에서 spin을 실행하는 함수"""
    executor.spin()

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    world = None
    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        world_time_step = 0.1 # 0.05
        original_settings = sim_world.get_settings()
        args.sync = True # 
        if args.sync:
            print("SYNC")
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = world_time_step
            sim_world.apply_settings(settings) # apply_settings 호출이 비동기적으로 작동할 수 있어서 변경된 값이 바로 반영 안될 수 있음.

            traffic_manager.set_synchronous_mode(True)

        
        
        ######## Node for ROS2 publisher
        rclpy.init()
        channels = 16
        upper_fov = 15.0
        lower_fov = -25.0
        lidar_node = CarlaLidarPublisher(channels, upper_fov-lower_fov, client.get_world())
        executor = MultiThreadedExecutor()
        executor.add_node(lidar_node)
        try:
            spin_thread = Thread(target=spin_executor, args=(executor,))
            spin_thread.start()
        except Exception as e:
            print(f"Exception in spin thread: {e}")

        print(f"Executor nodes: {executor.get_nodes()}")
        #################################

        ####Open3D setting####
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True
        # if args.show_axis:
        #     add_open3d_axis(vis)
        
        world = World(client.get_world(), args, lidar_node, world_time_step)
        
        ##args setting#####
        args.loop = True
        args.mine = False
        args.agent = "Basic"
        ###################


        ######variables setting for Kalman Filter#####
        player_point = world.player.get_transform()
        player_transform = Transform_data(player_point.location.x,player_point.location.y,player_point.rotation.yaw)
        world.set_player_trans(player_transform)
        pre_imu_data = (0.0, temp_accel()) # t-1시점의 (yaw,a)
        pre_gnss_data = world.gnss_sensor.get_val() # t-1시점의 (x,y,z)
        if args.mine:
            km_filter = km.KFilter(player_transform.x,player_transform.y,vx=0.0,vy=0.0,ax=0.0,ay=0.0,delta=0.0,
                            dt=0.0)

        pre_vel = carla.Vector3D(0.0,0.0,0.0)
        prev_time = world.world.get_snapshot().timestamp.elapsed_seconds  # 이전 시간
        pre_true_vel = world.player.get_velocity()
        pre_veh_vel = carla.Vector3D(0.0,0.0,0.0)
        pre_gnss_vel = np.array([0.0,0.0])
        pre_pos = player_point.location
        # pre_gnss_data = world.gnss_sensor.get_val()
        
        #################################################

        
        target_speed = 30
        if args.agent == "Basic":
            agent = BasicAgent(world.player, target_speed, player_transform)
            agent.follow_speed_limits(False)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, target_speed)
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        
        # destination = random.choice(spawn_points).location
        des_index = 30
        destination = spawn_points[des_index].location
        if args.mine:
            agent.my_set_destination(destination)
        else:
            agent.set_destination(destination)

        
        # print(world.player.get_physics_control())
        frame = 0
        dt0 = datetime.now()
        while True:
            if frame == 20:
                vis.add_geometry(world.point_list)
            vis.update_geometry(world.point_list)
            vis.poll_events()
            vis.update_renderer()

             # # This can fix Open3D jittering issues:
            time.sleep(0.005)
            world.world.tick()

            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1
            # if args.sync:
            #     world.world.tick()
            # else:
            #     world.world.wait_for_tick()
            
            # world.tick(clock) # world.world.tick() 이후에 받은 데이터로 info_text작성
            
            if agent.done():
                if args.loop:
                    des_index = (des_index+10) % len(spawn_points)
                    destination = spawn_points[des_index].location
                    if args.mine:
                        agent.my_set_destination(destination)
                    else:
                        agent.set_destination(destination)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break

            # Apply Kalman Filter

            cur_time = world.world.get_snapshot().timestamp.elapsed_seconds  # 현재 시간
            dt = cur_time - prev_time
            prev_time = cur_time
            
            # cur_true_vel = world.player.get_velocity()
            # cur_pos = world.player.get_transform().location
            # # cur_gnss_data = world.gnss_sensor.get_val()
            # # gnss_vel = np.array([cur_gnss_data[0]-pre_gnss_data[0],cur_gnss_data[1]-pre_gnss_data[1],0.0])
            # # gnss_vel /= dt
            # veh_vel = cal_change(pre_pos,cur_pos,dt)
            # # print(f'진짜 속도:{cur_true_vel}, 진짜 위치기반 속도:{veh_vel}, GNSS기반 속도:{gnss_vel}')
            # # pre_gnss_data = cur_gnss_data
            # pre_pos = cur_pos

            # true_accel = cal_accel(pre_true_vel,cur_true_vel,is_dt)
            # veh_accel = cal_accel(pre_veh_vel,veh_vel,is_dt)
            # gnss_accel = np.array([gnss_vel[0]-pre_gnss_vel[0],gnss_vel[1]-pre_gnss_vel[1],0.0])
            # gnss_accel /= is_dt
            # print(f'속도기반가속도:{true_accel}, 위치기반가속도:{veh_accel}, GNSS가속도:{gnss_accel}')
            # pre_imu_data = world.imu_sensor.get_val() # (yaw, accel)
            # print(f'x={pre_imu_data[1].x},y={pre_imu_data[1].y},z={pre_imu_data[1].z}\n###############################')
            # pre_true_vel = cur_true_vel
            # pre_gnss_vel = gnss_vel
            # pre_veh_vel = veh_vel
            
            if not args.mine:
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
            else:
                # 제어 입력 계산
                control = agent.my_run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
                u = np.array([control.throttle,control.brake,control.steer])

                ####
                x, y, vx, vy, ax, ay, delta = km_filter.get_state()
                km_filter.set_dt(dt)
                player_transform.set_val(x,y,pre_imu_data[0],vx,vy,delta)
                # print(f'현재 진짜 steer:{world.player.get_control().steer}, 현재 예측 steer:{delta}, 차이:{world.player.get_control().steer-delta}, 입력 steering:{u[2]}')
                # print(f'현재 진짜 steer:{world.player.get_control().steer}, 입력 steering:{u[2]}, 현재와 입력 차이:{world.player.get_control().steer-delta}')
                # 칼만 필터에 넣어줄 데이터 정리
                cur_gnss_data = world.gnss_sensor.get_val() # (x,y,z)
                gnss_vel = cal_change(pre_gnss_data,cur_gnss_data,dt)
                z = np.array([cur_gnss_data.x,cur_gnss_data.y,gnss_vel.x,gnss_vel.y]) # [x,y,vx,vy]
                imu_asInput = np.array([pre_imu_data[1].x,pre_imu_data[1].y])
                # Predict
                km_filter.predict_and_update(u,imu_asInput,z)
                # print(f'진짜 속도:{cur_true_vel},GNSS기반 속도:{gnss_vel},현재 칼만필터의 예측 속도:{km_filter.kf.x[2]},{km_filter.kf.x[3]}\n#####################################')
                # Get imu, gnss data for next step
                pre_imu_data = world.imu_sensor.get_val() # (yaw, accel)
                pre_gnss_data = cur_gnss_data
            
            # process_time = datetime.now() - dt0
            # sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
            # sys.stdout.flush()
            # dt0 = datetime.now()
            # frame += 1

    finally:

        if world is not None:
            world.world.apply_settings(original_settings)
            traffic_manager.set_synchronous_mode(True)
            ####################
            if rclpy.ok():  # ROS2 노드가 살아 있는지 확인 후 shutdown
                rclpy.shutdown()
            lidar_node.destroy_node()
            ####################
            world.destroy()
            vis.destroy_window()

       


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--upper-fov',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '--channels',
        default=16.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '--range',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '--points-per-second',
        default=250000,
        type=int,
        help='lidar\'s points per second (default: 250000)')
    argparser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    argparser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
