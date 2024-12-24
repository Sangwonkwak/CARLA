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
import datetime
import glob
import logging
import math
import os
import numpy.random as random
import re
import sys
import weakref

import kalmanfiltering as km
import utm
# import pyproj

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

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

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        ########
        self.imu_sensor = None
        self.lidar_sensor = None
        self.imu_data_handler = None
        ########
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
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
        cam_lidar_pos = 0
        my_cam_index = 0
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_lidar_pos
        self.camera_manager.set_sensor(my_cam_index, notify=True)
        ########
        # self.camera_manager.toggle_recording()

        self.lidar_sensor = CameraManager(self.player, self.hud)
        self.lidar_sensor.transform_index = cam_lidar_pos
        self.lidar_sensor.set_sensor(-1, notify=False)
        
        self.imu_sensor = IMUSensor(self.player)
        ########
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def set_player_trans(self, player_transform):
        self._player_transform = player_transform

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        # self.lidar_sensor.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.lidar_sensor.sensor,
            self.imu_sensor.sensor,
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        self.world = world
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    # def parse_events(self):
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             return True
    #         if event.type == pygame.KEYUP:
    #             if self._is_quit_shortcut(event.key):
    #                 return True
    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                # 카메라 위치 변경
                if event.key == pygame.K_z:
                    self.world.camera_manager.toggle_camera()
                    return False

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        
        gnss_location = world.gnss_sensor.get_val()
        estimated_xy = [world._player_transform.x, world._player_transform.y]
        estimated_yaw = [world._player_transform.yaw, world._player_transform.radian_yaw]
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Estimated Heading:% 5.0f, %5.0f' % (estimated_yaw[0],estimated_yaw[1]),
            'GNSS:% 20s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lon, world.gnss_sensor.lat)),
            'GNSS_xy:% 10s' % ('(% 2.3f, % 3.3f)' % (gnss_location.x, gnss_location.y)),
            'Location:%10s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'Estimated_location:% 10s' % ('(% 2.3f, % 3.3f)' % (estimated_xy[0], estimated_xy[1])),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

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
        # try:
        #     bp.set_attribute('sensor_tick', '0.05')  # 
        # except AttributeError:
        #     print("sensor_tick 속성을 지원하지 않습니다. 기본 설정을 사용합니다.")

        # bp_trans = carla.Transform(carla.Location(x=1.0, z=2.8))
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

        # cal_gravity = IMUSensor.correct_gravity(orientation)
        # self.accel = imu_data.accelerometer - cal_gravity
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
        
    # @staticmethod
    # def correct_gravity(orientation):
    #     g = 9.8 # 중력 가속도
    #     g_vec = np.array([0.0,0.0,g])
    #     yaw, pitch, roll = math.radians(orientation.yaw), math.radians(orientation.pitch), math.radians(orientation.roll)

    #     R_z = np.array([
    #             [np.cos(yaw), -np.sin(yaw), 0],
    #             [np.sin(yaw), np.cos(yaw), 0],
    #             [0, 0, 1]
    #         ])
    #     R_y = np.array([
    #             [np.cos(pitch), 0, np.sin(pitch)],
    #             [0, 1, 0],
    #             [-np.sin(pitch), 0, np.cos(pitch)]
    #         ])
    #     R_x = np.array([
    #             [1, 0, 0],
    #             [0, np.cos(roll), -np.sin(roll)],
    #             [0, np.sin(roll), np.cos(roll)]
    #         ])
    #     # Combined rotation matrix
    #     R = R_z @ R_y @ R_x

    #     g_vec = R @ g_vec
    #     return carla.Vector3D(g_vec[0],g_vec[1],g_vec[2])
        

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
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(0.0,0.0,z=bound_z*2.0)), attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            # 카메라 이미지 사이즈 조절
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4)) # lidar는 [x,y,z,intensity]로 구성
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            # image.save_to_disk('_out/%08d' % image.frame)
            image.save_to_disk('saved_camera_image/%08d.png' % image.frame)

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

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        world_time_step = 0.05
        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = world_time_step
            sim_world.apply_settings(settings) # apply_settings 호출이 비동기적으로 작동할 수 있어서 변경된 값이 바로 반영 안될 수 있음.

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        ##args setting#####
        args.loop = True
        args.mine = True
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

        clock = pygame.time.Clock()


        # print(world.player.get_physics_control())
        while True:
            clock.tick() # Rendering 주기를 결정, 인자 없을때는 cpu 최대 성능으로 알잘딱
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            world.tick(clock) # world.world.tick() 이후에 받은 데이터로 info_text작성
            world.render(display)
            pygame.display.flip()
            
            if agent.done():
                if args.loop:
                    des_index = (des_index+10) % len(spawn_points)
                    destination = spawn_points[des_index].location
                    if args.mine:
                        agent.my_set_destination(destination)
                    else:
                        agent.set_destination(destination)
                    world.hud.notification("Target reached", seconds=4.0)
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

    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


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
