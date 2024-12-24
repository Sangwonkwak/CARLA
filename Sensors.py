import carla
import time
import random

def setup_carla():
    client = carla.Client('localhost',2000)
    client.set_timeout(10.0)
    world = client.get_world()

    bp = world.get_blueprint_library()
    vehicle_bp = bp.filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()
    random.seed(int(time.time()))
    try:
        ego_vehicle = world.spawn_actor(random.choice(vehicle_bp),random.choice(spawn_points))
        # maybe issue
        ego_vehicle.set_attribute('role_name','ego')
        print("EGO Spawn!")
    except RuntimeError as e:
        print(f"Spawn Fail: {e}")
    
    sensors = []
    
    # LiDAR
    lidar_bp = bp.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range',50)
    lidar_transform = carla.Transform(carla.Location(z=2.5))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    sensors.append(lidar_sensor)

    # Camera
    camera_bp = bp.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5,z=2.4))
    camera_sensor = world.spawn_actor(camera_bp, camera_transform,attach_to=ego_vehicle)
    sensors.append(camera_sensor)

    # IMU
    imu_bp = bp.find('sensor.other.imu')
    imu_transform = carla.Transform(carla.Location(z=2.5))
    imu_sensor = world.spawn_actor(imu_bp,imu_transform,attach_to=ego_vehicle)
    sensors.append(imu_sensor)

    # GPS
    gps_bp = bp.find('sensor.other.gps')
    gps_transform = carla.Transform(carla.Location(z=2.5))
    gps_sensor = world.spawn_actor(gps_bp,gps_transform,attach_to=ego_vehicle)
    sensors.append(gps_sensor)

    print("Sensors attached.")
    return world, ego_vehicle, sensors

def lidar_callback(data):
    print(f"LiDAR data: {len(data)} points")

def camera_callback(image):
    print(f"Camera image size: {image.width}X{image.height}")

def imu_callback(data):
    print(f"IMU: Accelerometer {data.accelerometer}, Gyroscope {data.gyroscope}")

def gps_callback(data):
    print(f"GPS: {data.latitude}, {data.longtitude}, {data.altitude}")

import weakref

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


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

