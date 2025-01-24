import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import carla
import open3d as o3d 
import struct
import ctypes
import sys
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import struct
from std_msgs.msg import Header
import ros_compatibility as roscomp
from collections import deque

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ('b', 1)
_DATATYPES[PointField.UINT8] = ('B', 1)
_DATATYPES[PointField.INT16] = ('h', 2)
_DATATYPES[PointField.UINT16] = ('H', 2)
_DATATYPES[PointField.INT32] = ('i', 4)
_DATATYPES[PointField.UINT32] = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

class CarlaLidarPublisher(Node):
    def __init__(self, channels, fov, world=None):
        super().__init__('carla_lidar_publisher')
        # qos_profile = QoSProfile(
        # reliability=QoSReliabilityPolicy.RELIABLE,
        # durability=QoSDurabilityPolicy.VOLATILE,
        # depth=100  # 큐 크기
        # )
        self.publisher = self.create_publisher(PointCloud2, '/velodyne_points', 1)
        self.num_channels = channels
        self.vertical_fov = fov
        self.world = world

        # 퍼블리싱 주기 세팅
        self.publish_rate = 0.1 # 10hz
        self.latest_lidar_data = deque(maxlen=2000)
        self.timer = self.create_timer(self.publish_rate, self.publish_lidar_data)
        print("Timer created")

    # lidar의 콜백으로 일단 데이터 저장만 해둔다.
    def process_lidar_data(self, data):
        sys.stdout.write(f'\rReceived PointCloud2 data with {(data.shape[0])}points.')
        sys.stdout.flush()
        self.latest_lidar_data.append(data)
        # print("Received data")
        
    def publish_lidar_data(self):
        try:
            # print("Publishing Lidar Data...")
            if not self.latest_lidar_data:
                print("No data to publish.")
                return
            data = self.latest_lidar_data.popleft()
            point_cloud2_msg = CarlaLidarPublisher.create_point_cloud2(data, self.world)
            self.publisher.publish(point_cloud2_msg)
            # print("Lidar data published!")
        except Exception as e:
            print(f"Error in publish_lidar_data: {e}")

        # print("????")
        # if not self.latest_lidar_data:
        #     return
            
        # data = self.latest_lidar_data[0]
        # self.latest_lidar_data.popleft()

        # point_cloud2_msg = CarlaLidarPublisher.create_point_cloud2(data, self.world)
        # self.publisher.publish(point_cloud2_msg)
        # # data = CarlaLidarPublisher.add_ring_info(data,self.num_channels,self.vertical_fov)
        # # assert data.shape[1] == 4
        # # self.get_logger().info(f'Published PointCloud2 data with {data.shape[0]} points.')

    @staticmethod
    def create_point_cloud2(points, world):
        """Convert numpy array to ROS2 PointCloud2 message."""
        snapshot = world.get_snapshot()
        carla_time = snapshot.timestamp.elapsed_seconds  # CARLA 시뮬레이션 시간
        # ros_time = rclpy.time.Time(seconds=carla_time).to_msg()

        header = get_msg_header(frame_id = "velodyne", timestamp=carla_time)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        points[:, 1] *= -1 # carla와 ros는 y축이 반대
        # msg.data = np.asarray(points, dtype=np.float32).tobytes()
        return create_cloud(header,fields,points)
        
        # msg = PointCloud2()
        # msg.header.stamp = ros_time  # 메시지에 시뮬레이션 시간 적용
        # # msg.header.stamp = rclpy.time.Time().to_msg()
        # msg.header.frame_id = 'velodyne' 
        # msg.height = 1
        # msg.width = points.shape[0] # 포인트 총 갯수 N
        # msg.is_bigendian = False
        # # msg.is_dense = True
        # msg.is_dense = True
        # msg.fields = [
        #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        # ]
        # msg.point_step = 16  # x, y, z, intensity의 총 크기 (4 bytes * 4 fields)
        # # msg.fields = [
        # #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        # #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        # #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        # #     PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
        # #     PointField(name='ring', offset=20, datatype=PointField.UINT16, count=1)
        # # ]
        # # msg.point_step = 32 # 한 포인트의 bytes 크기
        # msg.row_step = msg.point_step * points.shape[0]
        # points[:, 1] *= -1 # carla와 ros는 y축이 반대
        # # msg.data = np.asarray(points, dtype=np.float32).tobytes()
        # msg.data = points
        # return msg
    
    @staticmethod
    def add_ring_info(points, num_vertical_scans, vertical_fov):
        """
        points: (N, 4) 배열 [x, y, z, intensity]
        num_vertical_scans: 수직 레이저 채널 수 (예: 64, 32, 16 등)
        vertical_fov: 수직 스캔 범위 (최대각도 - 최소각도)
        """
        vertical_angle = np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)) * (180.0 / np.pi)
        vertical_resolution = vertical_fov / num_vertical_scans

        # ring 값 생성 (0 ~ num_vertical_scans-1)
        ring_indices = ((vertical_angle - vertical_angle.min()) / vertical_resolution).astype(np.uint16)
        points_with_ring = np.hstack((points, ring_indices.reshape(-1, 1)))  # (N, 5) 배열로 변환
        print(points_with_ring.shape)
        return points_with_ring


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset)
                  if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [{}]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


def create_cloud(header, fields, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message.
    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param fields: The point cloud fields.
    @type  fields: iterable of L{sensor_msgs.msg.PointField}
    @param points: The point cloud points.
    @type  points: list of iterables, i.e. one iterable for each point, with the
                   elements of each iterable being the values of the fields for
                   that point (in the same order as the fields parameter)
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))

    buff = ctypes.create_string_buffer(cloud_struct.size * len(points))

    point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
    offset = 0
    for p in points:
        pack_into(buff, offset, *p)
        offset += point_step

    return PointCloud2(header=header,
                       height=1,
                       width=len(points),
                       is_dense=True, ## 원래 False
                       is_bigendian=False,
                       fields=fields,
                       point_step=cloud_struct.size,
                       row_step=cloud_struct.size * len(points),
                       data=buff.raw)

def get_msg_header(frame_id=None, timestamp=None):
        """
        Get a filled ROS message header
        :return: ROS message header
        :rtype: std_msgs.msg.Header
        """
        header = Header()
        header.frame_id = frame_id
        header.stamp = roscomp.ros_timestamp(sec=timestamp, from_sec=True)
        return header

# def main(args=None):
#     rclpy.init(args=args)
#     node = CarlaLidarPublisher()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
