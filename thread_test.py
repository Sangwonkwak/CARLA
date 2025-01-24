import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import time
from threading import Thread

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.create_timer(1.0, self.timer_callback)  # 1초마다 콜백 호출

    def timer_callback(self):
        self.get_logger().info('Timer callback called!')

def spin_executor(executor):
    """스레드에서 spin을 실행하는 함수"""
    executor.spin()

def main():
    rclpy.init()

    node = MyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # 스레드에서 spin 실행
    spin_thread = Thread(target=spin_executor, args=(executor,))
    spin_thread.start()

    print("This will print while spin is running!")
    time.sleep(3)  # 3초 동안 다른 작업
    print("Shutting down after some work...")

    # 노드 종료
    node.destroy_node()
    executor.shutdown()  # Executor 종료
    rclpy.shutdown()
    spin_thread.join()  # 스레드 종료

if __name__ == '__main__':
    main()
