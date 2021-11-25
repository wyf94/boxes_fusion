import rclpy
from rclpy.node import Node

from traffic_count.traffic_count_class import TrafficCountPublisher


def main(args=None):
    rclpy.init(args=args)

    traffic_count_publisher = TrafficCountPublisher()

    rclpy.spin(traffic_count_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()