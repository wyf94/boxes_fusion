from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='traffic_count',
            executable='traffic_count_node',
            name='traffic_count',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'bounding_boxes_topic': '/sort_track/targets'},
                {'detect_image_topic': "output_image_nms"},
                {'is_static_matrix': True},
                {'static_matrix_config_path': '/root/ros2_ws/src/traffic_count/config/static_matrix.json'},
                {'camera_config_path': '/root/ros2_ws/src/traffic_count/config/camera_config.yaml'},
                {'period': 20},
                {'is_CompressedImage': False},
                {'polygon_path': '/root/ros2_ws/src/traffic_count/json/polygon_final.json'},
                {'is_Web': False},
                {'url': 'http://10.31.200.139:8001/api/dataView/create'},
                {'size': [2448, 2048]},
                {'padding': [0, 50]},
                {'queue_speed': 5},
                {'show_image': False},
                {'publish_image': False},

            ]
        )
    ])