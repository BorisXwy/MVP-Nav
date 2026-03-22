import time
import sys
import os
import numpy as np
import rclpy
from rclpy.node import Node
from genie_msgs.msg import Position
from a2d_sdk.robot import RobotDds as Robot
from a2d_sdk.robot import CosineCamera
import cv2
from PIL import Image
import threading
import math


class ROS2OdometrySubscriber(Node):
    """ROS2里程计订阅器节点"""
    def __init__(self, node_name='odometry_subscriber'):
        super().__init__(node_name)
        
        # 创建订阅者
        self.subscription = self.create_subscription(
            Position,
            '/hal/position',
            self.odometry_callback,
            10
        )
        
        # 存储最新的里程计信息
        self.latest_odometry = {
            'odom_x': 0.0,
            'odom_y': 0.0,
            'odom_z': 0.0,
            'odom_angle': 0.0,  # 里程计角度（弧度）
            'linear_speed': 0.0,  # 线速度
            'angular_speed': 0.0,  # 角速度
            'timestamp': 0.0,
            'is_valid': False
        }
        
        self.get_logger().info(f'ROS2里程计订阅器已启动，监听话题: /hal/position')
    
    def odometry_callback(self, msg):
        """里程计回调函数"""
        try:
            # 提取里程计信息
            self.latest_odometry['odom_x'] = msg.odom_x
            self.latest_odometry['odom_y'] = msg.odom_y
            self.latest_odometry['odom_z'] = msg.odom_z
            self.latest_odometry['odom_angle'] = msg.odom_angle
            self.latest_odometry['linear_speed'] = msg.linear_speed
            self.latest_odometry['angular_speed'] = msg.angular_speed
            self.latest_odometry['is_valid'] = True
            
            # 时间戳
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                self.latest_odometry['timestamp'] = stamp.sec + stamp.nanosec * 1e-9
            else:
                self.latest_odometry['timestamp'] = time.time()
                
        except Exception as e:
            self.get_logger().error(f'处理里程计消息时出错: {e}')
    
    def get_odometry(self):
        """获取最新里程计数据"""
        return self.latest_odometry.copy()
    
    def is_odometry_valid(self):
        """检查里程计数据是否有效"""
        return self.latest_odometry['is_valid']


class RobotWheelController:
    def __init__(self, robot):
        self.robot = robot
        time.sleep(1)
        self.step_size = 0.2
        self.angle = np.pi/12
        # 添加倒退步长
        self.backward_step_size = 0.2

    def wheel_move(self, action):
        if not isinstance(action, (int, float)):
            raise ValueError("Action must be an integer or float")
        action = int(action)

        # 根据不同的动作设置不同的循环次数
        if action == 1:  # 前进
            rounds = 50
        elif action == 4:  # 后退
            rounds = 50
        else:  # 转向
            rounds = 96
            
        for i in range(rounds):
            if action == 1:  # 前进
                self.robot.move_wheel(self.step_size, 0.0)
                time.sleep(0.01)
            elif action == 2:  # 左转
                self.robot.move_wheel(0.0, self.angle)
                time.sleep(0.01)
            elif action == 3:  # 右转
                self.robot.move_wheel(0.0, -self.angle)
                time.sleep(0.01)
            elif action == 4:  # 后退
                # 注意：倒退是向相反方向移动
                self.robot.move_wheel(-self.backward_step_size, 0.0)
                time.sleep(0.01)


class ObserLoader:
    def __init__(self, camera, camera_group=None):
        self.camera = camera
        if camera_group is None:
            camera_group = "head"
        self.camera_group = camera_group
        time.sleep(1)

    def get_head_img(self) -> np.ndarray:
        camera_name = "head"
        image, image_ts = self.camera.get_latest_image(camera_name)
        return image


class Env:
    def __init__(self, camera, camera_group, robot, use_ros_odometry=True):
        self.camera = camera
        self.camera_group = camera_group
        self.robot = robot
        self.controller = RobotWheelController(self.robot)
        self.obser_loader = ObserLoader(self.camera, self.camera_group)
        
        self.step_size = self.controller.step_size
        self.angle = self.controller.angle
        
        # 初始化位姿 - 使用里程计
        self.pose = np.array([0.0, 0.0, 0.0]).astype('float')
        
        # ROS2里程计集成
        self.use_ros_odometry = use_ros_odometry
        self.ros_node = None
        self.ros_thread = None
        self.odometry_subscriber = None
        
        if use_ros_odometry:
            self.init_ros2()
    
    def init_ros2(self):
        """初始化ROS2"""
        try:
            # 检查ROS2是否已初始化
            if not rclpy.ok():
                rclpy.init()
            
            # 创建ROS节点
            self.ros_node = rclpy.create_node('env_odometry_node')
            
            # 创建订阅器
            self.odometry_subscriber = ROS2OdometrySubscriber()
            
            # 启动ROS2处理线程
            def ros_spin():
                while rclpy.ok():
                    rclpy.spin_once(self.odometry_subscriber, timeout_sec=0.1)
            
            self.ros_thread = threading.Thread(target=ros_spin, daemon=True)
            self.ros_thread.start()
            
            print("ROS2里程计订阅已启动")
            time.sleep(2)  # 等待订阅建立
            
        except Exception as e:
            print(f"初始化ROS2时出错: {e}")
            print("将使用内部里程计估计位置")
            self.use_ros_odometry = False
    
    def get_location(self):
        """
        获取当前位置
        优先使用ROS2提供的里程计数据，如果没有则使用内部估计
        """
        if self.use_ros_odometry and self.odometry_subscriber is not None:
            # 从ROS2获取里程计数据
            odometry = self.odometry_subscriber.get_odometry()
            
            if odometry['is_valid']:
                # 使用里程计数据
                x = odometry['odom_x']
                y = odometry['odom_y']
                angle = odometry['odom_angle']
                
                # 更新内部位姿
                self.pose[0] = x
                self.pose[1] = y
                self.pose[2] = angle
                normalized = angle % (2 * np.pi)
    
                # 处理负角度
                if normalized < 0:
                    normalized = 2 * np.pi - normalized

                
                # 返回里程计信息
                return (x,y,angle)
        
        # 如果没有ROS2里程计或数据无效，返回内部估计的位置
        return self.pose
    
    
    def step(self, action):
        action = action['action']
        prev_pose = self.pose.copy()
        self.controller.wheel_move(action)
        
        # 只在没有ROS2里程计时使用内部估计
        if not self.use_ros_odometry or not (self.odometry_subscriber and 
                                           self.odometry_subscriber.is_odometry_valid()):
            if action == 1:  # 前进
                dx = self.step_size * np.cos(prev_pose[2])
                dy = self.step_size * np.sin(prev_pose[2])
                self.pose[0] += dx
                self.pose[1] += dy
            elif action == 2:  # 左转
                self.pose[2] += self.angle
            elif action == 3:  # 右转
                self.pose[2] -= self.angle
            elif action == 4:  # 后退
                # 倒退是向相反方向移动
                dx = -self.controller.backward_step_size * np.cos(prev_pose[2])
                dy = -self.controller.backward_step_size * np.sin(prev_pose[2])
                self.pose[0] += dx
                self.pose[1] += dy
            
            # 规范化角度到 [-pi, pi]
            self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        
        # 获取当前位置（如果使用ROS2里程计，这会从里程计数据更新）
        current_loc = self.get_location()
        
        image = self.obser_loader.get_head_img()
        while image is None:
            image = self.obser_loader.get_head_img()
        
        obs = {'rgb': image}
        return obs, None, None
    
    def set_goal(self):
        image = cv2.imread("./goal_image/goal.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.instance_imagegoal = image
    
    def reset(self):
        """重置环境"""
        # 重置到原点，但保持ROS2里程计的连续性
        print(f"环境重置 - 位置归零")
        self.pose = np.array([0.0, 0.0, 0.0])
        
        image = self.obser_loader.get_head_img()
        while image is None:
            image = self.obser_loader.get_head_img()
        
        obs = {'rgb': image}
        return obs, None
    
    def cleanup(self):
        """清理资源"""
        if self.odometry_subscriber:
            self.odometry_subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def interactive_control_loop():
    """交互式控制循环"""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                  AGV机器人交互式控制台                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # 初始化硬件
    print("\n[1/3] 初始化硬件...")
    try:
        camera_group = ["head"]
        camera = CosineCamera(camera_group)
        robot = Robot()
        print("✓ 相机和机器人初始化成功")
    except Exception as e:
        print(f"✗ 硬件初始化失败: {e}")
        print("使用模拟模式进行测试...")
        # 使用模拟硬件
        class MockRobot:
            def move_wheel(self, linear, angular):
                print(f"[模拟] 轮子控制 - 线速度: {linear:.3f}, 角速度: {angular:.3f}")
                time.sleep(0.1)
        
        class MockCamera:
            def __init__(self, camera_group):
                self.counter = 0
            
            def get_latest_image(self, camera_name):
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.putText(image, f"Frame {self.counter}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.counter += 1
                return image, time.time()
        
        camera = MockCamera(camera_group)
        robot = MockRobot()
    
    # 创建环境实例
    print("[2/3] 创建环境实例...")
    env = Env(camera, camera_group, robot, use_ros_odometry=True)
    
    # 显示控制说明
    print("[3/3] 交互式控制就绪!")
    print("\n" + "="*60)
    print("控制指令说明:")
    print("  w / 1  - 前进")
    print("  a / 2  - 左转")
    print("  d / 3  - 右转")
    print("  s / 4  - 后退")
    print("  r      - 重置位置")
    print("  p      - 显示当前位置")
    print("  i      - 显示图像信息")
    print("  q      - 退出程序")
    print("="*60)
    
    # 动作映射字典
    action_map = {
        'w': 1, '1': 1,    # 前进
        'a': 2, '2': 2,    # 左转
        'd': 3, '3': 3,    # 右转
        's': 4, '4': 4,    # 后退
    }
    
    action_names = {
        1: "前进",
        2: "左转", 
        3: "右转",
        4: "后退"
    }
    
    try:
        while True:
            # 获取用户输入
            user_input = input("\n请输入指令: ").strip().lower()
            
            if user_input == '':
                continue
                
            if user_input in ['q', 'quit', 'exit']:
                print("退出交互式控制...")
                break
                
            elif user_input in ['r', 'reset']:
                # 重置环境
                print("执行环境重置...")
                obs, info = env.reset()
                current_loc = env.get_location()
                print(f"重置完成! 当前位置: x={current_loc[0]:.3f}, y={current_loc[1]:.3f}, 角度={current_loc[2]:.1f}")
                
            elif user_input in ['p', 'pos', 'position']:
                # 显示当前位置
                current_loc = env.get_location()
                print(f"当前位置: x={current_loc[0]:.3f}, y={current_loc[1]:.3f}, 角度={current_loc[2]:.1f}")
                
            elif user_input in ['i', 'img', 'image']:
                # 显示图像信息
                image = env.obser_loader.get_head_img()
                if image is not None:
                    print(f"图像尺寸: {image.shape}")
                    # 可选：保存当前图像
                    # cv2.imwrite(f"capture_{int(time.time())}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    # print("图像已保存")
                else:
                    print("无法获取图像")
                    
            elif user_input in action_map:
                # 执行动作
                action_value = action_map[user_input]
                action_dict = {'action': action_value}
                
                print(f"执行动作: {action_names[action_value]}")
                
                # 执行动作前的位置
                prev_loc = env.get_location()
                
                # 执行动作
                obs, reward, done = env.step(action_dict)
                
                # 执行后的位置
                current_loc = env.get_location()
                
                # 计算位移
                dx = current_loc[0] - prev_loc[0]
                dy = current_loc[1] - prev_loc[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                print(f"动作完成!")
                print(f"  位置变化: Δx={dx:.3f}, Δy={dy:.3f}, 距离={distance:.3f}")
                print(f"  当前位置: x={current_loc[0]:.3f}, y={current_loc[1]:.3f}, 角度={current_loc[2]:.1f}°")
                
                # 显示图像信息
                if obs['rgb'] is not None:
                    print(f"  图像尺寸: {obs['rgb'].shape}")
                    
            else:
                print("未知指令，请输入以下指令之一:")
                print("  w/1, a/2, d/3, s/4, r, p, i, q")
                
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在退出...")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在清理资源...")
        env.cleanup()
        print("程序结束")


if __name__ == '__main__':
    interactive_control_loop()
