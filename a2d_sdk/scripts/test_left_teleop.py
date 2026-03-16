#!/usr/bin/env python3
"""
智元G01 左臂简化遥操测试脚本
- 三目相机可视化: head, hand_left, hand_right
- 键盘控制: WASD=平面移动, ZX=上下, Q/E=夹爪
- 控制模式: DELTA_POSE (相对位姿)
- 参考坐标系: arm_base_link (胸部)
"""

import time
import cv2
import pygame
import numpy as np
from a2d_sdk.robot import RobotDds, RobotController, CosineCamera
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
# ============ 配置 ============
class Config:
    # 控制参数
    DELTA_POS = 0.015      # 位移增量: 1.5cm
    DELTA_GRIPPER = 5      # 夹爪增量: 5mm
    GRIPPER_MIN = 35       # mm, 完全张开
    GRIPPER_MAX = 120      # mm, 完全闭合
    CONTROL_LIFETIME = 2.0 # 指令有效期(秒)
    FPS = 30               # 控制循环频率
    
    # 相机配置
    CAMERA_NAMES = ["head", "hand_left", "hand_right"]
    WINDOW_TITLES = {
        "head": "📷 Head Camera",
        "hand_left": "📷 Left Hand Camera", 
        "hand_right": "📷 Right Hand Camera",
    }
    
    # 控制配置
    CONTROL_GROUP = "left_arm"      # 只控制左臂
    ROBOT_LINK = "arm_base_link"    # 胸部坐标系
    CONTROL_TYPE = "DELTA_POSE"     # 相对位姿模式
    HEAD_INIT_YAW = 0.0
    HEAD_INIT_PITCH = 15 * 3.1415 / 180   # 低头 15°
    WAIST_INIT_DIS = -15
    WAIST_INIT_PITCH = 30 * 3.1415 / 180  # 弯腰 15°


# ============ Home位姿回正函数 ============
def reset_to_home(robot, home_file: str = "home_pose.json", timeout: float = 15.0):
    """
    加载保存的home_pose.json并回正到该位姿
    :param robot: RobotDds 实例
    :param home_file: home位姿文件路径
    :param timeout: 等待回正完成的最大时间(秒)
    """
    import json
    import time
    
    print(f"📂 加载 Home 位姿: {home_file}")
    try:
        with open(home_file) as f:
            home = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 未找到 {home_file}，跳过回正")
        return False
    except Exception as e:
        print(f"❌ 加载 Home 位姿失败: {e}")
        return False
    
    print("🔄 执行回正到 Home 位姿...")
    
    try:
        # 1. 先松开夹爪（安全）
        print("   🤚 松开夹爪...")
        robot.move_gripper([35.0, 35.0])  # 完全张开
        time.sleep(0.5)
        
        # # 2. 发送腰部角度 [pitch_rad, height_cm]
        # waist = home.get("waist_joint_states", [0.0, -25.0])
        # print(f"   🦴 腰部: pitch={waist[0]:.3f}rad, height={waist[1]:.1f}cm")
        # robot.move_waist(waist)
        #
        # # 3. 发送头部角度 [yaw, pitch]
        # head = home.get("head_joint_states", [0.0, 0.0])
        # print(f"   🗣️ 头部: yaw={head[0]:.3f}rad, pitch={head[1]:.3f}rad")
        # robot.move_head(head)
        
        # 4. 发送手臂关节角度 (14个)
        arm = home.get("arm_joint_states")
        if arm and len(arm) == 14:
            print(f"   🦾 手臂: 14关节角度已发送")
            robot.move_arm(arm)
        else:
            print(f"   ⚠️ 手臂关节数据无效: {arm}")

        # 5. 发送夹爪角度
        gripper = home.get("gripper_states", [35.0, 35.0])
        print(f"   🤚 夹爪: L={gripper[0]:.1f}mm, R={gripper[1]:.1f}mm")
        robot.move_gripper(gripper)
        
        # 6. 等待动作执行完成
        print(f"   ⏳ 等待回正完成 (最多 {timeout} 秒)...")
        start = time.time()
        while time.time() - start < timeout:
            # 可选：检查当前关节角度是否接近目标
            current_arm, _ = robot.arm_joint_states()
            if current_arm and None not in current_arm:
                # 计算误差
                error = sum(abs(c - t) for c, t in zip(current_arm, arm)) / 14
                if error < 0.05:  # 平均误差 < 0.05 rad ≈ 3°
                    print("   ✅ 回正完成")
                    return True
            time.sleep(0.2)
        
        print("   ⚠️ 回正超时，但指令已发送")
        return True
        
    except Exception as e:
        print(f"❌ 回正过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False
# ==========================================


# ============ 键盘控制器 ============
class KeyboardController:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.display.set_caption("Left Arm Teleop")
        print("\n🎮 控制说明:")
        print("   W/S : 左臂 Y轴 ± (左/右)")
        print("   A/D : 左臂 X轴 ± (后/前)")
        print("   Z/X : 左臂 Z轴 ± (上/下)")
        print("   Q/E : 左夹爪 闭合/张开")
        print("   ESC : 退出程序")
        print(f"   单次增量: {Config.DELTA_POS*100:.1f}cm / {Config.DELTA_GRIPPER}mm")
        
    def get_command(self, current_gripper: float) -> tuple[list, float, bool]:
        """
        读取键盘输入
        :return: (delta_pose_6d, new_gripper_mm, should_exit,should_reset)
        """
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # 初始化增量 [dx, dy, dz, drx, dry, drz]
        delta = [0.0] * 6
        new_gripper = current_gripper
        should_exit = False
        
        # === 平面移动 (WASD) ===
        if keys[pygame.K_w]:  # +Y: 向左
            delta[0] = Config.DELTA_POS
        if keys[pygame.K_s]:  # -Y: 向右
            delta[0] = -Config.DELTA_POS
        if keys[pygame.K_a]:  # -X: 向后
            delta[1] = Config.DELTA_POS
        if keys[pygame.K_d]:  # +X: 向前
            delta[1] = -Config.DELTA_POS
        
        # === 垂直移动 (ZX) ===
        if keys[pygame.K_z]:  # +Z: 向上
            delta[2] = Config.DELTA_POS
        if keys[pygame.K_x]:  # -Z: 向下
            delta[2] = -Config.DELTA_POS
        
        # === 夹爪控制 (QE) ===
        if keys[pygame.K_q]:
            new_gripper = min(Config.GRIPPER_MAX, current_gripper + Config.DELTA_GRIPPER)
        if keys[pygame.K_e]:
            new_gripper = max(Config.GRIPPER_MIN, current_gripper - Config.DELTA_GRIPPER)
        
        should_reset = False
        if keys[pygame.K_r]:  # R 键触发回正
            should_reset = True
            print("🔄 回正信号触发")
            time.sleep(0.3)  # 防连触
        # === 退出 ===
        if keys[pygame.K_ESCAPE]:
            should_exit = True
            print("👋 退出信号")
        
        return delta, new_gripper, should_exit, should_reset
    
    def cleanup(self):
        pygame.quit()

# ============ 相机显示器 ============
# 替换 CameraViewer 类
class CameraViewer:
    def __init__(self, camera: CosineCamera):
        self.camera = camera
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.canvas.manager.set_window_title("📷 Camera Viewer")
        self.images = [None] * 3
        for i, name in enumerate(Config.CAMERA_NAMES):
            self.axes[i].set_title(Config.WINDOW_TITLES[name])
            self.axes[i].axis('off')
            # 创建空白图像占位
            h, w = (720, 1280) if name == "head" else (480, 848)
            self.images[i] = self.axes[i].imshow(np.zeros((h, w, 3), dtype=np.uint8))
        plt.tight_layout()
        plt.show(block=False)  # 非阻塞显示
        print("📷 相机窗口已创建 (matplotlib)")

    def show_images(self) -> int:
        """显示三目图像，返回按键（简化版，只刷新）"""
        for i, name in enumerate(Config.CAMERA_NAMES):
            img, ts = self.camera.get_latest_image(name)
            if img is not None and img.size > 0:
                # BGR → RGB for matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if 'cv2' in dir() else img
                self.images[i].set_data(img_rgb)
                # 添加时间戳
                self.axes[i].set_xlabel(f"{ts // 1_000_000}ms", fontsize=8)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # 简化：不处理按键，用 Ctrl+C 退出
        return -1

    def cleanup(self):
        plt.close(self.fig)
# class CameraViewer:
#     def __init__(self, camera: CosineCamera):
#         self.camera = camera
#         for name, title in Config.WINDOW_TITLES.items():
#             cv2.namedWindow(title, cv2.WINDOW_NORMAL)
#             # 按原始比例缩放显示
#             if name == "head":
#                 cv2.resizeWindow(title, 640, 360)  # 1280x720 → 1/2
#             else:
#                 cv2.resizeWindow(title, 424, 240)   # 848x480 → 1/2
#         print("📷 相机窗口已创建")
#
#     def show_images(self) -> int:
#         """显示三目图像，返回按键"""
#         for name in Config.CAMERA_NAMES:
#             img, ts = self.camera.get_latest_image(name)
#
#             if img is None or img.size == 0:
#                 # 黑屏占位
#                 h, w = (720, 1280) if name == "head" else (480, 848)
#                 img = np.zeros((h, w, 3), dtype=np.uint8)
#                 cv2.putText(img, f"Waiting {name}...", (30, h//2),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
#             else:
#                 # 添加时间戳和帧率信息
#                 cv2.putText(img, f"{name} | {ts//1_000_000}ms", (10, 20),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
#
#             cv2.imshow(Config.WINDOW_TITLES[name], img)
#
#         # 处理OpenCV事件 (1ms延迟)
#         return cv2.waitKey(1) & 0xFF
#
#     def cleanup(self):
#         for name in Config.CAMERA_NAMES:
#             cv2.destroyWindow(Config.WINDOW_TITLES[name])
#         cv2.destroyAllWindows()

# ============ 主程序 ============
def main():
    print("🤖 智元G01 左臂简化遥操测试")
    print("=" * 50)
    
    # 1. 初始化机器人接口
    print("🔌 初始化 RobotDds + RobotController...")
    robot = RobotDds()           # 状态查询 + 夹爪控制
    controller = RobotController()  # 运控指令发送
    time.sleep(1)  # 等待DDS连接
    # 2. 初始化相机
    print("📷 初始化 CosineCamera...")
    camera = CosineCamera(Config.CAMERA_NAMES)
    time.sleep(0.5)  # 等待相机预热
    viewer = CameraViewer(camera)
    # ========== 新增：启动前回正 ==========
    print("\n🏠 执行回正到 Home 位姿...")
    reset_to_home(robot, home_file="home_pose.json", timeout=5.0)
    time.sleep(1)  # 额外缓冲
    # ====================================
    # 3. 初始化键盘
    keyboard = KeyboardController()
    
     # ========== 新增：设置初始头部仰角 ==========
    print("\n🗣️ 设置初始头部仰角...")

    # 头部角度: [yaw, pitch]，单位：弧度
    # yaw: 横摆角度，0=正前方，正=向左转
    # pitch: 俯仰角度，0=水平，正=抬头，负=低头
    # 有效范围: pitch ∈ [-25°, 20°] ≈ [-0.436, 0.349] rad

    # 示例：低头
    robot.move_head([Config.HEAD_INIT_YAW, Config.HEAD_INIT_PITCH])
    robot.move_waist([Config.WAIST_INIT_PITCH,Config.WAIST_INIT_DIS])
    time.sleep(1)  # 等待头部移动完成
    # ===========================================
    # 4. 初始状态
    gripper_mm = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2  # 中间位置
    print(f"🤚 初始夹爪: {gripper_mm}mm")
    
    # 5. 主控制循环
    print("🎮 开始遥操 (按 ESC 退出)...")
    dt = 1.0 / Config.FPS
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            # === 1. 读取键盘输入 ===
            delta_pose, new_gripper, should_exit, should_reset = keyboard.get_command(gripper_mm)
            if should_reset:
                print("🏠 执行回正...")
                reset_to_home(robot, timeout=10.0)
                gripper_mm = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2  # 重置夹爪
                time.sleep(1)
                continue
            if should_exit:
                break
            
            gripper_mm = new_gripper
            
            # === 2. 发送夹爪控制 (如果变化) ===
            # move_gripper 接受 [left_mm, right_mm]
            robot.move_gripper([gripper_mm, gripper_mm])  # 双侧同步，或只改左侧
            
            # === 3. 发送左臂相对位姿控制 (如果有增量) ===
            if any(abs(d) > 1e-6 for d in delta_pose):
                # 获取当前关节状态作为参考
                arm_states, _ = robot.arm_joint_states()
                head_states, _ = robot.head_joint_states()
                waist_states, _ = robot.waist_joint_states()
                
                if arm_states and None not in arm_states:
                    robot_states = {
                        "head": head_states if head_states else [0, 0],
                        "waist": waist_states if waist_states else [0, 0],
                        "arm": arm_states,
                    }
                    
                    # 构建动作: 只控制左臂
                    robot_actions = [{
                        "left_arm": {
                            "action_data": delta_pose,      # [dx,dy,dz,drx,dry,drz]
                            "control_type": Config.CONTROL_TYPE
                        }
                    }]
                    
                    # 发送轨迹跟踪指令
                    controller.trajectory_tracking_control(
                        infer_timestamp=time.time_ns(),
                        robot_states=robot_states,
                        robot_actions=robot_actions,
                        robot_link=Config.ROBOT_LINK,
                        trajectory_reference_time=0.5  # 越小执行越快
                    )
            
            # === 4. 显示相机图像 ===
            cv_key = viewer.show_images()
            if cv_key == 27:  # ESC
                break
            
            # === 5. 打印状态 (每30帧) ===
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                dx, dy, dz = delta_pose[:3]
                print(f"📊 FPS:{fps:.1f} | ΔXYZ:[{dx:.2f},{dy:.2f},{dz:.2f}]cm | Gripper:{gripper_mm}mm")
            
            # === 6. 控制循环频率 ===
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # === 安全关闭 ===
        print("\n🔌 正在关闭...")
        
        # 夹爪松开
        robot.move_gripper([Config.GRIPPER_MIN, Config.GRIPPER_MIN])
        time.sleep(0.2)
        
        # 关闭资源
        viewer.cleanup()
        camera.close()
        robot.shutdown()
        keyboard.cleanup()
        
        print("✅ 安全退出")

if __name__ == "__main__":
    # 检查环境
    import sys
    if "a2d_sdk" not in str(sys.path):
        print("⚠️ 警告: 请先执行: cd ~/a2d_sdk && source env.sh")
    
    main()
