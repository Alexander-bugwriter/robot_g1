#!/usr/bin/env python3
"""
智元G01 键盘遥控 + 三目相机可视化测试脚本
控制: ↑↓ 左夹爪, ←→ 右夹爪
显示: head, hand_left, hand_right 三目实时图像
退出: [Q] 或 [ESC]
"""

import time
import cv2
import pygame
import numpy as np
from a2d_sdk.robot import RobotDds as Robot, CosineCamera as Camera

# ============ 配置 ============
class Config:
    # 夹爪参数
    GRIPPER_MIN = 35    # mm, 完全张开
    GRIPPER_MAX = 120   # mm, 完全闭合
    GRIPPER_STEP = 2    # mm, 每次按键变化量
    
    # 相机参数
    CAMERA_NAMES = ["head", "hand_left", "hand_right"]
    WINDOW_NAMES = {
        "head": "Head Camera (D455/D457)",
        "hand_left": "Left Hand Camera (D405)",
        "hand_right": "Right Hand Camera (D405)",
    }
    
    # 控制频率
    FPS = 30

# ============ 键盘控制器 ============
class KeyboardController:
    def __init__(self):
        pygame.init()
        # 创建隐藏窗口以接收键盘事件
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.display.set_caption("Teleop Control")
        print("🎮 控制说明:")
        print("   ↑ / ↓ : 左夹爪 开/关")
        print("   ← / → : 右夹爪 开/关")
        print("   [Q] / [ESC]: 退出")
        print(f"   夹爪范围: {Config.GRIPPER_MIN}~{Config.GRIPPER_MAX} mm")
        
    def get_gripper_command(self, current_left: float, current_right: float) -> tuple[float, float, bool]:
        """
        读取键盘输入，返回新的夹爪位置
        :param current_left: 当前左夹爪位置 (mm)
        :param current_right: 当前右夹爪位置 (mm)
        :return: (new_left, new_right, should_exit)
        """
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        new_left = current_left
        new_right = current_right
        should_exit = False
        
        # ↑↓ 控制左夹爪
        if keys[pygame.K_UP]:
            new_left = min(Config.GRIPPER_MAX, current_left + Config.GRIPPER_STEP)
        if keys[pygame.K_DOWN]:
            new_left = max(Config.GRIPPER_MIN, current_left - Config.GRIPPER_STEP)
        
        # ←→ 控制右夹爪
        if keys[pygame.K_RIGHT]:
            new_right = min(Config.GRIPPER_MAX, current_right + Config.GRIPPER_STEP)
        if keys[pygame.K_LEFT]:
            new_right = max(Config.GRIPPER_MIN, current_right - Config.GRIPPER_STEP)
        
        # 退出键
        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            should_exit = True
            print("👋 退出信号")
        
        return new_left, new_right, should_exit
    
    def cleanup(self):
        pygame.quit()

# ============ 相机显示器 ============
class CameraViewer:
    def __init__(self, camera: Camera):
        self.camera = camera
        # 创建 OpenCV 窗口
        for name, title in Config.WINDOW_NAMES.items():
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            size = (640, 360) if name == "head" else (480, 270)
            cv2.resizeWindow(title, size)
        print("📷 相机窗口已创建 (按任意键关闭窗口可调整大小)")
        
    def show_images(self):
        """获取并显示三目图像"""
        for name in Config.CAMERA_NAMES:
            # 获取最新图像
            img, ts = self.camera.get_latest_image(name)
            
            if img is None or img.size == 0:
                # 显示黑屏占位
                h, w = (720, 1280) if name == "head" else (480, 848)
                img = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(img, f"Waiting for {name}...", (50, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # GDK 返回的可能是 BGR，OpenCV 需要 BGR 所以不用转换
                # 添加状态信息
                cv2.putText(img, f"{name} | TS:{ts//1_000_000}ms", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 显示图像（OpenCV 默认是 BGR）
            cv2.imshow(Config.WINDOW_NAMES[name], img)
        
        # 关键：处理 OpenCV 事件 + 键盘响应
        key = cv2.waitKey(1) & 0xFF
        return key
    
    def cleanup(self):
        """关闭所有窗口"""
        for name in Config.CAMERA_NAMES:
            cv2.destroyWindow(Config.WINDOW_NAMES[name])
        cv2.destroyAllWindows()

# ============ 主程序 ============
def main():
    print("🚀 智元G01 键盘遥控测试启动...")
    
    # 1. 初始化机器人接口
    print("🔌 连接机器人...")
    robot = Robot()
    time.sleep(0.5)  # 等待 DDS 连接
    
    # 2. 初始化相机
    print("📷 初始化相机...")
    camera = Camera(Config.CAMERA_NAMES)
    time.sleep(0.5)  # 等待相机预热
    
    # 3. 初始化控制器和显示器
    keyboard = KeyboardController()
    viewer = CameraViewer(camera)
    
    # 4. 初始夹爪位置（中间值）
    left_gripper = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2
    right_gripper = left_gripper
    print(f"🤚 初始夹爪位置: L={left_gripper}mm, R={right_gripper}mm")
    
    # 5. 主控制循环
    print("🎮 开始遥控 (按 Q/ESC 退出)...")
    dt = 1.0 / Config.FPS
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            # === 1. 读取键盘输入 ===
            new_left, new_right, should_exit = keyboard.get_gripper_command(
                left_gripper, right_gripper
            )
            
            if should_exit:
                break
            
            # 更新夹爪状态
            left_gripper, right_gripper = new_left, new_right
            
            # === 2. 发送控制命令 ===
            # move_gripper 接受 [left_mm, right_mm] 或 [0-1, 0-1]
            robot.move_gripper([left_gripper, right_gripper])
            
            # === 3. 显示相机图像 ===
            cv_key = viewer.show_images()
            if cv_key == ord('q') or cv_key == 27:  # q or ESC
                break
            
            # === 4. 打印状态（每30帧）===
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"📊 FPS:{fps:.1f} | L:{left_gripper}mm | R:{right_gripper}mm")
            
            # === 5. 控制循环频率 ===
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
        
        # 1. 夹爪归位（可选）
        robot.move_gripper([Config.GRIPPER_MIN, Config.GRIPPER_MIN])
        time.sleep(0.3)
        
        # 2. 重置机器人（可选，根据需求）
        # robot.reset()
        
        # 3. 关闭资源
        viewer.cleanup()
        camera.close()
        robot.shutdown()
        keyboard.cleanup()
        
        print("✅ 安全退出")

if __name__ == "__main__":
    # 检查是否在 GDK 环境中
    import sys
    if "a2d_sdk" not in str(sys.path):
        print("⚠️ 警告: 请先执行: cd ~/a2d_sdk && source env.sh")
    
    main()
