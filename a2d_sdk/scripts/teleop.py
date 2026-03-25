# !/usr/bin/env python3
"""
智元 G01 双臂键盘遥操脚本（纯遥操，不含录制）
- 三目相机可视化：head, hand_left, hand_right
- 键盘控制：双臂 6D 相对位姿 + 夹爪（只用 dx,dy,dz + gripper，rot 保持 0）
- 控制模式：DELTA_POSE（相对位姿）
- 参考坐标系：arm_base_link（胸部）
- 数据保存：有键盘输入时保存帧到临时 npy 文件
"""

import time
import pygame
import json
from pathlib import Path
from datetime import datetime
from a2d_sdk.robot import RobotDds, RobotController, CosineCamera
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from record_lerobot_dataset import convert_session_to_lerobot

class Config:
    DELTA_POS = 0.015
    DELTA_GRIPPER = 0.1
    GRIPPER_MIN = 0
    GRIPPER_MAX = 1
    FPS = 30

    CAMERA_NAMES = ["head", "hand_left", "hand_right"]
    CAMERA_RESIZE = (448, 448)
    WINDOW_TITLES = {
        "head": "📷 Head",
        "hand_left": "📷 Left Wrist",
        "hand_right": "📷 Right Wrist",
    }

    CONTROL_TYPE = "DELTA_POSE"
    ROBOT_LINK = "arm_base_link"
    TRAJECTORY_TIME = 0.5

    DATA_ROOT = "./teleop_data"
    TASK_NAME = "pick_place"
    TASK_PROMPT = "Pick up something and place it in the bowl"

    NUM_JOINTS = 16
    ACTION_DIM = 14

def reset_to_home(robot, controller, home_file="home_pose.json", timeout=15.0):
    """
    使用轨迹跟踪控制回正到 Home 位姿
    注意：需要传入 RobotController 实例
    """
    print(f"📂 加载 Home 位姿：{home_file}")
    try:
        with open(home_file) as f:
            home = json.load(f)
    except Exception as e:
        print(f"❌ 加载 Home 位姿失败：{e}")
        return False

    print("🔄 执行轨迹跟踪回正...")
    try:
        # 1. 松开夹爪
        print("   🤚 松开夹爪...")
        robot.move_gripper([35.0, 35.0])
        time.sleep(0.5)

        # 2. 获取当前状态作为参考
        arm_states, _ = robot.arm_joint_states()
        head_states, _ = robot.head_joint_states()
        waist_states, _ = robot.waist_joint_states()

        if not arm_states or None in arm_states:
            print("   ⚠️  无法获取当前关节状态")
            return False

        # 3. 加载目标角度
        target_arm = home.get("arm_joint_states")
        if not target_arm or len(target_arm) != 14:
            print(f"   ⚠️  目标关节数据无效：{target_arm}")
            return False

        print(f"   🦾 目标角度已加载（14 关节）")

        # 4. 构建轨迹跟踪指令
        robot_states = {
            "head": head_states if head_states else [0.0, 0.0],
            "waist": waist_states if waist_states else [0.0, 0.0],
            "arm": arm_states,
        }

        # 关键：使用 ABS_JOINT 控制类型
        robot_actions = [{
            "left_arm": {
                "action_data": target_arm[:7],  # 左臂 7 关节
                "control_type": "ABS_JOINT"
            },
            "right_arm": {
                "action_data": target_arm[7:],  # 右臂 7 关节
                "control_type": "ABS_JOINT"
            }
        }]

        # 5. 发送轨迹跟踪指令
        print("   📤 发送 trajectory_tracking_control 指令...")
        controller.trajectory_tracking_control(
            infer_timestamp=time.time_ns(),
            robot_states=robot_states,
            robot_actions=robot_actions,
            robot_link="arm_base_link",
            trajectory_reference_time=2.0  # 2 秒内完成回正
        )

        # 6. 等待执行完成
        print(f"   ⏳ 等待回正完成 (最多 {timeout} 秒)...")
        start = time.time()
        while time.time() - start < timeout:
            current_arm, _ = robot.arm_joint_states()
            if current_arm and None not in current_arm:
                error = sum(abs(c - t) for c, t in zip(current_arm, target_arm)) / 14
                if error < 0.05:  # 平均误差 < 3°
                    print("   ✅ 回正完成")
                    return True
            time.sleep(0.2)

        print("   ⚠️  回正超时，但指令已发送")
        return True

    except Exception as e:
        print(f"❌ 回正过程出错：{e}")
        import traceback
        traceback.print_exc()
        return False

class KeyboardController:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.display.set_caption("Bimanual Teleop")
        print("\n🎮 控制说明:")
        print("   左臂控制:")
        print("     W/S : 左臂 Y 轴 ± (前/后)")
        print("     A/D : 左臂 X 轴 ± (左/右)")
        print("     Z/X : 左臂 Z 轴 ± (上/下)")
        print("     Q/E : 左夹爪 闭合/张开")
        print("   右臂控制:")
        print("     I/K : 右臂 Y 轴 ± (前/后)")
        print("     J/L : 右臂 X 轴 ± (左/右)")
        print("     N/M : 右臂 Z 轴 ± (上/下)")
        print("     U/O : 右夹爪 闭合/张开")
        print("   录制控制:")
        print("     R   : 回正 + 开始新采集")
        print("     T   : 保存当前采集到临时文件")
        print("     Y   : 废弃当前采集 + 回正")
        print("     ESC : 退出程序")
        print(f"   单次增量：{Config.DELTA_POS * 100:.1f}cm / {Config.DELTA_GRIPPER}mm")

    def get_command(self, left_gripper, right_gripper):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        left_action = [0.0] * 7
        right_action = [0.0] * 7
        new_left_gripper = left_gripper
        new_right_gripper = right_gripper

        should_exit = False
        should_reset = False
        should_save = False
        should_discard = False
        has_action = False

        if keys[pygame.K_w]:
            left_action[0] = Config.DELTA_POS
            has_action = True
        if keys[pygame.K_s]:
            left_action[0] = -Config.DELTA_POS
            has_action = True
        if keys[pygame.K_a]:
            left_action[1] = Config.DELTA_POS
            has_action = True
        if keys[pygame.K_d]:
            left_action[1] = -Config.DELTA_POS
            has_action = True
        if keys[pygame.K_z]:
            left_action[2] = Config.DELTA_POS
            has_action = True
        if keys[pygame.K_x]:
            left_action[2] = -Config.DELTA_POS
            has_action = True
        if keys[pygame.K_q]:
            new_left_gripper = min(Config.GRIPPER_MAX, left_gripper + Config.DELTA_GRIPPER)
            left_action[6] = new_left_gripper
            has_action = True
        if keys[pygame.K_e]:
            new_left_gripper = max(Config.GRIPPER_MIN, left_gripper - Config.DELTA_GRIPPER)
            left_action[6] = new_left_gripper
            has_action = True


        if keys[pygame.K_i]:
            right_action[0] = Config.DELTA_POS
            has_action = True
        if keys[pygame.K_k]:
            right_action[0] = -Config.DELTA_POS
            has_action = True
        if keys[pygame.K_j]:
            right_action[1] = Config.DELTA_POS
            has_action = True
        if keys[pygame.K_l]:
            right_action[1] = -Config.DELTA_POS
            has_action = True
        if keys[pygame.K_n]:
            right_action[2] = Config.DELTA_POS
            has_action = True
        if keys[pygame.K_m]:
            right_action[2] = -Config.DELTA_POS
            has_action = True
        if keys[pygame.K_u]:
            new_right_gripper = min(Config.GRIPPER_MAX, right_gripper + Config.DELTA_GRIPPER)
            right_action[6] = new_right_gripper
            has_action = True
        if keys[pygame.K_o]:
            new_right_gripper = max(Config.GRIPPER_MIN, right_gripper - Config.DELTA_GRIPPER)
            right_action[6] = new_right_gripper
            has_action = True

        if keys[pygame.K_r]:
            should_reset = True
            print("🔄 回正 + 开始新采集信号触发")
            time.sleep(0.3)
        if keys[pygame.K_t]:
            should_save = True
            print("💾 保存采集信号触发")
            time.sleep(0.3)
        if keys[pygame.K_y]:
            should_discard = True
            print("🗑️  废弃采集信号触发")
            time.sleep(0.3)
        if keys[pygame.K_ESCAPE]:
            should_exit = True
            print("👋 退出信号")

        return (left_action, right_action, new_left_gripper, new_right_gripper,
                should_exit, should_reset, should_save, should_discard,has_action)

    def cleanup(self):
        pygame.quit()


class CameraViewer:
    def __init__(self, camera):
        self.camera = camera
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.fig.canvas.manager.set_window_title("📷 Camera Viewer")
        self.images = [None] * 3
        for i, name in enumerate(Config.CAMERA_NAMES):
            self.axes[i].set_title(Config.WINDOW_TITLES[name])
            self.axes[i].axis('off')
            self.images[i] = self.axes[i].imshow(np.zeros((448, 448, 3), dtype=np.uint8))
        plt.tight_layout()
        plt.show(block=False)
        print("📷 相机窗口已创建 (matplotlib)")

    def show_images(self):
        for i, name in enumerate(Config.CAMERA_NAMES):
            img, ts = self.camera.get_latest_image(name)
            if img is not None and img.size > 0:
                img_resized = cv2.resize(img, Config.CAMERA_RESIZE)
                img_rgb = img_resized
                self.images[i].set_data(img_rgb)
                self.axes[i].set_xlabel(f"{name} | {ts // 1_000_000}ms", fontsize=8)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return -1

    def cleanup(self):
        plt.close(self.fig)


class DataCollector:
    def __init__(self):
        self.is_collecting = False
        self.frame_buffer = []
        self.session_id = None
        self.start_time = None

    def start_session(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_buffer = []
        self.is_collecting = True
        self.start_time = time.time()
        print(f"✅ 开始新采集 Session: {self.session_id}")

    def add_frame(self, images, states,  action):
        if not self.is_collecting:
            return
        frame_data = {
            "timestamp": time.time(),
            "frame_index": len(self.frame_buffer),
            "images": {k: v.copy() for k, v in images.items()},
            "states": list(states) ,# ✅ 用 list() 转换
            "action": list(action),  # ✅ 用 list() 转换
        }
        self.frame_buffer.append(frame_data)
        print(f"📝 采集帧：{len(self.frame_buffer)}", end="\r")

    def save_session(self):
        if not self.is_collecting or len(self.frame_buffer) == 0:
            print("⚠️  没有可保存的采集数据")
            return None
        save_dir = Path(Config.DATA_ROOT) / f"session_{self.session_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        # 修改为 .npz 格式
        save_path = save_dir / "frames.npz"
        np.savez(save_path, frames=self.frame_buffer)
        print(f"\n💾 已保存 {len(self.frame_buffer)} 帧到 {save_path}")
        self.is_collecting = False
        self.frame_buffer = []
        return save_path

    def discard_session(self):
        if not self.is_collecting:
            return
        count = len(self.frame_buffer)
        self.frame_buffer = []
        self.is_collecting = False
        print(f"🗑️  已废弃 {count} 帧采集数据")


def main():
    print("🤖 智元 G01 双臂遥操测试")
    print("=" * 60)

    print("🔌 初始化 RobotDds + RobotController...")
    robot = RobotDds()
    controller = RobotController()
    time.sleep(1)

    print("📷 初始化 CosineCamera...")
    camera = CosineCamera(Config.CAMERA_NAMES)
    time.sleep(0.5)
    viewer = CameraViewer(camera)

    print("\n🏠 执行回正到 Home 位姿...")
    reset_to_home(robot,controller, home_file="home_pose.json", timeout=5.0)
    time.sleep(1)

    keyboard = KeyboardController()

    left_gripper = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2
    right_gripper = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2
    print(f"🤚 初始夹爪：L={left_gripper}mm, R={right_gripper}mm")

    collector = DataCollector()

    frame_count = 0
    start_time = time.time()

    print("\n🎮 开始遥操 (按 R 开始采集，ESC 退出)...")

    try:
        while True:
            left_action, right_action, new_left_gripper, new_right_gripper, \
                should_exit, should_reset, should_save, should_discard,has_action = \
                keyboard.get_command(left_gripper, right_gripper)

            if should_reset:
                reset_to_home(robot,controller, home_file="home_pose.json", timeout=5.0)
                left_gripper = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2
                right_gripper = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2
                collector.start_session()
                time.sleep(1)
                continue

            if should_save:
                save_path = collector.save_session()  # 保存为 npz
                if save_path:
                    convert_session_to_lerobot(
                        npz_path=save_path,
                        task_name=Config.TASK_NAME,  # 硬编码在 config 里
                        task_prompt=Config.TASK_PROMPT,  # 硬编码在 config 里
                    )
                time.sleep(1)
                continue

            if should_discard:
                collector.discard_session()
                reset_to_home(robot,controller, home_file="home_pose.json", timeout=5.0)
                left_gripper = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2
                right_gripper = (Config.GRIPPER_MIN + Config.GRIPPER_MAX) // 2
                time.sleep(1)
                continue

            if should_exit:
                break

            left_gripper = new_left_gripper
            right_gripper = new_right_gripper

            if has_action:
                images = {}
                for name in Config.CAMERA_NAMES:
                    img, ts = camera.get_latest_image(name)
                    if img is not None and img.size > 0:
                        img = cv2.resize(img, Config.CAMERA_RESIZE)
                        images[name] = img

                arm_states, arm_ts = robot.arm_joint_states()
                gripper_states, gripper_ts = robot.gripper_states()

                action_14d = left_action + right_action
                print("left_action:",left_action)
                print("right_action:",right_action)
                print("action_14d:",action_14d)
                print("gripper_states:",gripper_states)
                print("arm_states:",arm_states)
                print("left_gripper:",left_gripper)
                print("right_gripper:",right_gripper)
                total_states=list(arm_states) + gripper_states
                print("total_states:",total_states)

                if collector.is_collecting and images:
                    collector.add_frame(
                        images=images,
                        states=total_states ,
                        action=action_14d,
                    )

                if arm_states and None not in arm_states:
                    robot_states = {
                        "head": [0.0, 0.0],
                        "waist": [0.0, 0.0],
                        "arm": arm_states,
                    }

                    robot_actions = [{
                        "left_arm": {
                            "action_data": left_action[:6],
                            "control_type": Config.CONTROL_TYPE
                        },
                        "right_arm": {
                            "action_data": right_action[:6],
                            "control_type": Config.CONTROL_TYPE
                        }
                    }]

                    controller.trajectory_tracking_control(
                        infer_timestamp=time.time_ns(),
                        robot_states=robot_states,
                        robot_actions=robot_actions,
                        robot_link=Config.ROBOT_LINK,
                        trajectory_reference_time=Config.TRAJECTORY_TIME
                    )

                    robot.move_gripper([left_gripper, right_gripper])

            display_images = {}
            for name in Config.CAMERA_NAMES:
                img, _ = camera.get_latest_image(name)
                if img is not None and img.size > 0:
                    img = cv2.resize(img, Config.CAMERA_RESIZE)
                    display_images[name] = img
            viewer.show_images()

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                status = "🔴 采集中" if collector.is_collecting else "⚪ 空闲"
                print(f"\n📊 FPS:{fps:.1f} | 状态:{status} | 帧数:{len(collector.frame_buffer)} | "
                      f"L:{left_gripper}mm R:{right_gripper}mm")

            time.sleep(1.0 / Config.FPS)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔌 正在关闭...")
        if collector.is_collecting:
            collector.save_session()
        robot.move_gripper([Config.GRIPPER_MIN, Config.GRIPPER_MIN])
        time.sleep(0.2)
        viewer.cleanup()
        camera.close()
        robot.shutdown()
        keyboard.cleanup()
        print("✅ 安全退出")


if __name__ == "__main__":
    import sys

    if "a2d_sdk" not in str(sys.path):
        print("⚠️  警告：请先执行：cd ~/a2d_sdk && source env.sh")
    main()