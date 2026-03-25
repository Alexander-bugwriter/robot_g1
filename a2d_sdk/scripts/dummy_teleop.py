#!/usr/bin/env python3
"""
Dummy Teleop 测试脚本 - 随机生成 frame data 测试 LeRobot 数据集写入
- 不需要真实机器人
- 随机生成 10 帧数据（images, states, action）
- 按 T 键触发保存并调用 record_lerobot_dataset
"""
import time
import pygame
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from record_lerobot_dataset import convert_session_to_lerobot


class Config:
    FPS = 30
    CAMERA_NAMES = ["head", "hand_left", "hand_right"]
    CAMERA_RESIZE = (448, 448)
    CAMERA_SHAPE = (3, 448, 448)  # CHW 格式

    NUM_JOINTS = 16
    ACTION_DIM = 14

    DATA_ROOT = "./dummy_teleop_data"
    TASK_NAME = "dummy_test"
    TASK_PROMPT = "Dummy data for testing LeRobot dataset writing"

    DUMMY_FRAMES = 15  # 每次生成 10 帧


class KeyboardController:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.display.set_caption("Dummy Teleop Test")
        print("\n🎮 控制说明:")
        print("   T   : 生成随机数据并保存")
        print("   ESC : 退出程序")
        print("=" * 60)

    def get_command(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        should_save = False
        should_exit = False

        if keys[pygame.K_t]:
            should_save = True
            print("💾 保存信号触发")
            time.sleep(0.3)

        if keys[pygame.K_ESCAPE]:
            should_exit = True
            print("👋 退出信号")

        return should_exit, should_save

    def cleanup(self):
        pygame.quit()


class DummyDataGenerator:
    """随机生成 dummy frame data"""

    @staticmethod
    def generate_random_image():
        """生成随机 RGB 图像 (HWC 格式，后续会转 CHW)"""
        # 生成随机彩色图像
        img = np.random.randint(0, 255,
                                (Config.CAMERA_RESIZE[1], Config.CAMERA_RESIZE[0], 3),
                                dtype=np.uint8)
        return img

    @staticmethod
    def generate_random_state():
        """生成随机机器人状态 (16 维)"""
        # 14 关节角度 + 2 夹爪状态
        joint_states = np.random.uniform(-3.14, 3.14, 14).astype(np.float32)
        gripper_states = np.random.uniform(0, 1, 2).astype(np.float32)
        return np.concatenate([joint_states, gripper_states])

    @staticmethod
    def generate_random_action():
        """生成随机动作 (14 维：左 6D+1 夹爪 + 右 6D+1 夹爪)"""
        # 左臂 6D + 左夹爪 1 + 右臂 6D + 右夹爪 1
        action = np.random.uniform(-1, 1, Config.ACTION_DIM).astype(np.float32)
        return action

    @staticmethod
    def generate_session_frames(num_frames=Config.DUMMY_FRAMES):
        """生成一个 session 的所有帧数据"""
        frames = []
        for i in range(num_frames):
            frame_data = {
                "timestamp": time.time(),
                "frame_index": i,
                "images": {
                    "head": DummyDataGenerator.generate_random_image(),
                    "hand_left": DummyDataGenerator.generate_random_image(),
                    "hand_right": DummyDataGenerator.generate_random_image(),
                },
                "states": DummyDataGenerator.generate_random_state().tolist(),
                "action": DummyDataGenerator.generate_random_action().tolist(),
            }
            frames.append(frame_data)
            print(f"📝 生成帧：{i + 1}/{num_frames}", end="\r")
        print(f"\n✅ 已生成 {num_frames} 帧 dummy 数据")
        return frames


class DataCollector:
    def __init__(self):
        self.is_collecting = False
        self.frame_buffer = []
        self.session_id = None

    def start_session(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_buffer = []
        self.is_collecting = True
        print(f"✅ 开始新采集 Session: {self.session_id}")

    def add_frames(self, frames):
        """批量添加帧"""
        self.frame_buffer.extend(frames)
        print(f"📝 已添加 {len(frames)} 帧到 buffer")

    def save_session(self):
        if len(self.frame_buffer) == 0:
            print("⚠️  没有可保存的采集数据")
            return None

        save_dir = Path(Config.DATA_ROOT) / f"session_{self.session_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "frames.npz"

        np.savez(save_path, frames=self.frame_buffer)
        print(f"\n💾 已保存 {len(self.frame_buffer)} 帧到 {save_path}")
        self.is_collecting = False
        return save_path

    def clear_session(self):
        self.frame_buffer = []
        self.is_collecting = False


def main():
    print("🤖 Dummy Teleop 测试脚本")
    print("=" * 60)
    print("📝 用途：随机生成数据测试 LeRobot 数据集写入功能")
    print("=" * 60)

    keyboard = KeyboardController()
    collector = DataCollector()

    print("\n🎮 开始测试 (按 T 生成并保存数据，ESC 退出)...")

    try:
        while True:
            should_exit, should_save = keyboard.get_command()

            if should_exit:
                break

            if should_save:
                # 1. 开始新 session
                collector.start_session()

                # 2. 生成随机数据
                frames = DummyDataGenerator.generate_session_frames()
                collector.add_frames(frames)

                # 3. 保存到 npz
                save_path = collector.save_session()

                if save_path:
                    # 4. 调用 record_lerobot_dataset 转换
                    print("\n🔄 开始转换到 LeRobot 格式...")
                    success = convert_session_to_lerobot(
                        npz_path=save_path,
                        task_name=Config.TASK_NAME,
                        task_prompt=Config.TASK_PROMPT,
                    )

                    if success:
                        print("\n✅ 测试成功！数据集已写入")
                    else:
                        print("\n❌ 转换失败")

                # 5. 清理 session
                collector.clear_session()
                time.sleep(1)

            time.sleep(1.0 / Config.FPS)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔌 正在关闭...")
        keyboard.cleanup()
        print("✅ 安全退出")


if __name__ == "__main__":
    import sys

    main()