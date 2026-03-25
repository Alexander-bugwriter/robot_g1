#!/usr/bin/env python3
"""
将遥操采集的临时 npz 文件转换为 LeRobot v2.1 格式数据集
- 读取临时 npz 文件
- 构建 LeRobot features
- 视频编码 (mp4)
- 追加到现有数据集或创建新数据集
- 合并多个数据集
"""

import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from datasets import concatenate_datasets
import shutil


class Config:
    # 数据集配置
    FPS = 30
    DATASET_ROOT = "./test_datasets"

    # 相机配置
    CAMERA_NAMES = ["image", "left_wrist_image", "right_wrist_image"]
    CAMERA_SHAPE = [3, 448, 448]

    # 机器人配置
    NUM_JOINTS = 16
    ACTION_DIM = 14

    # 视频编码
    USE_VIDEOS = True
    VIDEO_ENCODING_BATCH_SIZE = 1
    IMAGE_WRITER_PROCESSES = 0
    IMAGE_WRITER_THREADS = 4 * 3


def build_features() -> dict:
    """构建 LeRobot v2.1 格式的 features 定义"""
    features = {}

    # 相机特征 (视频编码)
    for cam in Config.CAMERA_NAMES:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": Config.CAMERA_SHAPE,
            "names": ["channels", "height", "width"],
        }

    # 机器人状态 (14 关节)
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [Config.NUM_JOINTS],
        "names": [
            *[f"joint_left_{i}" for i in range(7)],
            *[f"joint_right_{i}" for i in range(7)],
        ],
    }

    # 动作 (14 维：左 6D+1 夹爪 + 右 6D+1 夹爪)
    features["action"] = {
        "dtype": "float32",
        "shape": [Config.ACTION_DIM],
        "names": [
            "l_dx", "l_dy", "l_dz", "l_drx", "l_dry", "l_drz", "l_gripper",
            "r_dx", "r_dy", "r_dz", "r_drx", "r_dry", "r_drz", "r_gripper",
        ],
    }

    return features


class LeRobotRecorder:
    def __init__(self, task_name: str, task_prompt: str):
        self.task_name = task_name
        self.task_prompt = task_prompt
        self.repo_id = f"g01/{task_name.replace(' ', '_')}"
        # self.dataset_path = Path(Config.DATASET_ROOT) / self.repo_id
        self.dataset_path = Path(Config.DATASET_ROOT)
        self.dataset = None
        self.video_manager = None
        self.is_new_dataset = False

    def load_or_create_dataset(self):
        """加载现有数据集或创建新数据集"""
        features = build_features()

        # ✅ 修复：正确检查数据集是否存在
        meta_info_path = self.dataset_path / "meta" / "info.json"
        print(f"🔍 检查数据集路径：{self.dataset_path}")
        print(f"🔍 检查 meta/info.json: {meta_info_path}")
        print(f"🔍 路径存在：{self.dataset_path.exists()}")
        print(f"🔍 info.json 存在：{meta_info_path.exists()}")

        if self.dataset_path.exists() and meta_info_path.exists():
            # 加载现有数据集
            print(f"📂 加载现有数据集：{self.dataset_path}")
            try:
                self.dataset = LeRobotDataset(
                    self.repo_id,
                    root=Config.DATASET_ROOT,
                    batch_encoding_size=Config.VIDEO_ENCODING_BATCH_SIZE,
                )
                self.is_new_dataset = False
                print(f"   现有 episodes: {self.dataset.num_episodes}")
                print(f"   现有 frames: {self.dataset.num_frames}")
            except Exception as e:
                print(f"⚠️  加载失败：{e}，将创建新数据集")
                self.is_new_dataset = True
        else:
            # 创建新数据集
            print(f"🆕 创建新数据集：{self.dataset_path}")
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=Config.FPS,
                features=features,
                root=Config.DATASET_ROOT,
                robot_type="custom_g01_bimanual",
                use_videos=Config.USE_VIDEOS,
                image_writer_processes=Config.IMAGE_WRITER_PROCESSES,
                image_writer_threads=Config.IMAGE_WRITER_THREADS,
                batch_encoding_size=Config.VIDEO_ENCODING_BATCH_SIZE,
            )
            self.dataset.start_image_writer(
                num_processes=Config.IMAGE_WRITER_PROCESSES,
                num_threads=Config.IMAGE_WRITER_THREADS,
            )
            self.is_new_dataset = True
            print(f"   Features 已初始化")

        # 启动视频编码管理器
        self.video_manager = VideoEncodingManager(self.dataset)
        self.video_manager.__enter__()

        return self.is_new_dataset

    def convert_npz_to_lerobot(self, npz_path: Path):
        """将 npz 文件转换为 LeRobot 格式并追加到数据集"""
        print(f"📂 读取临时文件：{npz_path}")

        if not npz_path.exists():
            print(f"❌ 文件不存在：{npz_path}")
            return False

        # 加载 npz 数据
        data = np.load(npz_path, allow_pickle=True)
        frames = data['frames'] if 'frames' in data else data.arr_0

        if len(frames) == 0:
            print("⚠️  没有可转换的帧")
            return False

        print(f"🎬 转换 {len(frames)} 帧到 LeRobot 格式...")

        # 逐帧添加到数据集
        for i, frame_data in enumerate(frames):
            frame = {}

            # 相机图像 (转换名称：head→image, hand_left→left_wrist_image, hand_right→right_wrist_image)
            images = frame_data.get('images', {})
            if "head" in images:
                frame["observation.images.image"] = images["head"]
            if "hand_left" in images:
                frame["observation.images.left_wrist_image"] = images["hand_left"]
            if "hand_right" in images:
                frame["observation.images.right_wrist_image"] = images["hand_right"]

            # 状态和动作
            frame["observation.state"] = np.array(
                frame_data.get('states', [0.0] * 16),
                dtype=np.float32
            )
            frame["action"] = np.array(
                frame_data.get('action', [0.0] * 14),
                dtype=np.float32
            )

            # 添加到数据集
            self.dataset.add_frame(frame, task=self.task_prompt)

            # 进度显示
            if (i + 1) % 30 == 0:
                print(f"   📝 已转换 {i + 1}/{len(frames)} 帧", end="\r")

        print(f"\n✅ 已添加 {len(frames)} 帧到 episode buffer")
        return True

    def save_episode(self):
        """保存 episode 并触发视频编码"""
        if self.dataset is None:
            print("⚠️  数据集未初始化")
            return False

        print("💾 保存 episode...")
        self.dataset.save_episode()

        print("🎬 触发视频编码...")
        self.video_manager.__exit__(None, None, None)
        self.dataset.stop_image_writer()

        print(f"✅ Episode 已保存到：{self.dataset_path}")
        print(f"   总 episodes: {self.dataset.num_episodes}")
        print(f"   总 frames: {self.dataset.num_frames}")
        return True

    def cleanup(self):
        """清理资源"""
        if self.video_manager:
            self.video_manager.__exit__(None, None, None)
        if self.dataset and hasattr(self.dataset, 'stop_image_writer'):
            self.dataset.stop_image_writer()
        print("✅ 资源已清理")


def convert_session_to_lerobot(
        npz_path: Path,
        task_name: str,
        task_prompt: str,
) -> bool:
    """
    将单个 session 的 npz 文件转换为 LeRobot 数据集的一个 episode
    - 如果数据集不存在则创建
    - 如果数据集存在则追加 episode

    Args:
        npz_path: npz 文件路径
        task_name: 任务名称 (用于目录命名)
        task_prompt: 任务描述 (硬编码)

    Returns:
        bool: 是否成功
    """
    print("=" * 60)
    print("🤖 LeRobot 数据集转换工具")
    print("=" * 60)
    print(f"📂 输入文件：{npz_path}")
    print(f"📝 任务：{task_name}")
    print(f"📋 Prompt: {task_prompt}")
    print("=" * 60)

    if not npz_path.exists():
        print(f"❌ 文件不存在：{npz_path}")
        return False

    # 创建录制器
    recorder = LeRobotRecorder(task_name=task_name, task_prompt=task_prompt)

    try:
        # 1. 加载或创建数据集
        is_new = recorder.load_or_create_dataset()
        if is_new:
            print(f"🆕 创建了新数据集：{recorder.dataset_path}")
        else:
            print(f"📂 追加到现有数据集：{recorder.dataset_path}")

        # 2. 转换数据
        success = recorder.convert_npz_to_lerobot(npz_path)
        if not success:
            return False

        # 3. 保存 episode
        recorder.save_episode()

        # 4. 删除临时 npz 文件
        print(f"\n🗑️  删除临时文件：{npz_path}")
        npz_path.unlink()

        print("\n" + "=" * 60)
        print("✅ 转换完成！")
        print("=" * 60)
        print(f"📁 数据集位置：{recorder.dataset_path}")
        print(f"\n🔍 可视化命令:")
        print(f"   python -m lerobot.scripts.visualize_dataset \\")
        print(f"       --repo-id {recorder.repo_id} \\")
        print(f"       --root {Config.DATASET_ROOT} \\")
        print(f"       --episode-index {recorder.dataset.num_episodes - 1}")

        return True

    except Exception as e:
        print(f"❌ 转换失败：{e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        recorder.cleanup()


def merge_lerobot_datasets(
        dataset_paths: list[Path],
        output_name: str,
) -> bool:
    """
    合并多个已存在的 LeRobot 数据集到一个大合集
    保留每个 episode 的原始 task 描述字符串

    Args:
        dataset_paths: LeRobot 数据集路径列表
        output_name: 输出数据集名称

    Returns:
        bool: 是否成功
    """
    print("=" * 60)
    print("🤖 LeRobot 数据集合并工具")
    print("=" * 60)
    print(f"📂 输入数据集：{len(dataset_paths)} 个")
    for p in dataset_paths:
        print(f"   - {p}")
    print(f"📁 输出数据集：{output_name}")
    print("=" * 60)

    if len(dataset_paths) == 0:
        print("⚠️  没有可合并的数据集")
        return False

    output_path = Path(Config.DATASET_ROOT) / output_name
    output_repo_id = f"g01/{output_name.replace(' ', '_')}"

    # 检查输出目录是否已存在
    if output_path.exists():
        print(f"⚠️  输出目录已存在：{output_path}")
        response = input("是否覆盖？(y/n): ")
        if response.lower() != 'y':
            print("❌ 取消合并")
            return False
        shutil.rmtree(output_path)

    try:
        # 1. 收集所有唯一的 task 描述
        all_tasks = {}  # task_prompt -> task_index
        task_index_counter = 0

        datasets_info = []
        total_frames = 0
        total_episodes = 0

        for i, ds_path in enumerate(dataset_paths):
            if not ds_path.exists():
                print(f"⚠️  数据集不存在，跳过：{ds_path}")
                continue

            print(f"\n📂 加载数据集 {i + 1}/{len(dataset_paths)}: {ds_path.name}")
            ds = LeRobotDataset(
                ds_path.name,
                root=Config.DATASET_ROOT,
                download_videos=True,
            )

            # 收集该数据集的所有 tasks
            ds_tasks = ds.meta.tasks  # {task_index: task_prompt}
            print(f"   该数据集 tasks: {ds_tasks}")

            # 映射：确保每个 task_prompt 有全局唯一的 task_index
            for local_idx, task_prompt in ds_tasks.items():
                if task_prompt not in all_tasks:
                    all_tasks[task_prompt] = task_index_counter
                    task_index_counter += 1

            datasets_info.append({
                "dataset": ds,
                "episodes": ds.meta.episodes,  # {ep_index: {episode_index, tasks, length}}
            })

            total_frames += ds.num_frames
            total_episodes += ds.num_episodes
            print(f"   Episodes: {ds.num_episodes}, Frames: {ds.num_frames}")

        if len(datasets_info) == 0:
            print("❌ 没有成功加载任何数据集")
            return False

        print(f"\n📋 合并后总 tasks: {len(all_tasks)}")
        for prompt, idx in all_tasks.items():
            print(f"   [{idx}] {prompt[:60]}...")

        # 2. 创建输出数据集
        print(f"\n🆕 创建输出数据集：{output_path}")
        features = build_features()
        output_dataset = LeRobotDataset.create(
            repo_id=output_repo_id,
            fps=Config.FPS,
            features=features,
            root=Config.DATASET_ROOT,
            robot_type="custom_g01_bimanual",
            use_videos=Config.USE_VIDEOS,
            image_writer_processes=Config.IMAGE_WRITER_PROCESSES,
            image_writer_threads=Config.IMAGE_WRITER_THREADS,
            batch_encoding_size=Config.VIDEO_ENCODING_BATCH_SIZE,
        )
        output_dataset.start_image_writer(
            num_processes=Config.IMAGE_WRITER_PROCESSES,
            num_threads=Config.IMAGE_WRITER_THREADS,
        )

        video_manager = VideoEncodingManager(output_dataset)
        video_manager.__enter__()

        # 3. 手动写入 tasks.jsonl（使用收集到的所有唯一 task）
        from lerobot.datasets.utils import write_json, TASKS_PATH
        tasks_list = [
            {"task_index": idx, "task": prompt}
            for prompt, idx in sorted(all_tasks.items(), key=lambda x: x[1])
        ]
        write_json(tasks_list, output_path / TASKS_PATH)

        # 4. 复制所有 episode（保留原始 task 描述字符串）
        print(f"\n📋 开始合并 {total_episodes} 个 episodes...")
        episode_count = 0

        for ds_info in datasets_info:
            ds = ds_info["dataset"]
            episodes = ds_info["episodes"]

            for local_ep_idx in range(ds.num_episodes):
                print(f"   处理 episode {episode_count + 1}/{total_episodes}", end="\r")

                # 获取该 episode 的 task 信息（任务描述字符串列表）
                ep_meta = episodes.get(local_ep_idx, {})
                ep_tasks = ep_meta.get("tasks", [])  # 这是任务描述字符串列表！

                # 获取该 episode 的所有帧
                ep_frames = []
                for frame_idx in range(ds.num_frames):
                    item = ds.hf_dataset[frame_idx]
                    if item["episode_index"].item() == local_ep_idx:
                        ep_frames.append(frame_idx)

                if len(ep_frames) == 0:
                    continue

                # 逐帧添加到输出数据集（使用原始 task 描述字符串）
                for frame_idx in ep_frames:
                    item = ds.hf_dataset[frame_idx]
                    frame = {}

                    # 复制所有特征
                    for key in features.keys():
                        if key in item:
                            val = item[key]
                            if hasattr(val, 'numpy'):
                                val = val.numpy()
                            frame[key] = val

                    # 使用原始 task 描述字符串（第一个 task）
                    task_prompt = ep_tasks[0] if ep_tasks else "Unknown"
                    output_dataset.add_frame(frame, task=task_prompt)

                # 保存该 episode
                output_dataset.save_episode()
                episode_count += 1

        # 5. 完成编码
        video_manager.__exit__(None, None, None)
        output_dataset.stop_image_writer()

        # 6. 验证合并后的 meta
        print(f"\n\n✅ 合并完成！")
        print(f"📁 输出位置：{output_path}")
        print(f"📊 总 episodes: {output_dataset.num_episodes}")
        print(f"📊 总 frames: {output_dataset.num_frames}")
        print(f"📊 总 tasks: {output_dataset.meta.total_tasks}")

        # 验证 episodes.jsonl 格式
        print(f"\n📋 episodes.jsonl 示例:")
        with open(output_path / "meta" / "episodes.jsonl") as f:
            for i, line in enumerate(f):
                if i < 3:
                    print(f"   {line.strip()}")

        # 验证 tasks.jsonl 格式
        print(f"\n📋 tasks.jsonl 示例:")
        with open(output_path / "meta" / "tasks.jsonl") as f:
            for i, line in enumerate(f):
                if i < 3:
                    print(f"   {line.strip()}")

        return True

    except Exception as e:
        print(f"❌ 合并失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """命令行工具入口"""
    import argparse

    parser = argparse.ArgumentParser(description="LeRobot 数据集工具")
    subparsers = parser.add_subparsers(dest="command", help="命令类型")

    # convert 命令
    convert_parser = subparsers.add_parser("convert", help="将 npz 转换为 LeRobot 数据集")
    convert_parser.add_argument("--input", type=str, required=True, help="输入 npz 文件路径")
    convert_parser.add_argument("--task-name", type=str, required=True, help="任务名称")
    convert_parser.add_argument("--task-prompt", type=str, required=True, help="任务描述")

    # merge 命令
    merge_parser = subparsers.add_parser("merge", help="合并多个 LeRobot 数据集")
    merge_parser.add_argument("--inputs", type=str, required=True, help="输入数据集路径 (逗号分隔)")
    merge_parser.add_argument("--output", type=str, required=True, help="输出数据集名称")
    merge_parser.add_argument("--prompt", type=str, default="Merged dataset", help="任务描述")

    args = parser.parse_args()

    if args.command == "convert":
        success = convert_session_to_lerobot(
            npz_path=Path(args.input),
            task_name=args.task_name,
            task_prompt=args.task_prompt,
        )
    elif args.command == "merge":
        input_paths = [Path(p.strip()) for p in args.inputs.split(",")]
        success = merge_lerobot_datasets(
            dataset_paths=input_paths,
            output_name=args.output,
            output_prompt=args.prompt,
        )
    else:
        parser.print_help()
        success = False

    exit(0 if success else 1)


if __name__ == "__main__":
    main()