#!/usr/bin/env python3
"""
智元 G01 双臂端云协同脚本
- 完整复用 teleop.py 的键盘控制逻辑与配置
- 支持模型预测自动控制 (AUTO MODE) 与 Action Chunking
- 支持 WebSocket 保活机制
- 自动处理数据维度转换 (16 <-> 24)
"""

import asyncio
import websockets
import json
import time
import os
import numpy as np
import cv2
import pygame
from pathlib import Path
from datetime import datetime


# ==========================================
# 1. 完整配置参数 (融合 teleop.py 的 Config)
# ==========================================
class Args:
    # --- 网络配置 ---
    SERVER_URL = "ws://0.0.0.0:8000"

    # --- 动作执行配置 ---
    action_interval = 0.5  # 显式时间间隔 (秒)：每个动作执行的时长
    n_action_steps = 10  # 一次推理后执行的步数
    prediction_horizon = 50  # 模型预测的总长度

    # --- WebSocket 保活 ---
    keepalive_interval = 5.0  # TELEOP 模式下心跳包间隔 (秒)

    # --- 机器人控制配置 (来自 teleop.py) ---
    DELTA_POS = 0.015  # 键盘控制单步位移
    DELTA_GRIPPER = 0.1  # 键盘控制夹爪增量
    GRIPPER_MIN = 0
    GRIPPER_MAX = 1
    CONTROL_TYPE = "DELTA_POSE"
    ROBOT_LINK = "arm_base_link"
    TRAJECTORY_TIME = 0.5  # 轨迹执行时间

    # --- 相机配置 ---
    CAMERA_NAMES = ["head", "hand_left", "hand_right"]
    CAMERA_RESIZE = (448, 448)

    # --- 数据维度 ---
    REAL_STATE_DIM = 16
    REAL_ACTION_DIM = 14
    MODEL_IO_DIM = 24

    # --- 任务配置 ---
    TASK_PROMPT = "Pick up the lemon and place it into the bowl"
    HOME_FILE = "home_pose.json"


# ==========================================
# 2. 键盘控制器 (完整移植自 teleop.py)
# ==========================================
class KeyboardController:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.display.set_caption("G1 Client")
        print("\n🎮 控制说明:")
        print(" [模式切换]")
        print(" P : 切换到遥操作模式
        print(" T : 切换到自动推理模式
        print(" R : 回正到 Home 位姿
        print(" ESC : 退出程序")
        print(" [遥操作控制]")
        print(" 左臂: W/S(A/D) X/Y, Z/X(Z), Q/E(Gripper)")
        print(" 右臂: I/K(J/L) X/Y, N/M(Z), U/O(Gripper)")

    def get_command(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # 1. 模式控制信号
        should_teleop = keys[pygame.K_p]
        should_auto = keys[pygame.K_t]
        should_reset = keys[pygame.K_r]
        should_exit = keys[pygame.K_ESCAPE]

        # 2. 遥操作动作计算 (完整移植)
        left_action = [0.0] * 7
        right_action = [0.0] * 7
        has_action = False

        # --- 左臂位移 ---
        if keys[pygame.K_w]: left_action[0] = Args.DELTA_POS; has_action = True
        if keys[pygame.K_s]: left_action[0] = -Args.DELTA_POS; has_action = True
        if keys[pygame.K_a]: left_action[1] = Args.DELTA_POS; has_action = True
        if keys[pygame.K_d]: left_action[1] = -Args.DELTA_POS; has_action = True
        if keys[pygame.K_z]: left_action[2] = Args.DELTA_POS; has_action = True
        if keys[pygame.K_x]: left_action[2] = -Args.DELTA_POS; has_action = True

        # --- 右臂位移 ---
        if keys[pygame.K_i]: right_action[0] = Args.DELTA_POS; has_action = True
        if keys[pygame.K_k]: right_action[0] = -Args.DELTA_POS; has_action = True
        if keys[pygame.K_j]: right_action[1] = Args.DELTA_POS; has_action = True
        if keys[pygame.K_l]: right_action[1] = -Args.DELTA_POS; has_action = True
        if keys[pygame.K_n]: right_action[2] = Args.DELTA_POS; has_action = True
        if keys[pygame.K_m]: right_action[2] = -Args.DELTA_POS; has_action = True

        # 注意：夹爪控制需要在主循环中维护状态，这里只返回信号
        # Q/E: 左夹爪, U/O: 右夹爪

        return {
            "switch_to_teleop": should_teleop,
            "switch_to_auto": should_auto,
            "should_reset": should_reset,
            "should_exit": should_exit,
            "teleop_left": left_action,
            "teleop_right": right_action,
            "has_teleop_input": has_action,
            "gripper_left_close": keys[pygame.K_q],
            "gripper_left_open": keys[pygame.K_e],
            "gripper_right_close": keys[pygame.K_u],
            "gripper_right_open": keys[pygame.K_o],
        }

    def cleanup(self):
        pygame.quit()


# ==========================================
# 3. 机器人控制核心逻辑
# ==========================================

def reset_to_home(robot, controller):
    """ 回正功能 (移植自 teleop.py) """
    print(f"🔄 执行回正: {Args.HOME_FILE}")
    if not os.path.exists(Args.HOME_FILE):
        print(f"❌ 未找到 Home 文件: {Args.HOME_FILE}")
        return

    try:
        with open(Args.HOME_FILE) as f:
            home = json.load(f)

        robot.move_gripper([35.0, 35.0])  # 松开夹爪
        time.sleep(0.2)

        arm_states, _ = robot.arm_joint_states()
        head_states, _ = robot.head_joint_states()
        waist_states, _ = robot.waist_joint_states()

        if not arm_states or None in arm_states:
            print("⚠️ 无法获取当前关节状态")
            return

        target_arm = home.get("arm_joint_states")
        if not target_arm or len(target_arm) != 14:
            print(f"⚠️ Home 文件关节数据无效")
            return

        robot_states = {
            "head": head_states if head_states else [0.0, 0.0],
            "waist": waist_states if waist_states else [0.0, 0.0],
            "arm": arm_states,
        }

        robot_actions = [{
            "left_arm": {"action_data": target_arm[:7], "control_type": "ABS_JOINT"},
            "right_arm": {"action_data": target_arm[7:], "control_type": "ABS_JOINT"}
        }]

        controller.trajectory_tracking_control(
            infer_timestamp=time.time_ns(),
            robot_states=robot_states,
            robot_actions=robot_actions,
            robot_link=Args.ROBOT_LINK,
            trajectory_reference_time=2.0
        )
        print("✅ 回正指令已发送")
        time.sleep(2.5)  # 等待执行完成

    except Exception as e:
        print(f"❌ 回正失败: {e}")


def send_trajectory_control(controller, robot, left_action, right_action):
    """
    发送 DELTA_POSE 控制指令
    参数:
    left_action/right_action: 7维 list [dx, dy, dz, drx, dry, drz, gripper]
    """
    arm_states, _ = robot.arm_joint_states()

    robot_states = {
        "head": [0.0, 0.0],
        "waist": [0.0, 0.0],
        "arm": arm_states,
    }

    robot_actions = [{
        "left_arm": {
            "action_data": left_action[:6],  # 只取位姿增量
            "control_type": Args.CONTROL_TYPE
        },
        "right_arm": {
            "action_data": right_action[:6],
            "control_type": Args.CONTROL_TYPE
        }
    }]

    controller.trajectory_tracking_control(
        infer_timestamp=time.time_ns(),
        robot_states=robot_states,
        robot_actions=robot_actions,
        robot_link=Args.ROBOT_LINK,
        trajectory_reference_time=Args.TRAJECTORY_TIME
    )


def prepare_observation_payload(images, arm_states, gripper_states):
    """ 组装 Server 需要的 JSON 数据 (State 16->24) """
    # 1. State 处理
    real_state = list(arm_states) + list(gripper_states)
    if len(real_state) != Args.REAL_STATE_DIM:
        # 如果数据不对，填充0保底
        print(f"⚠️ State 维度异常: {len(real_state)}")
        real_state = [0.0] * Args.REAL_STATE_DIM

    padded_state = real_state + [0.0] * (Args.MODEL_IO_DIM - Args.REAL_STATE_DIM)

    # 2. Image 处理
    img_list = []
    for img in images:
        if img.shape[:2] != Args.CAMERA_RESIZE:
            img = cv2.resize(img, Args.CAMERA_RESIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img_rgb.astype(np.uint8).tolist())

    # 3. 组装
    payload = {
        "image": img_list,
        "state": padded_state,
        "prompt": Args.TASK_PROMPT,
        "image_mask": [1, 1, 1],
        "action_mask": [1] * 14 + [0] * 10
    }
    return json.dumps(payload)


def execute_model_action(controller, robot, action_24d):
    """ 执行模型预测的单个动作 (Action 24->14) """
    # 1. 截取前 14 维
    action_14d = action_24d[:Args.REAL_ACTION_DIM]

    # 2. 分离左右臂
    left_cmd = action_14d[:7]
    right_cmd = action_14d[7:]

    # 3. 发送位姿控制
    send_trajectory_control(controller, robot, left_cmd, right_cmd)

    # 4. 控制夹爪 (绝对位置)
    # 假设模型输出值域符合机器人接口要求
    robot.move_gripper([left_cmd[6], right_cmd[6]])


# ==========================================
# 4. 主程序
# ==========================================

async def main():
    print("🤖 初始化 G1 Client...")

    # 1. 硬件初始化
    try:
        from a2d_sdk.robot import RobotDds, RobotController, CosineCamera
    except ImportError:
        print("❌ 未找到 a2d_sdk，请检查环境")
        return

    robot = RobotDds()
    controller = RobotController()
    camera = CosineCamera(Args.CAMERA_NAMES)
    keyboard = KeyboardController()

    # 2. 初始状态
    mode = "AUTO"
    last_heartbeat_time = time.time()

    # 夹爪状态变量 (用于遥操作模式)
    left_gripper = 0.5
    right_gripper = 0.5

    print(f"🔌 连接服务器: {Args.SERVER_URL}")

    try:
        async with websockets.connect(Args.SERVER_URL, max_size=100_000_000) as ws:
            print("✅ 连接成功！开始运行...")

            while True:
                # ==============================
                # Step 1: 获取观测数据 (通用)
                # ==============================
                images = []
                for name in Args.CAMERA_NAMES:
                    img, ts = camera.get_latest_image(name)
                    images.append(img if img is not None else np.zeros((448, 448, 3), dtype=np.uint8))

                arm_states, _ = robot.arm_joint_states()
                gripper_states, _ = robot.gripper_states()

                # ==============================
                # Step 2: 键盘输入处理
                # ==============================
                cmd = keyboard.get_command()

                if cmd["should_exit"]:
                    break

                if cmd["should_reset"]:
                    reset_to_home(robot, controller)
                    # 回正后重置夹爪状态
                    left_gripper = 0.5
                    right_gripper = 0.5
                    mode = "AUTO"
                    continue

                # 模式切换逻辑
                if cmd["switch_to_teleop"] and mode != "TELEOP":
                    print("⏸️ 切换至 TELEOP 模式")
                    mode = "TELEOP"
                    # 防止按键粘连
                    time.sleep(0.3)

                elif cmd["switch_to_auto"] and mode != "AUTO":
                    print("▶️ 切换至 AUTO 模式")
                    mode = "AUTO"
                    time.sleep(0.3)

                # ==============================
                # Step 3: 核心执行逻辑
                # ==============================

                if mode == "AUTO":
                    # --- AUTO MODE: 模型推理 ---

                    # A. 发送观测
                    payload = prepare_observation_payload(images, arm_states, gripper_states)
                    await ws.send(payload)

                    # B. 接收 Action Chunk
                    try:
                        recv_result = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        actions_chunk = json.loads(recv_result)
                    except asyncio.TimeoutError:
                        print("⚠️ 接收超时，跳过本帧")
                        continue

                    # C. 执行 Action Chunk (前 N 步)
                    steps_to_run = min(len(actions_chunk), Args.n_action_steps)

                    for i in range(steps_to_run):
                        # **紧急接管检测**: 在动作序列执行中持续检测 P 键
                        pygame.event.pump()
                        if pygame.key.get_pressed()[pygame.K_p]:
                            print("⚠️ 检测到接管信号，中断动作序列")
                            mode = "TELEOP"
                            break

                        # 执行单步动作
                        current_action = actions_chunk[i]
                        execute_model_action(controller, robot, current_action)

                        # 显式等待 (控制频率)
                        time.sleep(Args.action_interval)

                elif mode == "TELEOP":
                    # --- TELEOP MODE: 键盘控制 ---

                    # A. 处理夹爪状态更新
                    if cmd["gripper_left_close"]:
                        left_gripper = min(Args.GRIPPER_MAX, left_gripper + Args.DELTA_GRIPPER)
                    if cmd["gripper_left_open"]:
                        left_gripper = max(Args.GRIPPER_MIN, left_gripper - Args.DELTA_GRIPPER)
                    if cmd["gripper_right_close"]:
                        right_gripper = min(Args.GRIPPER_MAX, right_gripper + Args.DELTA_GRIPPER)
                    if cmd["gripper_right_open"]:
                        right_gripper = max(Args.GRIPPER_MIN, right_gripper - Args.DELTA_GRIPPER)

                    # B. 如果有位移或夹爪输入，执行动作
                    if cmd["has_teleop_input"]:
                        # 将当前夹爪值填入 action
                        left_action = cmd["teleop_left"]
                        right_action = cmd["teleop_right"]
                        left_action[6] = left_gripper
                        right_action[6] = right_gripper

                        # 发送控制指令
                        send_trajectory_control(controller, robot, left_action, right_action)
                        robot.move_gripper([left_gripper, right_gripper])

                    # C. WebSocket 保活
                    current_time = time.time()
                    if current_time - last_heartbeat_time > Args.keepalive_interval:
                        # 发送当前观测以保持连接
                        payload = prepare_observation_payload(images, arm_states, gripper_states)
                        await ws.send(payload)
                        # 接收并丢弃模型预测 (防止缓冲区堆积)
                        _ = await ws.recv()
                        last_heartbeat_time = current_time
                        print("💓 [TELEOP] Heartbeat sent.")

                    # 遥操循环频率控制
                    time.sleep(0.01)

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔌 关闭连接...")
        keyboard.cleanup()
        camera.close()
        robot.shutdown()
        print("✅ 安全退出")


if __name__ == "__main__":
    # 兼容无显示器环境
    if "DISPLAY" not in os.environ and os.name != 'nt':
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    asyncio.run(main())
