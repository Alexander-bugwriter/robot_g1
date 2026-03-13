#!/usr/bin/env python3
"""
记录机器人 Home 位姿（回正后的关节角度）
使用方法：
  1. 先用 robot-controller 的 re 命令回正
  2. 运行此脚本读取并保存关节角度
"""

import time
import json
from pathlib import Path
from a2d_sdk.robot import RobotDds

def to_list(obj):
    """将 protobuf RepeatedScalarFieldContainer 转换为普通 list"""
    if obj is None:
        return None
    try:
        return list(obj)
    except (TypeError, ValueError):
        return obj

def save_home_pose(output_file: str = "home_pose.json"):
    print("🔌 连接机器人...")
    robot = RobotDds()
    time.sleep(2)  # 等待 DDS 连接
    
    print("📍 读取关节状态...")
    
    # 读取各部分状态
    arm_states, ts_arm = robot.arm_joint_states()
    head_states, ts_head = robot.head_joint_states()
    waist_states, ts_waist = robot.waist_joint_states()
    gripper_states, ts_gripper = robot.gripper_states()
    
    # 检查数据有效性
    if not arm_states or None in arm_states:
        print("❌ 手臂关节状态无效")
        robot.shutdown()
        return False
    
    # ========== 关键修复：转换为原生 Python 类型 ==========
    home_pose = {
        "timestamp": time.time(),
        "arm_joint_states": to_list(arm_states),  # ← 转换！
        "head_joint_states": to_list(head_states) if head_states else [0.0, 0.0],
        "waist_joint_states": to_list(waist_states) if waist_states else [0.0, 30.0],
        "gripper_states": to_list(gripper_states) if gripper_states else [35.0, 35.0],
    }
    # =====================================================
    
    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(home_pose, f, indent=2)
    
    print(f"✅ Home 位姿已保存到 {output_file}")
    print("\n📊 关节角度详情:")
    print(f"   手臂 (14 关节，弧度): {home_pose['arm_joint_states']}")
    print(f"   头部 [yaw, pitch]: {home_pose['head_joint_states']}")
    print(f"   腰部 [pitch_rad, height_cm]: {home_pose['waist_joint_states']}")
    print(f"   夹爪 [left_mm, right_mm]: {home_pose['gripper_states']}")
    
    robot.shutdown()
    return True

if __name__ == "__main__":
    save_home_pose()
