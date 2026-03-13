#!/usr/bin/env python3
import time
import sys
from a2d_sdk.robot import RobotDds as Robot

def test_connection():
    print("🔄 初始化 RobotDds...")
    try:
        robot = Robot()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    # 等待数据就绪（关键！）
    print("⏳ 等待机器人数据就绪 (最多10秒)...")
    for i in range(20):  # 20 * 0.5s = 10s
        time.sleep(0.5)
        head, ts = robot.head_joint_states()
        if head and None not in head:
            print(f"✅ 连接成功! 头部状态: {head}, 时间戳: {ts}")
            break
    else:
        print("❌ 超时：未收到有效数据，请检查：")
        print("   1. 机器人是否切换到 copilot 模式？")
        print("   2. 网线是否连接正确？")
        print("   3. 是否等待了足够初始化时间？")
        robot.shutdown()
        return False
    
    # 测试其他接口
    try:
        arm, ts = robot.arm_joint_states()
        print(f"✅ 手臂状态: {arm[:3]}... (显示前3个关节)")
        
        gripper, ts = robot.gripper_states()
        print(f"✅ 夹爪状态: {gripper}")
    except Exception as e:
        print(f"⚠️ 读取状态异常: {e}")
    
    # 安全关闭
    robot.shutdown()
    print("🔌 连接已关闭")
    return True

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
