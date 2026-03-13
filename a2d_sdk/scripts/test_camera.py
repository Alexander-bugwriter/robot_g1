#!/usr/bin/env python3
import time
import cv2  # 需要 opencv-python
from a2d_sdk.robot import CosineCamera as Camera

def test_camera():
    print("📷 初始化相机...")
    
    # 初始化相机组（根据机器人配置选择）
    camera_names = ["head", "hand_left", "hand_right"]
    camera = Camera(camera_names)
    
    time.sleep(1)  # 等待相机初始化
    
    print("🔄 获取图像 (第一帧可能是 None，正常)...")
    for i in range(5):
        image, timestamp = camera.get_latest_image("hand_left")
        if image is not None:
            print(f"✅ 第{i+1}帧: 形状={image.shape}, 时间戳={timestamp}")
            # 可选：显示图像
            cv2.imshow("Camera", image)
            cv2.waitKey(5)
        else:
            print(f"⏳ 第{i+1}帧: None (等待中...)")
        time.sleep(0.5)
    
    # 测试延迟统计
    fps = camera.get_fps("head")
    print(f"📊 头部相机帧率: {fps} FPS")
    
    # 重要：释放资源
    camera.close()
    print("🔌 相机连接已关闭")

if __name__ == "__main__":
    try:
        test_camera()
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
