有线网络已经配置好了
使用前先
cd a2d_sdk
source env.sh
robot-service -s -c ./conf/copilot.pbtxt --no-ros

重置机器人 
robot-controller re

可以调用来采集数据集 保存的lerobot子集目录就在record_lerobot_dataset脚本的config里
python scripts/teleop.py

可以用合并数据集
python scripts/record_lerobot_dataset.py merge     --inputs "./test_datasets,./test_datasets_2,./test_datasets_3"     --output "./test_datasets_all"


可以用可视化
python -m lerobot.scripts.visualize_dataset        --repo-id test_datasets_all        --root ./test_datasets_all        --episode-index 0

