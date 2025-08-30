import numpy as np
import os

# 配置路径
input_dir = "../pythonProject/processed_results/"  # 输入文件目录
output_dir = "../pythonProject/labeled_results/"  # 输出目录（需提前创建）
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

# 标签模板（固定参数）
fixed_labels = {
    "density": 1000,
    "diameter": 0.01,
    "reynolds": 100,
    "st": 0.16
}

# 遍历目录下的所有.npz文件
for filename in os.listdir(input_dir):
    if filename.endswith(".npz"):
        file_path = os.path.join(input_dir, filename)

        try:
            # 1. 加载原始数据
            data = np.load(file_path, allow_pickle=True)
            time = data['time']
            original = data['original']
            convolved = data['convolved']
            final = data['final']
            fs = data['fs']

            # 2. 解析文件名中的参数（粘度、速度）
            params_part = filename.split('__')[1].split('_processed')[0]
            params = params_part.split('_')

            if len(params) >= 2:
                viscosity = float(params[0])
                velocity = float(params[1])
            else:
                print(f"跳过文件 {filename}：参数不足")
                continue

            # 3. 合并标签
            labels = {
                "viscosity": viscosity,
                "velocity": velocity,
             ** fixed_labels
            }

            # 4. 保存新文件（保留原名，避免覆盖）
            output_path = os.path.join(output_dir, f"labeled_{filename}")
            np.savez_compressed(
                output_path,
                time=time,
                original=original,
                convolved=convolved,
                final=final,
                fs=fs,
             ** labels
            )

            print(f"处理完成: {filename} -> labeled_{filename}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue

print("批量处理完成！")