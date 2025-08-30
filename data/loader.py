

import glob
import numpy as np
from typing import List, Dict, Any


def batch_read_processed_npz(
        folder_path: str,
        required_fields: set = {'time', 'original', 'fs', 'viscosity'},
        default_values: dict = {'density': 1000, 'diameter': 0.01}
) -> List[Dict[str, Any]]:
    file_list = glob.glob(f"{folder_path}/labeled_receiver2__*.npz")
    if not file_list:
        raise FileNotFoundError(f"No matching NPZ files found in {folder_path}")
    results = []

    for file_path in file_list:
        try:
            with np.load(file_path) as data:
                # 验证必需字段
                missing = required_fields - set(data.files)
                if missing:
                    print(f"跳过文件 {file_path}: 缺少字段 {missing}")
                    continue

                # 构建结果字典
                result = {
                    'time': data['time'],
                    'fs': data['fs'].item(),
                    'signals': {
                        'original': data['original'],
                        'convolved': data.get('convolved', None),
                        'final': data.get('final', None)
                    },
                    'labels': {
                        'viscosity': data['viscosity'].item(),
                        'velocity': data['velocity'].item(),
                        'density': data.get('density', default_values['density']).item(),
                        'diameter': data.get('diameter', default_values['diameter']).item(),
                        'reynolds': data['reynolds'].item(),
                        'st': data['st'].item()
                    },
                    'file_path': file_path  # 保留源文件路径
                }
                results.append(result)

        except Exception as e:
            print(f"处理文件 {file_path} 失败: {str(e)}")
            continue

    return results






