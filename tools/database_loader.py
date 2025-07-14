import pandas as pd
import os
import glob
from ..config import DATABASE_DIR, DATABASE_CSV_FILES, validate_config

# 全局变量，用于存储加载后的数据库DataFrames
DB = {} 

def load_database():
    """
    加载数据库目录中的所有CSV文件到全局的DB字典中。
    使用config.py中的配置。
    """
    print("正在加载数据库到内存...")
    
    # 验证配置
    config_errors = validate_config()
    if config_errors:
        print("配置错误:")
        for error in config_errors:
            print(f"  - {error}")
        return
    
    if not DATABASE_DIR.exists():
        print(f"致命错误：数据库目录未找到于 '{DATABASE_DIR}'。Agent无法启动。")
        return

    # 使用配置中的CSV文件列表
    for csv_file in DATABASE_CSV_FILES:
        file_path = DATABASE_DIR / csv_file
        if not file_path.exists():
            print(f"警告：数据表文件 '{csv_file}' 不存在于 '{DATABASE_DIR}'")
            continue
            
        try:
            # 使用文件名（不含扩展名）作为字典的键
            key = csv_file.split('.')[0]
            DB[key] = pd.read_csv(file_path)
            print(f"  - 已加载数据表 '{key}' (共 {len(DB[key])} 行)")
        except Exception as e:
            print(f"  - 加载数据表 '{csv_file}' 失败: {e}")
    
    print("数据库加载完成。")

# 当此模块被首次导入时，自动执行加载数据库的操作
load_database()