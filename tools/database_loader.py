# tools/database_loader.py

import pandas as pd
from pathlib import Path

# 导入配置，这个保持不变
from ..CONFIG import DATABASE_DIR, DATABASE_CSV_FILES, validate_config

# 全局变量，用于存储加载后的数据库DataFrames
DB = {} 

def load_database():
    """
    加载数据库目录中的所有CSV文件到全局的DB字典中。
    这个函数现在需要被外部显式调用。
    """
    # 首先检查DATABASE_DIR变量是否存在
    if 'DATABASE_DIR' not in globals() or DATABASE_DIR is {}:
         raise ValueError("DIAGNOSTIC_TEST: DATABASE_DIR was not imported correctly from CONFIG.py.")

    # 抛出异常，将路径作为消息的一部分
    # raise ValueError(f"DIAGNOSTIC_TEST: The calculated database path is '{DATABASE_DIR.resolve()}'. Please verify if this path and its contents actually exist from the script's execution context.")

    # 如果已经加载过了，就不要重复加载
    if DB:
        print("数据库已经加载，无需重复操作。")
        return

    print("--- [INFO] 正在执行数据库加载程序... ---")
    
    config_errors = validate_config()
    if config_errors:
        print(f"配置错误: {config_errors}")
        return
    
    if not DATABASE_DIR.exists():
        print(f"致命错误：数据库目录未找到于 '{DATABASE_DIR.resolve()}'")
        return

    for csv_file in DATABASE_CSV_FILES:
        file_path = DATABASE_DIR / csv_file
        if not file_path.exists():
            print(f"警告：数据文件 '{file_path}' 不存在，跳过。")
            continue
        try:
            key = csv_file.split('.')[0]
            DB[key] = pd.read_csv(file_path)
            # print(f"  - 已加载数据表 '{key}'") # 在生产环境中可以注释掉，减少打印
        except Exception as e:
            print(f"  - 加载数据表 '{csv_file}' 失败: {e}")
    
    if not DB:
        print("--- [ERROR] 数据库加载完毕，但内容为空！请检查路径和文件。 ---")
    else:
        print(f"--- [INFO] 数据库加载成功，共 {len(DB)} 个数据表。 ---")

load_database()