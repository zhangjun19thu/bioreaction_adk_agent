"""
生物反应科研Agent配置文件

这个文件包含了系统运行所需的各种配置参数。
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据库配置
DATABASE_DIR = PROJECT_ROOT / "data" / "papers1000_database"
DATABASE_CSV_FILES = [
    "1_reactions_core.csv",
    "2_enzymes.csv", 
    "3_experimental_conditions.csv",
    "4_activity_performance.csv",
    "5_reaction_participants.csv",
    "6_kinetic_parameters.csv",
    "7_mutants_characterized.csv",
    "8_inhibitors_main.csv",
    "9_inhibition_params.csv",
    "10_auxiliary_factors.csv"
]

# 文献元数据配置
METADATA_BASE_DIR = "/share/6_19batch_label/papers1000_parser"

# Agent配置
AGENT_CONFIG = {
    "model": "gemini-2.5-flash",
    "app_name": "bioreaction_research_app",
    "session_service": "InMemorySessionService"
}

# 查询配置
QUERY_CONFIG = {
    "max_results": 10,
    "min_data_points": 5,
    "default_top_n": 5
}

# 分析配置
ANALYSIS_CONFIG = {
    "temperature_ranges": {
        "low": (0, 20),
        "room": (20, 37), 
        "medium": (37, 50),
        "high": (50, 100)
    },
    "ph_ranges": {
        "acidic": (0, 5),
        "weak_acidic": (5, 7),
        "neutral_weak_basic": (7, 9),
        "strong_basic": (9, 14)
    },
    "correlation_threshold": 0.3
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "agent.log"
}

# 缓存配置
CACHE_CONFIG = {
    "enable": True,
    "ttl": 3600,  # 1小时
    "max_size": 1000
}

# 错误处理配置
ERROR_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "show_traceback": False
}

def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查数据库目录
    if not DATABASE_DIR.exists():
        errors.append(f"数据库目录不存在: {DATABASE_DIR}")
    
    # 检查文献元数据目录
    if not os.path.exists(METADATA_BASE_DIR):
        errors.append(f"文献元数据目录不存在: {METADATA_BASE_DIR}")
    
    # 检查必要的环境变量
    if not os.getenv("GEMINI_API_KEY"):
        errors.append("未设置GEMINI_API_KEY环境变量")
    
    return errors

def get_database_path(table_name: str) -> Path:
    """获取指定数据表的完整路径"""
    return DATABASE_DIR / f"{table_name}.csv"

def get_metadata_path(literature_id: str) -> Path:
    """获取指定文献的元数据文件路径"""
    return Path(METADATA_BASE_DIR) / literature_id / f"{literature_id}_parser.md"

# 导出配置
__all__ = [
    'PROJECT_ROOT',
    'DATABASE_DIR', 
    'DATABASE_CSV_FILES',
    'METADATA_BASE_DIR',
    'AGENT_CONFIG',
    'QUERY_CONFIG',
    'ANALYSIS_CONFIG',
    'LOG_CONFIG',
    'CACHE_CONFIG',
    'ERROR_CONFIG',
    'validate_config',
    'get_database_path',
    'get_metadata_path'
] 