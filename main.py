#!/usr/bin/env python3
"""
生物反应科研Agent主程序

这个文件用于启动ADK Web开发UI界面。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """主函数"""
    print("🚀 启动生物反应科研Agent...")
    
    # 检查环境变量
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  警告: 未设置GEMINI_API_KEY环境变量")
        print("请设置环境变量: export GEMINI_API_KEY='your_api_key'")
        print("Agent功能可能受限...\n")
    
    # 验证配置
    try:
        from bioreaction_adk_agent.config import validate_config
        config_errors = validate_config()
        if config_errors:
            print("❌ 配置错误:")
            for error in config_errors:
                print(f"  - {error}")
            print("\n请修复配置错误后重试。")
            sys.exit(1)
        else:
            print("✅ 配置验证通过")
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        sys.exit(1)
    
    # 检查数据库加载
    try:
        from bioreaction_adk_agent.tools.database_loader import DB
        if not DB:
            print("❌ 数据库未加载")
            sys.exit(1)
        print(f"✅ 数据库已加载，包含 {len(DB)} 个表")
    except Exception as e:
        print(f"❌ 数据库加载失败: {e}")
        sys.exit(1)
    
    # 检查Agent
    try:
        from bioreaction_adk_agent.agent import root_agent
        print(f"✅ Agent已准备就绪: {root_agent.name}")
    except Exception as e:
        print(f"❌ Agent初始化失败: {e}")
        sys.exit(1)
    
    print("\n🎯 系统启动成功！")
    print("现在可以使用 'adk web' 命令启动开发UI界面。")
    print("\n使用说明:")
    print("1. 在终端中运行: adk web")
    print("2. 浏览器会自动打开开发界面")
    print("3. 开始与生物反应科研Agent交互")
    print("\n示例查询:")
    print("- '查找ADK酶的反应'")
    print("- '分析E. coli中酶反应的性能趋势'")
    print("- '比较不同物种的酶活性'")
    print("- '获取数据库统计信息'")

if __name__ == "__main__":
    main() 