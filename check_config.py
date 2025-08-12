#!/usr/bin/env python3
"""
生物反应科研Agent配置检查脚本
"""

import os
import sys
from pathlib import Path

def check_environment():
    """检查环境变量"""
    print("=== 环境变量检查 ===")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"✅ GEMINI_API_KEY: 已设置")
    else:
        print("❌ GEMINI_API_KEY: 未设置")
        return False
    
    python_version = sys.version_info
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n=== 项目结构检查 ===")
    
    project_root = Path(__file__).parent
    required_files = [
        "agent_config.py",
        "agent.py", 
        "main.py",
        "test_system.py",
        "requirements.txt",
        "__init__.py",
        "tools/__init__.py",
        "tools/database_loader.py",
        "tools/database_query_tools.py",
        "tools/deep_research_tools.py",
        "tools/advanced_tools.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ 缺少 {len(missing_files)} 个必需文件")
        return False
    
    print("✅ 项目结构完整")
    return True

def check_dependencies():
    """检查依赖包"""
    print("\n=== 依赖包检查 ===")
    
    required_packages = [
        "google.adk",
        "pandas",
        "numpy",
        "google.genai"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少 {len(missing_packages)} 个依赖包")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_config_file():
    """检查配置文件"""
    print("\n=== 配置文件检查 ===")
    
    try:
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from bioreaction_adk_agent.CONFIG import (
            DATABASE_DIR, 
            METADATA_BASE_DIR,
            AGENT_CONFIG,
            QUERY_CONFIG,
            ANALYSIS_CONFIG,
            validate_config
        )
        
        print("✅ 配置文件导入成功")
        print(f"✅ 数据库目录: {DATABASE_DIR}")
        print(f"✅ 元数据目录: {METADATA_BASE_DIR}")
        print(f"✅ Agent模型: {AGENT_CONFIG['model']}")
        
        config_errors = validate_config()
        if config_errors:
            print("❌ 配置验证失败:")
            for error in config_errors:
                print(f"  - {error}")
            return False
        else:
            print("✅ 配置验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件检查失败: {e}")
        return False

def check_database():
    """检查数据库文件"""
    print("\n=== 数据库检查 ===")
    
    try:
        from bioreaction_adk_agent.CONFIG import DATABASE_DIR, DATABASE_CSV_FILES
        
        if not DATABASE_DIR.exists():
            print(f"❌ 数据库目录不存在: {DATABASE_DIR}")
            return False
        
        print(f"✅ 数据库目录存在: {DATABASE_DIR}")
        
        missing_files = []
        for csv_file in DATABASE_CSV_FILES:
            file_path = DATABASE_DIR / csv_file
            if file_path.exists():
                print(f"✅ {csv_file}")
            else:
                print(f"❌ {csv_file} - 文件不存在")
                missing_files.append(csv_file)
        
        if missing_files:
            print(f"\n❌ 缺少 {len(missing_files)} 个数据库文件")
            return False
        
        print("✅ 所有数据库文件存在")
        return True
        
    except Exception as e:
        print(f"❌ 数据库检查失败: {e}")
        return False

def check_agent_creation():
    """检查Agent创建"""
    print("\n=== Agent创建检查 ===")
    
    try:
        from bioreaction_adk_agent.agent import root_agent
        
        print(f"✅ 主Agent: {root_agent.name}")
        tool_count = len(root_agent.tools)
        print(f"✅ 工具数量: {tool_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent创建检查失败: {e}")
        return False

def check_database_loading():
    """检查数据库加载"""
    print("\n=== 数据库加载检查 ===")
    
    try:
        from bioreaction_adk_agent.tools.database_loader import DB
        
        if not DB:
            print("❌ 数据库未加载")
            return False
        
        print(f"✅ 数据库已加载，包含 {len(DB)} 个表")
        
        for table_name, df in DB.items():
            print(f"  - {table_name}: {len(df)} 行")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库加载检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 生物反应科研Agent配置检查")
    print("=" * 50)
    
    checks = [
        ("环境变量", check_environment),
        ("项目结构", check_project_structure),
        ("依赖包", check_dependencies),
        ("配置文件", check_config_file),
        ("数据库文件", check_database),
        ("Agent创建", check_agent_creation),
        ("数据库加载", check_database_loading),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name} 检查通过")
            else:
                print(f"❌ {check_name} 检查失败")
        except Exception as e:
            print(f"❌ {check_name} 检查异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 检查结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有检查通过！系统配置正确。")
        print("\n现在可以:")
        print("1. 运行测试: python -m bioreaction_adk_agent.test_system")
        print("2. 启动UI: adk web")
        print("3. 使用API: python main.py")
        return True
    else:
        print("⚠️  部分检查失败，请修复问题后重试。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 