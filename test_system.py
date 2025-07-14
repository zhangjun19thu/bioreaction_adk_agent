#!/usr/bin/env python3
"""
生物反应科研Agent系统测试脚本

这个脚本用于测试系统的各个组件和功能。
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bioreaction_adk_agent.agent import query_agent, root_agent
from bioreaction_adk_agent.tools.database_loader import DB
from bioreaction_adk_agent.config import validate_config, QUERY_CONFIG, ANALYSIS_CONFIG

def test_config():
    """测试配置系统"""
    print("=== 测试配置系统 ===")
    
    # 验证配置
    config_errors = validate_config()
    if config_errors:
        print("❌ 配置错误:")
        for error in config_errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ 配置验证通过")
    
    # 测试配置值
    print(f"✅ 查询配置: 最大结果数={QUERY_CONFIG['max_results']}")
    print(f"✅ 分析配置: 相关性阈值={ANALYSIS_CONFIG['correlation_threshold']}")
    
    return True

def test_database_loading():
    """测试数据库加载"""
    print("\n=== 测试数据库加载 ===")
    
    if not DB:
        print("❌ 数据库未加载")
        return False
    
    print(f"✅ 数据库已加载，包含 {len(DB)} 个表")
    
    for table_name, df in DB.items():
        print(f"  - {table_name}: {len(df)} 行")
    
    return True

def test_basic_queries():
    """测试基本查询功能"""
    print("\n=== 测试基本查询功能 ===")
    
    # 测试数据库统计
    try:
        from bioreaction_adk_agent.tools.database_query_tools import get_database_statistics
        stats = get_database_statistics()
        print("✅ 数据库统计功能正常")
        print(f"统计信息长度: {len(stats)} 字符")
    except Exception as e:
        print(f"❌ 数据库统计功能失败: {e}")
        return False
    
    # 测试智能搜索
    try:
        from bioreaction_adk_agent.tools.database_query_tools import smart_search_reactions
        search_result = smart_search_reactions(
            search_query="enzyme",
            search_fields=["enzyme_name"],
            max_results=3
        )
        print("✅ 智能搜索功能正常")
        print(f"搜索结果长度: {len(search_result)} 字符")
    except Exception as e:
        print(f"❌ 智能搜索功能失败: {e}")
        return False
    
    return True

async def test_agent_queries():
    """测试Agent查询功能"""
    print("\n=== 测试Agent查询功能 ===")
    
    test_queries = [
        "请告诉我数据库中有哪些表？",
        "查找与酶相关的反应，最多返回3个结果",
        "分析反应趋势，酶名为ADK，物种为E. coli"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i}: {query} ---")
        try:
            response = await asyncio.wait_for(query_agent(query), timeout=30)
            print(f"✅ 查询成功，响应长度: {len(response)} 字符")
            print(f"响应预览: {response[:200]}...")
        except asyncio.TimeoutError:
            print("❌ 查询超时，可能是Agent未响应")
            return False
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return False
    
    return True

def test_advanced_tools():
    """测试高级分析工具"""
    print("\n=== 测试高级分析工具 ===")
    
    # 测试趋势分析
    try:
        from bioreaction_adk_agent.tools.advanced_tools import analyze_reaction_trends
        trend_result = analyze_reaction_trends(
            enzyme_name="ADK",
            organism="E. coli",
            metric="conversion_rate",
            min_data_points=3
        )
        print("✅ 趋势分析功能正常")
        print(f"分析结果长度: {len(trend_result)} 字符")
    except Exception as e:
        print(f"❌ 趋势分析功能失败: {e}")
        return False
    
    # 测试模式分析
    try:
        from bioreaction_adk_agent.tools.database_query_tools import analyze_reaction_patterns
        pattern_result = analyze_reaction_patterns(
            pattern_type="enzyme_frequency",
            min_occurrences=2
        )
        print("✅ 模式分析功能正常")
        print(f"分析结果长度: {len(pattern_result)} 字符")
    except Exception as e:
        print(f"❌ 模式分析功能失败: {e}")
        return False
    
    return True

def test_deep_research_tools():
    """测试深度研究工具"""
    print("\n=== 测试深度研究工具 ===")
    
    # 测试文献上下文获取
    try:
        from bioreaction_adk_agent.tools.deep_research_tools import get_literature_context
        context_result = get_literature_context("test_literature_id")
        print("✅ 文献上下文功能正常")
        print(f"上下文信息长度: {len(str(context_result))} 字符")
    except Exception as e:
        print(f"❌ 文献上下文功能失败: {e}")
        return False
    
    return True

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试无效查询
    try:
        from bioreaction_adk_agent.tools.database_query_tools import find_reactions_by_enzyme
        result = find_reactions_by_enzyme(
            enzyme_name="",
            organism="",
            max_results=5
        )
        if "请提供酶名称或物种信息" in result:
            print("✅ 错误处理正常 - 空参数检测")
        else:
            print("❌ 错误处理异常 - 未检测到空参数")
            return False
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False
    
    return True

async def run_all_tests():
    """运行所有测试"""
    print("🚀 开始生物反应科研Agent系统测试\n")
    
    tests = [
        ("配置系统", test_config),
        ("数据库加载", test_database_loading),
        ("基本查询", test_basic_queries),
        ("Agent查询", test_agent_queries),
        ("高级工具", test_advanced_tools),
        ("深度研究", test_deep_research_tools),
        ("错误处理", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统运行正常。")
        return True
    else:
        print("⚠️  部分测试失败，请检查系统配置。")
        return False
    
# async def main():
#     print("发送测试请求...")
#     resp = await query_agent("你好")
#     print("Agent响应：", resp)

if __name__ == "__main__":
    asyncio.run(run_all_tests())
