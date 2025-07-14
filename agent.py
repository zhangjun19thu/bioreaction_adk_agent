from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
import os
import asyncio
from typing import List, Dict, Any

# --- 从工具文件中导入 FunctionTool *实例* ---
from .tools.database_query_tools import (
    get_reaction_summary_tool,
    # find_reactions_by_enzyme_tool,  # 移除底层函数
    # find_inhibition_data_tool,
    # find_reactions_by_organism_tool,
    # find_reactions_by_condition_tool,
    # find_reactions_with_pdb_id_tool,
    # find_top_reactions_by_performance_tool,
    # find_kinetic_parameters_tool,
    # find_conditions_by_enzyme_tool,
    # find_enzymes_by_participant_tool,
    # 新增智能查询工具
    smart_search_reactions_tool,
    get_database_statistics_tool,
    find_similar_reactions_tool,
    analyze_reaction_patterns_tool
)
from .tools.deep_research_tools import (
    get_summary_from_literature_tool,
    analyze_multiple_literature_tool,
    get_literature_context_tool,
    find_related_literature_tool
)
from .tools.advanced_tools import (
    analyze_reaction_trends_tool,
    compare_reactions_tool,
    suggest_optimization_tool,
    literature_comparison_tool
)
from .tools.universal_query_agent import universal_query_agent_tool

# 导入配置
from .config import AGENT_CONFIG, validate_config

# 建议通过环境变量设置您的API密钥
# os.environ['GEMINI_API_KEY'] = "YOUR_API_KEY" 

# --- 主Agent核心指令 (Prompt) ---
main_instructions = """
你是Biochemist-GPT，一个专注于生物化学反应领域的专家级AI研究助手。
你的核心使命是高效地运用你所拥有的工具，为用户提供精准、数据驱动的答案。

## 数据库字段说明（请严格参考字段名进行意图映射和参数生成）
- reaction_equation: 反应方程式（如"A + B -> C + D"）
- reaction_type_reversible: 反应是否可逆（如"Yes"/"No"/"Not specified"）
- notes: 反应备注
- enzyme_name: 酶名称（如"Ornithine transcarbamoylase"）
- enzyme_synonyms: 酶同义词（如"OTC|Ornithine carbamoyltransferase"）
- gene_name: 基因名称
- organism: 物种（如"Escherichia coli"）
- ec_number: EC号（如"2.1.1.1"）
- participant_name: 参与分子（底物/产物/抑制剂等）
- role: 分子角色（如"substrate"/"product"/"inhibitor"）
- literature_id: 文献编号（如PMID）
- reaction_id: 反应编号
- 其他字段请参考get_database_statistics工具输出

## 智能意图识别与参数生成
- 你必须根据用户输入内容，自动推断最合适的字段和参数。
- 你只能调用universal_query_agent工具（见下），并严格传递必填参数。
- 你不能直接调用底层数据库函数。
- 你必须保证所有function call参数都为必填项（无缺省值）。
- 你应优先用reaction_equation、enzyme_name、participant_name、organism等字段进行检索。
- 如用户未指定organism等参数，自动补全为"全部"。
- 如用户输入模糊或多意图，优先召回更多结果。

## 工具说明
- universal_query_agent：统一数据库查询入口，参数：user_query: str, max_results: int
- get_reaction_summary_tool：反应摘要（需literature_id和reaction_id）
- 其他文献/高级分析工具同上

## 响应标准
- 数据准确：所有信息必须来自数据库或文献
- 逻辑清晰：按重要性组织信息
- 实用性强：提供可操作的见解和建议
- 诚实透明：明确说明数据来源和局限性
"""

# --- 创建专门的子Agent ---

# 数据分析Agent
data_analysis_agent = Agent(
    name="biochemical_data_analyzer",
    model=AGENT_CONFIG["model"],
    instruction="""
    你是生物化学数据分析专家。你的任务是：
    1. 分析反应数据中的模式和趋势
    2. 识别性能指标的关键影响因素
    3. 提供数据驱动的优化建议
    4. 生成清晰的数据可视化描述
    
    始终基于实际数据进行分析，避免推测。
    """,
    description="专门用于生物化学反应数据分析和趋势识别的Agent"
)

# 文献研究Agent  
literature_research_agent = Agent(
    name="literature_research_specialist", 
    model=AGENT_CONFIG["model"],
    instruction="""
    你是生物化学文献研究专家。你的任务是：
    1. 深入分析文献内容，提取关键信息
    2. 回答用户的具体问题
    3. 识别文献间的关联和差异
    4. 提供实验方法和结果的详细解释
    
    只基于提供的文献内容回答，不添加外部知识。
    """,
    description="专门用于文献深度分析和研究的Agent"
)

# 优化建议Agent
optimization_agent = Agent(
    name="biochemical_optimization_advisor",
    model=AGENT_CONFIG["model"], 
    instruction="""
    你是生物化学反应优化专家。你的任务是：
    1. 基于数据分析结果提供优化建议
    2. 识别影响反应性能的关键因素
    3. 建议实验条件改进方案
    4. 预测优化后的预期效果
    
    建议必须基于实际数据，避免空泛的表述。
    """,
    description="专门用于提供生物化学反应优化建议的Agent"
)

# --- 创建主Agent实例 ---
root_agent = Agent(
    name="bioreaction_deep_research_agent",
    model=AGENT_CONFIG["model"],
    instruction=main_instructions,
    tools=[
        # 统一数据库查询工具
        universal_query_agent_tool,
        # 反应摘要
        get_reaction_summary_tool,
        # 智能查询工具（如需）
        smart_search_reactions_tool,
        get_database_statistics_tool,
        find_similar_reactions_tool,
        analyze_reaction_patterns_tool,
        # 深度研究工具
        get_summary_from_literature_tool,
        analyze_multiple_literature_tool,
        get_literature_context_tool,
        find_related_literature_tool,
        # 高级分析工具
        analyze_reaction_trends_tool,
        compare_reactions_tool,
        suggest_optimization_tool,
        literature_comparison_tool,
        # Agent工具
        AgentTool(agent=data_analysis_agent),
        AgentTool(agent=literature_research_agent),
        AgentTool(agent=optimization_agent),
    ]
)

# --- 创建Runner实例 ---
session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name=AGENT_CONFIG["app_name"],
    session_service=session_service
)

# --- 便捷函数 ---
async def query_agent(user_query: str, user_id: str = "default_user") -> str:
    """
    便捷函数：向agent发送查询并获取响应
    """
    from google.genai import types
    
    # 验证配置
    config_errors = validate_config()
    if config_errors:
        return f"配置错误，无法启动Agent:\n" + "\n".join(config_errors)
    
    session = await session_service.create_session(
        app_name=AGENT_CONFIG["app_name"],
        user_id=user_id,
        session_id=f"session_{user_id}"
    )
    
    user_content = types.Content(role='user', parts=[types.Part(text=user_query)])
    events = runner.run_async(user_id=user_id, session_id=f"session_{user_id}", new_message=user_content)
    
    print(f"[DEBUG] 向Agent发送: {user_query}")
    async for event in events:
        print(f"[DEBUG] event: {event}")
        if event.is_final_response():
            return event.content.parts[0].text
    return "未能获取有效响应"

# --- 导出主要组件 ---
__all__ = ['root_agent', 'runner', 'query_agent', 'data_analysis_agent', 'literature_research_agent', 'optimization_agent']