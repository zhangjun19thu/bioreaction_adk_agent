from google.adk.agents import LlmAgent
from .database_query_tools import (
    get_reaction_summary_tool,
    find_reactions_by_enzyme_tool,
    find_inhibition_data_tool,
    find_reactions_by_organism_tool,
    find_reactions_by_condition_tool,
    find_reactions_with_pdb_id_tool,
    find_top_reactions_by_performance_tool,
    find_kinetic_parameters_tool,
    find_conditions_by_enzyme_tool,
    find_enzymes_by_participant_tool,
    smart_search_reactions_tool,
    get_database_statistics_tool,
    find_similar_reactions_tool,
    analyze_reaction_patterns_tool
)
from google.adk.tools.agent_tool import AgentTool
# 导入配置
from ..config import AGENT_CONFIG, validate_config

# 强prompt，指导大模型如何意图映射和参数推理
universal_query_agent_prompt = """
你是一个生物化学数据库智能检索Agent。你的任务是：
1. 解析用户输入，推断其真实意图（如：按酶、物种、底物、产物、抑制剂、实验条件、PDB、性能、动力学参数、模式分析、统计等）。
2. 自动将自然语言问题映射为数据库字段和参数，字段包括：
   - reaction_equation, reaction_type_reversible, notes, enzyme_name, enzyme_synonyms, gene_name, organism, ec_number, participant_name, role, literature_id, reaction_id 等。
3. 你必须严格校验参数与数据库字段一致性，缺失时自动补全为"全部"，模糊意图时优先召回更多结果。
4. 你只能调用下方提供的数据库检索工具（FunctionTool），并根据推理结果选择最合适的工具和参数。
5. 工具调用结果要结构化、准确、可追溯。
6. 如参数不合法或无结果，需给出详细报错或友好提示。
7. 你不能直接返回原始用户输入，必须经过意图映射和参数推理。
"""

universal_query_agent = LlmAgent(
    name="universal_query_agent",
    model=AGENT_CONFIG["model"],
    instruction=universal_query_agent_prompt,
    tools=[
        get_reaction_summary_tool,
        find_reactions_by_enzyme_tool,
        find_inhibition_data_tool,
        find_reactions_by_organism_tool,
        find_reactions_by_condition_tool,
        find_reactions_with_pdb_id_tool,
        find_top_reactions_by_performance_tool,
        find_kinetic_parameters_tool,
        find_conditions_by_enzyme_tool,
        find_enzymes_by_participant_tool,
        smart_search_reactions_tool,
        get_database_statistics_tool,
        find_similar_reactions_tool,
        analyze_reaction_patterns_tool
    ]
)

# 保留底层函数实现，供LlmAgent工具调用
from .database_query_tools import *

# 导出LlmAgent工具
universal_query_agent_tool = AgentTool(universal_query_agent)