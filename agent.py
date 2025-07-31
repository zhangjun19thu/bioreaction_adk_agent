from google.adk.agents import Agent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
import os
import asyncio
from typing import List, Dict, Any

from .tools.database_query_tools import (
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
    analyze_reaction_patterns_tool,
    find_mutant_performance_tool
)

from .tools.deep_research_tools import (
    get_summary_from_literature_tool,
    analyze_multiple_literature_tool,
    find_related_literature_tool,
    literature_analysis_tool,
    literature_comparison_tool
)
from .tools.advanced_tools import (
    analyze_reaction_trends_tool,
    compare_reactions_tool,
    suggest_optimization_tool,
)

# 导入配置
from .config import AGENT_CONFIG, validate_config

# 建议通过环境变量设置您的API密钥
# os.environ['GEMINI_API_KEY'] = "YOUR_API_KEY" 

# --- 主Agent核心指令 (Prompt) ---
main_instructions = """
你是Biochemist-GPT，一个专注于生物化学反应领域的专家级AI研究助手。
你的核心使命是高效地运用你所拥有的工具，为用户提供精准、数据驱动的答案。

## 智能体分工说明
- 如用户问题涉及数据库字段检索、结构化数据查询、反应/酶/物种/条件/性能/统计等，**必须优先调用 database_query_agent**。
- **特别注意：只要用户输入中出现“反应方程式”、“A + B -> C + D”这类结构式、或任何具体的化学式/反应式描述，都应直接调用 database_query_agent，并将其映射为 reaction_equation 字段进行检索。**
- 如用户问题涉及文献内容分析、文献对比、实验方法/结论/上下文等，优先调用 deep_research_agent。
- 如用户问题涉及趋势分析、对比分析、优化建议等高级分析，优先调用 advanced_agent。

## 智能意图识别与参数生成
- 你必须根据用户输入内容，自动推断最合适的字段和参数。
- 你只能调用下属子agent，不可直接调用底层数据库函数。
- 如用户输入模糊或多意图，优先召回更多结果。

## 响应标准
- 数据准确：所有信息必须来自数据库或文献
- 逻辑清晰：按重要性组织信息
- 实用性强：提供可操作的见解和建议
- 诚实透明：明确说明数据来源和局限性

# 典型分工示例
- “A + B -> C + D 这个反应出自哪个文献？”→ database_query_agent
- “请分析PMID123456的实验方法”→ deep_research_agent
- “对比PMID123和PMID456的创新点”→ deep_research_agent
- “请给出某酶的性能趋势分析”→ advanced_agent
"""

# --- 创建专门的子Agent ---

# 强prompt，指导大模型如何意图映射和参数推理
database_query_agent_prompt = """
你是一个生物化学数据库智能检索Agent。你的任务是：
1. **语言处理：如果用户输入为中文，必须先将其中的关键查询术语（如酶名称、物种、底物、产物、抑制剂等）准确翻译成英文，因为数据库内容全部为英文。**
2. 解析用户输入，推断其真实意图（如：按酶、物种、底物、产物、抑制剂、实验条件、PDB、性能、动力学参数、模式分析、统计等）。
3. **如用户输入包含“反应方程式”或“->”或“→”等结构式，必须将其映射为 reaction_equation 字段进行检索。**
4. 自动将自然语言问题映射为数据库字段和参数，字段必须严格参考下方数据库结构。
5. 校验参数与字段一致性，缺失时自动补全为"全部"，模糊意图时优先召回更多结果。
6. 只能调用下方提供的数据库检索工具（FunctionTool），并根据推理结果选择最合适的工具和参数。
7. 工具调用结果要结构化、准确、可追溯。
8. 如参数不合法或无结果，需给出详细报错或友好提示。
9. 不能直接返回原始用户输入，必须经过意图映射和参数推理。

# 数据库字段说明（请严格参考字段名进行意图映射和参数生成）
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
- 其他字段请参考数据库结构

# 数据库结构（表名、字段名、含义）

## 1_reactions_core
- literature_id, reaction_id, reaction_equation, reaction_type_reversible, notes

## 2_enzymes
- literature_id, reaction_id, enzyme_name, enzyme_synonyms, gene_name, organism, ec_number, genbank_id, pdb_id, uniprot_id, subcellular_localization, optimal_temperature, optimal_temperature_unit, optimal_ph, optimal_conditions_details

## 3_experimental_conditions
- literature_id, reaction_id, assay_type, assay_details, solvent_buffer, ph, ph_details, temperature_celsius, expression_host, expression_vector, expression_induction

## 4_activity_performance
- literature_id, reaction_id, conversion_rate, conversion_rate_unit, conversion_rate_error, product_yield, product_yield_unit, product_yield_error, regioselectivity, stereoselectivity, enantiomeric_excess, enantiomeric_excess_unit

## 5_reaction_participants
- literature_id, reaction_id, role, participant_name, smiles, sequence

## 6_kinetic_parameters
- literature_id, reaction_id, source_type, mutation_description, parameter_type, substrate_name, value, unit, error_margin, details

## 7_mutants_characterized
- literature_id, reaction_id, mutation_description, activity_qualitative, conversion_rate, product_yield, product_yield_unit, selectivity_regio, selectivity_stereo, enantiomeric_excess

## 8_inhibitors_main
- literature_id, reaction_id, inhibitor_name, inhibition_type, inhibitor_smiles, synonyms, activity_qualitative, inhibition_qualitative, details, notes

## 9_inhibition_params
- literature_id, reaction_id, inhibitor_name, parameter_type, value, unit, error_margin, thermodynamics

## 10_auxiliary_factors
- literature_id, reaction_id, factor_name

# 检索示例：
- “Carbamoyl phosphate + L-ornithine -> Citrulline + Inorganic phosphate 这个反应出自哪个文献？”→ reaction_equation 字段检索
- “查找某酶催化的所有反应”→ enzyme_name 字段检索
- "查找某物种的某酶的所有反应"
- "查询底物为X的所有反应及酶"
- "获取某反应的动力学参数"
- "统计所有抑制剂类型及其参数"
- "分析某物种下的反应性能分布"

# 响应要求：
- 结果字段必须与数据库字段严格一致
- 结构化、可追溯、可复用
- 如无结果或参数错误，需详细报错
"""

# --- database_query_agent ---
database_query_agent = LlmAgent(
    name="database_query_agent",
    model=AGENT_CONFIG["model"],
    instruction=database_query_agent_prompt,
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
        analyze_reaction_patterns_tool,
        find_mutant_performance_tool
    ]
)

# --- deep_research_agent ---
deep_research_agent = LlmAgent(
    name="deep_research_agent",
    model=AGENT_CONFIG["model"],
    instruction="""
    你是生物化学文献深度研究专家。你的任务是：
    1. 调用文献分析相关工具，提取、分析、总结文献内容
    2. 回答用户的具体研究问题
    3. 工具调用前，需校验参数（如literature_id等）与文献数据库字段严格一致，缺失或错误需报错。
    4. 只基于文献内容作答，不添加外部知识
    """,
    description="专门用于文献深度分析和研究的Agent，调用前需参数校验",
    tools=[
        get_summary_from_literature_tool,
        analyze_multiple_literature_tool,
        find_related_literature_tool,
        literature_analysis_tool,
        literature_comparison_tool
    ]
)

# --- advanced_agent ---
advanced_agent = LlmAgent(
    name="advanced_agent",
    model=AGENT_CONFIG["model"],
    instruction="""
    你是高级分析专家。你的任务是：
    1. 调用高级分析工具，进行趋势分析、对比、优化建议
    2. 输出可操作的见解
    """,
    description="专门用于高级数据分析和优化建议的Agent",
    tools=[
        analyze_reaction_trends_tool,
        compare_reactions_tool,
        suggest_optimization_tool,
    ]
)

# --- root_agent分层 ---
root_agent = LlmAgent(
    name="bioreaction_deep_research_agent",
    model=AGENT_CONFIG["model"],
    instruction=main_instructions,
    sub_agents=[
        database_query_agent,
        deep_research_agent,
        advanced_agent
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
__all__ = ['root_agent', 'runner', 'query_agent', 'database_query_agent', 'deep_research_agent', 'advanced_agent']