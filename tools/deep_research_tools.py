from google.adk.tools import FunctionTool
import os
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
from typing import List, Dict, Optional
import pandas as pd
from .database_loader import DB
from ..config import METADATA_BASE_DIR, get_metadata_path, AGENT_CONFIG
import concurrent.futures

# from utils.text_parser import preprocess_text_for_llm

try:
    from utils.text_parser import preprocess_text_for_llm
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.text_parser import preprocess_text_for_llm

# 使用配置中的元数据目录
# METADATA_BASE_DIR 已在 config.py 中定义

# 定义专门的文献分析Agent
literature_analysis_agent = Agent(
    name="advanced_literature_analyzer",
    model=AGENT_CONFIG["model"],
    instruction="""
    你是生物化学文献深度分析专家。你的任务是：
    
    1. **精确信息提取**：从文献中提取用户需要的具体信息
    2. **上下文理解**：结合生物化学背景知识理解实验内容
    3. **关键发现识别**：识别文献中的关键发现和重要结论
    4. **方法学分析**：分析实验方法的优缺点和创新点
    5. **结果解释**：解释实验结果的意义和潜在应用
    
    回答要求：
    - 基于文献内容，不添加外部知识
    - 提供具体的数据和事实
    - 使用清晰的科学术语
    - 如果信息不足，明确说明
    """,
    description="专门用于生物化学文献深度分析和信息提取的Agent"
)

# 定义文献对比Agent
literature_comparison_agent = Agent(
    name="literature_comparison_specialist",
    model=AGENT_CONFIG["model"],
    instruction="""
    你是文献对比分析专家。你的任务是：
    
    1. **方法学对比**：比较不同文献的实验方法
    2. **结果对比**：对比实验结果和性能指标
    3. **结论对比**：分析不同文献的结论和观点
    4. **创新点识别**：识别各文献的创新点和贡献
    5. **综合评估**：提供综合性的对比分析
    
    对比要求：
    - 客观公正，避免偏见
    - 突出差异和相似点
    - 提供数据支持
    - 给出实用建议
    """,
    description="专门用于多文献对比分析的Agent"
)

# 包装为工具
literature_analysis_tool = AgentTool(agent=literature_analysis_agent)
literature_comparison_tool = AgentTool(agent=literature_comparison_agent)

async def get_summary_from_literature(
    literature_id: str, 
    question: str,
    analysis_type: str
) -> dict:
    """
    读取并分析给定文献ID的元数据/摘要文件，以回答用户的问题或提供摘要。
    
    :param literature_id: 文献ID
    :param question: 用户问题
    :param analysis_type: 分析类型 ('general', 'methodology', ...)
    """
    print(f"[深度研究工具]: 收到对 {literature_id} 的请求，问题是: '{question}'，分析类型: {analysis_type}")
    
    # 使用配置中的路径获取函数
    metadata_path = get_metadata_path(literature_id)
    
    if not metadata_path.exists():
        return {"status": "error", "error_message": f"错误：无法为文献ID '{literature_id}' 找到元数据文件。"}
        
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 新增：预处理文献内容
        if preprocess_text_for_llm:
            content = preprocess_text_for_llm(content)
    except Exception as e:
        return {"status": "error", "error_message": f"错误：读取文献 '{literature_id}' 的元数据文件失败: {e}"}
    
    # 根据分析类型调整提示词
    if analysis_type == "methodology":
        enhanced_question = f"请重点分析这篇文献的实验方法，包括：{question}"
    elif analysis_type == "results":
        enhanced_question = f"请重点分析这篇文献的实验结果，包括：{question}"
    elif analysis_type == "conclusions":
        enhanced_question = f"请重点分析这篇文献的结论和意义，包括：{question}"
    elif analysis_type == "detailed":
        enhanced_question = f"请对这篇文献进行详细分析，包括方法、结果、结论等各个方面：{question}"
    else:
        enhanced_question = question
    
    # 构建增强的提示词
    prompt = f"""
    请仅根据以下提供的文本内容，为接下来的问题提供一个详细、准确的答案。

    ---文献内容开始---
    {content} 
    ---文献内容结束---

    问题: {enhanced_question}
    
    要求：
    1. 基于文献内容回答，不要添加外部知识
    2. 提供具体的数据、方法和结果
    3. 使用清晰的科学术语
    4. 如果信息不足，请明确说明
    """
    
    # 使用专门的文献分析Agent
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=AGENT_CONFIG["app_name"], 
        user_id="user1234", 
        session_id="session1"
    )
    runner = Runner(
        agent=literature_analysis_agent, 
        app_name=AGENT_CONFIG["app_name"], 
        session_service=session_service
    )
    user_content = types.Content(role='user', parts=[types.Part(text=prompt)])
    events = runner.run_async(user_id="user1234", session_id="session1", new_message=user_content)
    
    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            return {
                "status": "success", 
                "summary": f"## 文献 {literature_id} 分析结果\n\n{final_response}",
                "literature_id": literature_id,
                "analysis_type": analysis_type
            }
    
    return {"status": "error", "error_message": "Agent未返回最终响应。"}

async def analyze_multiple_literature(
    literature_ids: List[str],
    comparison_question: str,
    comparison_focus: str
) -> dict:
    """
    分析多篇文献并进行对比。
    
    :param literature_ids: 文献ID列表
    :param comparison_question: 对比问题
    :param comparison_focus: 对比重点
    """
    if len(literature_ids) < 2:
        return {"status": "error", "error_message": "至少需要2篇文献进行对比分析。"}
    
    print(f"[多文献分析工具]: 收到对 {len(literature_ids)} 篇文献的对比请求")
    
    # 读取所有文献内容
    literature_contents = {}
    for lit_id in literature_ids:
        metadata_path = get_metadata_path(lit_id)
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # 新增：预处理文献内容
                if preprocess_text_for_llm:
                    content = preprocess_text_for_llm(content)
                literature_contents[lit_id] = content
            except Exception as e:
                print(f"警告：无法读取文献 {lit_id}: {e}")
        else:
            print(f"警告：文献 {lit_id} 的元数据文件不存在")
    
    if len(literature_contents) < 2:
        return {"status": "error", "error_message": "可用的文献数量不足，无法进行对比分析。"}
    
    # 构建对比提示词
    content_text = ""
    for lit_id, content in literature_contents.items():
        content_text += f"\n---文献 {lit_id} 内容---\n{content}\n"
    
    prompt = f"""
    请对比分析以下多篇文献，回答用户的具体问题。

    {content_text}

    对比问题: {comparison_question}
    对比重点: {comparison_focus}
    
    要求：
    1. 客观公正地对比各文献
    2. 突出差异和相似点
    3. 提供具体的数据支持
    4. 给出实用的见解和建议
    5. 使用表格或列表形式组织信息
    """
    
    # 使用文献对比Agent
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=AGENT_CONFIG["app_name"], 
        user_id="user1234", 
        session_id="session2"
    )
    runner = Runner(
        agent=literature_comparison_agent, 
        app_name=AGENT_CONFIG["app_name"], 
        session_service=session_service
    )
    user_content = types.Content(role='user', parts=[types.Part(text=prompt)])
    events = runner.run_async(user_id="user1234", session_id="session2", new_message=user_content)
    
    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            return {
                "status": "success", 
                "comparison": f"## 多文献对比分析结果\n\n{final_response}",
                "literature_ids": literature_ids,
                "comparison_focus": comparison_focus
            }
    
    return {"status": "error", "error_message": "Agent未返回最终响应。"}

def find_related_literature(
    target_literature_id: str,
    similarity_criteria: str,
    max_results: int
) -> dict:
    """
    基于特定标准查找相关文献。
    
    :param target_literature_id: 文献ID
    :param similarity_criteria: 相似性标准
    :param max_results: 最大结果数量
    """
    if not DB: return {"status": "error", "error_message": "数据库未加载。"}
    
    # 获取目标文献信息
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    
    if enzymes_df.empty or core_df.empty:
        return {"status": "error", "error_message": "核心数据表未加载。"}
    
    target_enzyme = enzymes_df[enzymes_df['literature_id'] == target_literature_id]
    target_reaction = core_df[core_df['literature_id'] == target_literature_id]
    
    if target_enzyme.empty and target_reaction.empty:
        return {"status": "error", "error_message": f"未找到文献 {target_literature_id} 的记录。"}
    
    related_literature = []
    
    if similarity_criteria == "enzyme" and not target_enzyme.empty:
        # 基于酶名称查找相关文献
        enzyme_name = target_enzyme.iloc[0]['enzyme_name']
        related = enzymes_df[
            (enzymes_df['enzyme_name'].str.contains(enzyme_name.split('_')[0], case=False, na=False)) &
            (enzymes_df['literature_id'] != target_literature_id)
        ]
        related_literature = related['literature_id'].unique()[:max_results].tolist()
    
    elif similarity_criteria == "organism" and not target_enzyme.empty:
        # 基于物种查找相关文献
        organism = target_enzyme.iloc[0]['organism']
        related = enzymes_df[
            (enzymes_df['organism'].str.contains(organism.split()[0], case=False, na=False)) &
            (enzymes_df['literature_id'] != target_literature_id)
        ]
        related_literature = related['literature_id'].unique()[:max_results].tolist()
    
    elif similarity_criteria == "ec_number" and not target_enzyme.empty:
        # 基于EC号查找相关文献
        ec_number = str(target_enzyme.iloc[0].get('ec_number', ''))
        if ec_number:
            related = enzymes_df[
                (enzymes_df['ec_number'].astype(str).str.contains(ec_number.split('.')[0], case=False, na=False)) &
                (enzymes_df['literature_id'] != target_literature_id)
            ]
            related_literature = related['literature_id'].unique()[:max_results].tolist()
    
    return {
        "status": "success",
        "target_literature": target_literature_id,
        "similarity_criteria": similarity_criteria,
        "related_literature": related_literature,
        "count": len(related_literature)
    }

# # --- 异步转同步包装 ---
# def get_summary_from_literature_sync(*args, **kwargs):
#     import asyncio
#     return asyncio.run(get_summary_from_literature(*args, **kwargs))

# def analyze_multiple_literature_sync(*args, **kwargs):
#     import asyncio
#     return asyncio.run(analyze_multiple_literature(*args, **kwargs))

# --- 将函数包装成 FunctionTool 实例 ---
def get_summary_from_literature_sync(
    literature_id: str,
    question: str,
    analysis_type: str
) -> dict:
    import asyncio
    def runner():
        return asyncio.run(get_summary_from_literature(
            literature_id, question, analysis_type
        ))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(runner)
        result = future.result()
        # 只返回主要内容
        # ADK/FunctionTool/前端通常只会显示字符串或简单结构，如果你返回的是 dict，而不是直接的字符串，前端可能不会自动渲染出来。
        # 如果 FunctionTool 期望返回字符串，但你返回了字典，ADK主Agent不会自动把字典内容渲染到前端。
        if isinstance(result, dict) and 'comparison' in result:
            return result['comparison']
        elif isinstance(result, dict) and 'summary' in result:
            return result['summary']
        else:
            return str(result)

def analyze_multiple_literature_sync(
    literature_ids: List[str],
    comparison_question: str,
    comparison_focus: str
) -> dict:
    import asyncio
    def runner():
        return asyncio.run(analyze_multiple_literature(
            literature_ids, comparison_question, comparison_focus
        ))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(runner)
        result = future.result()
        if isinstance(result, dict) and 'summary' in result:
            return result['summary']
        else:
            return str(result)

get_summary_from_literature_tool = FunctionTool(func=get_summary_from_literature_sync)
analyze_multiple_literature_tool = FunctionTool(func=analyze_multiple_literature_sync)
find_related_literature_tool = FunctionTool(func=find_related_literature)