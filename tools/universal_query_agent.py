from google.adk.tools import FunctionTool
from .database_query_tools import (
    find_reactions_by_enzyme,
    find_enzymes_by_participant,
    smart_search_reactions,
    find_reactions_by_organism,
    get_reaction_summary,
    get_database_statistics,
)
import re

def universal_query_agent(user_query: str, max_results: int) -> str:
    """
    统一智能数据库查询入口。
    :param user_query: 用户自然语言问题
    :param max_results: 返回最大结果数
    """
    # 1. 酶名意图
    enzyme_match = re.search(r'(?:酶|enzyme)[^\u4e00-\u9fa5a-zA-Z0-9]*([\u4e00-\u9fa5a-zA-Z0-9\-\'\s]+)', user_query)
    if enzyme_match:
        enzyme_name = enzyme_match.group(1).strip()
        organism_match = re.search(r'(?:物种|organism|来源)[^\u4e00-\u9fa5a-zA-Z0-9]*([\u4e00-\u9fa5a-zA-Z0-9\-\'\s]+)', user_query)
        organism = organism_match.group(1).strip() if organism_match else None
        return find_reactions_by_enzyme(enzyme_name=enzyme_name, organism=organism, max_results=max_results)
    # 2. 反应式意图
    if '->' in user_query or '→' in user_query:
        return smart_search_reactions(search_query=user_query, search_fields=['reaction_equation'], max_results=max_results)
    # 3. 底物/产物意图
    if '底物' in user_query or '产物' in user_query or 'substrate' in user_query.lower() or 'product' in user_query.lower():
        # 尝试提取分子名
        mol_match = re.search(r'(?:底物|产物|substrate|product)[^\u4e00-\u9fa5a-zA-Z0-9]*([\u4e00-\u9fa5a-zA-Z0-9\-\'\s]+)', user_query)
        participant_name = mol_match.group(1).strip() if mol_match else user_query
        return find_enzymes_by_participant(participant_name=participant_name, max_results=max_results)
    # 4. 物种+EC号意图
    if '物种' in user_query or 'organism' in user_query.lower() or 'EC' in user_query or 'ec_number' in user_query:
        org_match = re.search(r'(?:物种|organism)[^\u4e00-\u9fa5a-zA-Z0-9]*([\u4e00-\u9fa5a-zA-Z0-9\-\'\s]+)', user_query)
        ec_match = re.search(r'(?:EC|ec_number)[^\d]*(\d+\.\d+\.\d+\.\d+)', user_query)
        organism = org_match.group(1).strip() if org_match else None
        ec_number = ec_match.group(1).strip() if ec_match else None
        return find_reactions_by_organism(organism=organism, ec_number=ec_number, max_results=max_results)
    # 5. 反应摘要
    if '摘要' in user_query or 'summary' in user_query.lower():
        # 尝试提取文献和反应编号
        lit_match = re.search(r'(PMID\d+)', user_query)
        rid_match = re.search(r'(reaction_\d+)', user_query)
        if lit_match and rid_match:
            return get_reaction_summary(literature_id=lit_match.group(1), reaction_id=rid_match.group(1))
    # 6. fallback: 智能模糊检索
    return smart_search_reactions(search_query=user_query, search_fields=[], max_results=max_results)

universal_query_agent_tool = FunctionTool(
    func=universal_query_agent,
    name="universal_query_agent",
    description="统一数据库智能查询入口，自动解析用户意图并调度最优检索函数。参数：user_query: str, max_results: int（必填）"
) 