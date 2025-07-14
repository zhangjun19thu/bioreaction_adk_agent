from google.adk.tools import FunctionTool
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from .database_loader import DB
from ..config import QUERY_CONFIG, ANALYSIS_CONFIG
import re
import logging

def normalize_enzyme_name(name: str) -> str:
    """
    归一化酶名：去除空格、特殊字符、转小写。
    """
    if not isinstance(name, str):
        return ''
    # 去除空格和常见特殊字符，仅保留字母数字
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()

def get_reaction_summary(
    literature_id: str,
    reaction_id: str
) -> str:
    """
    获取特定反应的完整摘要信息。
    
    :param literature_id: str
    :param reaction_id: str
    """
    if not DB: return "数据库未加载。"
    
    # 获取各表数据
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    activity_df = DB.get('4_activity_performance', pd.DataFrame())
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    participants_df = DB.get('5_reaction_participants', pd.DataFrame())
    
    # 筛选目标反应
    reaction_core = core_df[(core_df['literature_id'] == literature_id) & (core_df['reaction_id'] == reaction_id)]
    reaction_enzyme = enzymes_df[(enzymes_df['literature_id'] == literature_id) & (enzymes_df['reaction_id'] == reaction_id)]
    reaction_activity = activity_df[(activity_df['literature_id'] == literature_id) & (activity_df['reaction_id'] == reaction_id)]
    reaction_conditions = conditions_df[(conditions_df['literature_id'] == literature_id) & (conditions_df['reaction_id'] == reaction_id)]
    reaction_participants = participants_df[(participants_df['literature_id'] == literature_id) & (participants_df['reaction_id'] == reaction_id)]
    
    if reaction_core.empty:
        return f"未找到反应 {literature_id}:{reaction_id}"
    
    # 构建摘要
    summary = f"# 反应摘要: {literature_id}:{reaction_id}\n\n"
    
    # 基本信息
    core_info = reaction_core.iloc[0]
    summary += f"**反应方程式**: {core_info.get('reaction_equation', 'N/A')}\n"
    summary += f"**反应类型**: {core_info.get('reaction_type_reversible', 'N/A')}\n"
    summary += f"**文献标题**: {core_info.get('title', 'N/A')}\n\n"
    
    # 酶信息
    if not reaction_enzyme.empty:
        enzyme_info = reaction_enzyme.iloc[0]
        summary += f"**酶名称**: {enzyme_info.get('enzyme_name', 'N/A')}\n"
        summary += f"**基因名称**: {enzyme_info.get('gene_name', 'N/A')}\n"
        summary += f"**物种来源**: {enzyme_info.get('organism', 'N/A')}\n"
        summary += f"**酶分类**: {enzyme_info.get('ec_number', 'N/A')}\n\n"
    
    # 性能信息
    if not reaction_activity.empty:
        activity_info = reaction_activity.iloc[0]
        summary += f"**转化率**: {activity_info.get('conversion_rate', 'N/A')}\n"
        summary += f"**产率**: {activity_info.get('product_yield', 'N/A')}\n"
        summary += f"**活性**: {activity_info.get('activity', 'N/A')}\n\n"
    
    # 实验条件
    if not reaction_conditions.empty:
        condition_info = reaction_conditions.iloc[0]
        summary += f"**温度**: {condition_info.get('temperature_celsius', 'N/A')}°C\n"
        summary += f"**pH**: {condition_info.get('ph', 'N/A')}\n"
        summary += f"**反应时间**: {condition_info.get('reaction_time_hours', 'N/A')}小时\n\n"
    
    # 反应参与分子
    if not reaction_participants.empty:
        summary += "**反应参与分子**:\n"
        for _, participant in reaction_participants.iterrows():
            summary += f"- {participant.get('participant_name', 'N/A')} ({participant.get('role', 'N/A')})\n"
    
    return summary

def _enzyme_name_or_synonym_match(df, enzyme_name):
    """
    支持enzyme_name和enzyme_synonyms（|分隔）模糊匹配，归一化后再比对。
    """
    norm_query = normalize_enzyme_name(enzyme_name)
    if 'enzyme_synonyms' not in df.columns:
        return df['enzyme_name'].apply(lambda x: norm_query in normalize_enzyme_name(x))
    def match_synonyms(synonyms, enzyme_name_col):
        # 检查主酶名
        if norm_query in normalize_enzyme_name(enzyme_name_col):
            return True
        # 检查同义词
        if pd.isnull(synonyms):
            return False
        for syn in str(synonyms).split('|'):
            if norm_query in normalize_enzyme_name(syn):
                return True
        return False
    return df.apply(lambda row: match_synonyms(row.get('enzyme_synonyms', None), row.get('enzyme_name', '')), axis=1)

def find_reactions_by_enzyme(
    enzyme_name: str = None,
    organism: str = None,
    max_results: int = 10
) -> str:
    """
    根据酶名称和物种查找相关反应。参数均可选。
    :param enzyme_name: str，可选
    :param organism: str，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    if enzymes_df.empty or core_df.empty:
        return "核心数据表未加载。"
    # 构建查询条件
    query_conditions = []
    if enzyme_name:
        query_conditions.append(_enzyme_name_or_synonym_match(enzymes_df, enzyme_name))
    if organism:
        query_conditions.append(enzymes_df['organism'].str.contains(organism, case=False, na=False))
    # 应用查询条件
    if query_conditions:
        filtered_enzymes = enzymes_df[pd.concat(query_conditions, axis=1).all(axis=1)]
    else:
        filtered_enzymes = enzymes_df
    if filtered_enzymes.empty:
        return f"未找到匹配酶 '{enzyme_name}' 和物种 '{organism}' 的反应。"
    # 合并反应信息
    merged_df = pd.merge(filtered_enzymes, core_df, on=['literature_id', 'reaction_id'])
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = merged_df.head(max_results)
    # 格式化输出
    result = f"# 酶相关反应查询结果\n\n"
    result += f"**查询条件**: 酶={enzyme_name if enzyme_name else '全部'}, 物种={organism if organism else '全部'}\n"
    result += f"**找到反应数**: {len(result_df)} (共{len(merged_df)}个)\n\n"
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **物种**: {row['organism']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **反应是否可逆**: {row['reaction_type_reversible']}\n\n"
    return result

def find_inhibition_data(
    inhibitor_name: str = None,
    enzyme_name: str = None,
    max_results: int = 10
) -> str:
    """
    查找抑制剂相关的数据。参数均可选。
    
    :param inhibitor_name: str，可选
    :param enzyme_name: str，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    
    inhibitors_df = DB.get('8_inhibitors_main', pd.DataFrame())
    inhibition_params_df = DB.get('9_inhibition_params', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    
    if inhibitors_df.empty or inhibition_params_df.empty:
        return "抑制剂数据表未加载。"
    
    # 构建查询条件
    query_conditions = []
    if inhibitor_name:
        query_conditions.append(inhibitors_df['inhibitor_name'].str.contains(inhibitor_name, case=False, na=False))
    if enzyme_name:
        # 适配酶同义词
        enzyme_match = _enzyme_name_or_synonym_match(inhibitors_df, enzyme_name) if 'enzyme_synonyms' in inhibitors_df.columns else inhibitors_df['enzyme_name'].str.contains(enzyme_name, case=False, na=False)
        query_conditions.append(enzyme_match)
    
    if not query_conditions:
        return "请提供抑制剂名称或酶名称。"
    
    # 应用查询条件
    filtered_inhibitors = inhibitors_df[pd.concat(query_conditions, axis=1).all(axis=1)]
    
    if filtered_inhibitors.empty:
        return f"未找到匹配的抑制剂数据。"
    
    # 合并抑制参数
    merged_df = pd.merge(filtered_inhibitors, inhibition_params_df, on=['literature_id', 'reaction_id'], how='left')
    
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 抑制剂数据查询结果\n\n"
    result += f"**查询条件**: 抑制剂={inhibitor_name if inhibitor_name else '全部'}, 酶={enzyme_name if enzyme_name else '全部'}\n"
    result += f"**找到记录数**: {len(result_df)}\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **抑制剂**: {row['inhibitor_name']}\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **抑制类型**: {row.get('inhibition_type', 'N/A')}\n"
        result += f"- **Ki值**: {row.get('ki_value', 'N/A')}\n"
        result += f"- **IC50**: {row.get('ic50_value', 'N/A')}\n\n"
    
    return result

def find_reactions_by_organism(
    organism: str = None,
    ec_number: str = None,
    max_results: int = 10
) -> str:
    """
    根据物种和酶EC号查找反应。参数均可选。
    
    :param organism: str，可选
    :param ec_number: str，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    
    if enzymes_df.empty or core_df.empty:
        return "核心数据表未加载。"
    
    # 构建查询条件
    query_conditions = []
    if organism:
        query_conditions.append(enzymes_df['organism'].str.contains(organism, case=False, na=False))
    if ec_number:
        query_conditions.append(enzymes_df['ec_number'].astype(str).str.contains(ec_number, case=False, na=False))
    
    if not query_conditions:
        return "请提供物种或酶EC号信息。"
    
    # 应用查询条件
    filtered_enzymes = enzymes_df[pd.concat(query_conditions, axis=1).all(axis=1)]
    if filtered_enzymes.empty:
        return f"未找到匹配的反应。"
    
    # 合并反应信息
    merged_df = pd.merge(filtered_enzymes, core_df, on=['literature_id', 'reaction_id'])
    merged_df = pd.merge(merged_df, conditions_df, on=['literature_id', 'reaction_id'], how='left')
    
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 物种+EC号反应查询结果\n\n"
    result += f"**查询条件**: 物种={organism if organism else '全部'}, EC号={ec_number if ec_number else '全部'}\n"
    result += f"**找到反应数**: {len(result_df)} (共{len(merged_df)}个)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **物种**: {row.get('organism', 'N/A')}\n"
        result += f"- **酶**: {row.get('enzyme_name', 'N/A')}\n"
        result += f"- **EC号**: {row.get('ec_number', 'N/A')}\n"
        result += f"- **反应**: {row.get('reaction_equation', 'N/A')}\n"
        # 补充实验条件等字段
        result += f"- **温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        result += f"- **pH**: {row.get('ph', 'N/A')}\n"
        result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"
    return result

def find_reactions_by_condition(
    temperature_range: str = None,
    ph_range: str = None,
    max_results: int = 10
) -> str:
    """
    根据实验条件查找反应。参数均可选。
    
    :param temperature_range: str (例如: "20-37", ">50", "<20")，可选
    :param ph_range: str (例如: "7-9", ">9", "<5")，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    
    if conditions_df.empty or core_df.empty:
        return "核心数据表未加载。"
    
    # 解析温度范围
    temp_condition = None
    if temperature_range:
        if '-' in temperature_range:
            min_temp, max_temp = map(float, temperature_range.split('-'))
            temp_condition = conditions_df['temperature_celsius'].between(min_temp, max_temp)
        elif temperature_range.startswith('>'):
            min_temp = float(temperature_range[1:])
            temp_condition = conditions_df['temperature_celsius'] > min_temp
        elif temperature_range.startswith('<'):
            max_temp = float(temperature_range[1:])
            temp_condition = conditions_df['temperature_celsius'] < max_temp
    
    # 解析pH范围
    ph_condition = None
    if ph_range:
        if '-' in ph_range:
            min_ph, max_ph = map(float, ph_range.split('-'))
            ph_condition = conditions_df['ph'].between(min_ph, max_ph)
        elif ph_range.startswith('>'):
            min_ph = float(ph_range[1:])
            ph_condition = conditions_df['ph'] > min_ph
        elif ph_range.startswith('<'):
            max_ph = float(ph_range[1:])
            ph_condition = conditions_df['ph'] < max_ph
    
    # 应用条件
    query_conditions = []
    if temp_condition is not None:
        query_conditions.append(temp_condition)
    if ph_condition is not None:
        query_conditions.append(ph_condition)
    
    if not query_conditions:
        return "请提供有效的温度或pH范围。"
    
    filtered_conditions = conditions_df[pd.concat(query_conditions, axis=1).all(axis=1)]
    
    if filtered_conditions.empty:
        return f"未找到匹配条件的反应。"
    
    # 合并数据
    merged_df = pd.merge(filtered_conditions, core_df, on=['literature_id', 'reaction_id'])
    merged_df = pd.merge(merged_df, enzymes_df, on=['literature_id', 'reaction_id'])
    
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 条件查询结果\n\n"
    result += f"**查询条件**: 温度={temperature_range if temperature_range else '全部'}, pH={ph_range if ph_range else '全部'}\n"
    result += f"**找到反应数**: {len(result_df)} (共{len(merged_df)}个)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"

        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **EC号**: {row['ec_number']}\n"
        result += f"- **温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        result += f"- **pH**: {row.get('ph', 'N/A')}\n"
        result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"
    return result

def find_reactions_with_pdb_id(
    pdb_id: str = None,
    max_results: int = 10
) -> str:
    """
    查找具有PDB ID的反应。参数均可选。
    
    :param pdb_id: str，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    
    if enzymes_df.empty or core_df.empty:
        return "核心数据表未加载。"
    
    # 查找PDB ID
    pdb_condition = enzymes_df['pdb_id'].str.contains(pdb_id, case=False, na=False) if pdb_id else True
    filtered_enzymes = enzymes_df[pdb_condition]
    
    if filtered_enzymes.empty:
        return f"未找到PDB ID为 '{pdb_id}' 的反应。"
    
    # 合并数据
    merged_df = pd.merge(filtered_enzymes, core_df, on=['literature_id', 'reaction_id'])
    
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# PDB ID查询结果\n\n"
    result += f"**查询PDB ID**: {pdb_id if pdb_id else '全部'}\n"
    result += f"**找到反应数**: {len(result_df)} (共{len(merged_df)}个)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **PDB ID**: {row['pdb_id']}\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **物种**: {row['organism']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **EC号**: {row['ec_number']}\n"

        result += f"- **温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        result += f"- **pH**: {row.get('ph', 'N/A')}\n"
        result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"    
    return result

def find_top_reactions_by_performance(
    metric: str = None,
    top_n: int = 10,
    min_data_points: int = 10
) -> str:
    """
    根据性能指标（如conversion_rate、product_yield等，来源于4_activity_performance.csv）查找表现最好的反应。
    不处理动力学参数（如kcat、Km、Vmax等），动力学参数请用find_kinetic_parameters。
    参数均可选。
    
    :param metric: str，如'conversion_rate'、'product_yield'等，可选
    :param top_n: int
    :param min_data_points: int
    """
    if not DB: return "数据库未加载。"
    
    activity_df = DB.get('4_activity_performance', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    
    if activity_df.empty or core_df.empty:
        return "核心数据表未加载。"
    
    # 只允许处理4_activity_performance.csv中的性能指标
    allowed_metrics = [
        'conversion_rate', 'product_yield', 'enantiomeric_excess', 'regioselectivity', 'stereoselectivity'
    ]
    if metric and metric not in activity_df.columns or metric not in allowed_metrics:
        return f"不支持的性能指标: {metric}。仅支持: {', '.join(allowed_metrics)}。"
    
    # 区分数值型和字符串型指标
    string_metrics = ['regioselectivity', 'stereoselectivity']
    numeric_metrics = ['conversion_rate', 'product_yield', 'enantiomeric_excess']
    
    # 只对数值型做排序和过滤
    if metric and metric in numeric_metrics:
        # 确保有足够的数据点
        min_data_points = max(min_data_points, QUERY_CONFIG["min_data_points"])
        if len(activity_df) < min_data_points:
            return f"数据点不足（{len(activity_df)} < {min_data_points}）。"
        # 数值转换
        activity_df[metric] = pd.to_numeric(activity_df[metric], errors='coerce')
        activity_df = activity_df.dropna(subset=[metric])
        if len(activity_df) < min_data_points:
            return f"有效数据点不足（{len(activity_df)} < {min_data_points}）。"
        # 排序并获取前N个
        top_n = min(top_n, QUERY_CONFIG["default_top_n"])
        top_reactions = activity_df.nlargest(top_n, metric)
    else:
        # 字符串型指标，直接取前N条非空记录
        top_n = min(top_n, QUERY_CONFIG["default_top_n"])
        top_reactions = activity_df[activity_df[metric].notnull() & (activity_df[metric] != '')].head(top_n)
    
    # 合并数据
    merged_df = pd.merge(top_reactions, core_df, on=['literature_id', 'reaction_id'])
    merged_df = pd.merge(merged_df, enzymes_df, on=['literature_id', 'reaction_id'])
    
    # 格式化输出
    result = f"# 性能排名查询结果\n\n"
    result += f"**性能指标**: {metric if metric else '全部'}\n"
    result += f"**排名数量**: {top_n}\n"
    result += f"**总数据点**: {len(activity_df)}\n\n"
    
    # 单位、误差字段自动适配
    unit_col = f"{metric}_unit" if metric else None
    error_col = f"{metric}_error" if metric else None
    for i, (_, row) in enumerate(merged_df.iterrows(), 1):
        result += f"## 第{i}名: {row['literature_id']}:{row['reaction_id']}\n"
        value = row[metric] if metric else row['conversion_rate'] # 默认值
        if metric and metric in numeric_metrics:
            unit = row[unit_col] if unit_col in row and pd.notnull(row[unit_col]) else ''
            error = row[error_col] if error_col in row and pd.notnull(row[error_col]) else ''
            result += f"- **{metric}**: {value} {unit} {f'(误差: {error})' if error else ''}\n"
        else:
            result += f"- **{metric}**: {value}\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **物种**: {row['organism']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **EC号**: {row['ec_number']}\n"
        result += f"- **温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        result += f"- **pH**: {row.get('ph', 'N/A')}\n"
        result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"
    return result

def find_conditions_by_enzyme(
    enzyme_name: str = None,
    max_results: int = 10
) -> str:
    """
    查找特定酶的实验条件。参数均可选。
    
    :param enzyme_name: str，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    
    if enzymes_df.empty or conditions_df.empty:
        return "核心数据表未加载。"
    
    # 查找酶
    enzyme_condition = _enzyme_name_or_synonym_match(enzymes_df, enzyme_name) if enzyme_name else True
    filtered_enzymes = enzymes_df[enzyme_condition]
    
    if filtered_enzymes.empty:
        return f"未找到酶 '{enzyme_name}' 的记录。"
    
    # 合并条件数据
    merged_df = pd.merge(filtered_enzymes, conditions_df, on=['literature_id', 'reaction_id'])
    
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 酶条件查询结果\n\n"
    result += f"**目标酶**: {enzyme_name if enzyme_name else '全部'}\n"
    result += f"**找到记录数**: {len(result_df)} (共{len(merged_df)}个)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## 文献编号: {row['literature_id']} 反应编号: {row['reaction_id']}\n"
        result += f"- **酶名称**: {row.get('enzyme_name', 'N/A')}\n"
        result += f"- **来源物种**: {row.get('organism', 'N/A')}\n"
        result += f"- **实验温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        result += f"- **pH值**: {row.get('ph', 'N/A')}\n"
        result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"
      
    return result

def find_enzymes_by_participant(
    participant_name: str = None,
    max_results: int = 10
) -> str:
    """
    根据反应参与分子查找相关酶。参数均可选。
    
    :param participant_name: str，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    
    participants_df = DB.get('5_reaction_participants', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    
    if participants_df.empty or enzymes_df.empty:
        return "核心数据表未加载。"
    
    # 查找参与者
    participant_condition = participants_df['participant_name'].str.contains(participant_name, case=False, na=False) if participant_name else True
    filtered_participants = participants_df[participant_condition]
    
    if filtered_participants.empty:
        return f"未找到参与者 '{participant_name}' 的记录。"
    
    # 合并数据
    merged_df = pd.merge(filtered_participants, enzymes_df, on=['literature_id', 'reaction_id'])
    merged_df = pd.merge(merged_df, core_df, on=['literature_id', 'reaction_id'])
    
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 参与者酶查询结果\n\n"
    result += f"**目标参与者**: {participant_name if participant_name else '全部'}\n"
    result += f"**找到记录数**: {len(result_df)} (共{len(merged_df)}个)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **参与者**: {row['participant_name']} ({row['role']})\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **物种**: {row['organism']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **EC号**: {row['ec_number']}\n"

        result += f"- **温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        result += f"- **pH**: {row.get('ph', 'N/A')}\n"
        result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"    
    return result

# 新增智能查询工具
def smart_search_reactions(
    search_query: str = None,
    search_fields: List[str] = None,
    max_results: int = 10
) -> str:
    """
    智能搜索反应，支持多字段模糊匹配。参数均可选。
    
    :param search_query: str，可选
    :param search_fields: List[str]，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    participants_df = DB.get('5_reaction_participants', pd.DataFrame())
    
    if core_df.empty or enzymes_df.empty:
        return "核心数据表未加载。"
    
    # 合并数据
    merged_df = pd.merge(core_df, enzymes_df, on=['literature_id', 'reaction_id'])
    
    # 如果涉及底物/产物/参与者，合并参与者表
    if any(f in ['participant_name', 'role'] for f in (search_fields or [])):
        merged_df = pd.merge(merged_df, participants_df, on=['literature_id', 'reaction_id'], how='left')
    
    # 字段推断
    valid_fields = [f for f in (search_fields or []) if f in merged_df.columns]
    if not valid_fields:
        valid_fields = guess_search_fields(search_query)
        valid_fields = [f for f in valid_fields if f in merged_df.columns]
    # fallback: 全字段
    if not valid_fields:
        valid_fields = [col for col in [
            'reaction_equation', 'reaction_type_reversible', 'notes',
            'enzyme_name', 'enzyme_synonyms', 'gene_name', 'organism', 'ec_number',
            'participant_name', 'role'
        ] if col in merged_df.columns]
    # 构建搜索条件
    search_conditions = []
    for field in valid_fields:
        if field in merged_df.columns:
            if field == "enzyme_name" or field == "enzyme_synonyms":
                search_conditions.append(_enzyme_name_or_synonym_match(merged_df, search_query))
            else:
                search_conditions.append(merged_df[field].astype(str).str.contains(search_query, case=False, na=False))
    
    if not search_conditions:
        return "未找到有效的搜索字段。"
    
    # 应用搜索条件（OR逻辑）
    combined_condition = pd.concat(search_conditions, axis=1).any(axis=1)
    filtered_df = merged_df[combined_condition]
    
    if filtered_df.empty:
        return f"未找到匹配查询 '{search_query}' 的反应。"
    
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = filtered_df.head(max_results)
    
    # 格式化输出
    result = f"# 智能搜索结果\n\n"
    result += f"**搜索查询**: {search_query if search_query else '全部'}\n"
    result += f"**搜索字段**: {', '.join(valid_fields) if valid_fields else '全部'}\n"
    result += f"**找到反应数**: {len(result_df)} (共{len(filtered_df)}个)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **酶**: {row.get('enzyme_name', 'N/A')}\n"
        result += f"- **物种**: {row.get('organism', 'N/A')}\n"
        result += f"- **反应**: {row.get('reaction_equation', 'N/A')}\n"
        result += f"- **反应是否可逆**: {row.get('reaction_type_reversible', 'N/A')}\n"
        if 'participant_name' in row and pd.notnull(row['participant_name']):
            result += f"- **参与分子**: {row['participant_name']} ({row.get('role', 'N/A')})\n"
        if 'notes' in row and pd.notnull(row['notes']):
            result += f"- **备注**: {row['notes']}\n"
        result += "\n"
    return result

def guess_search_fields(user_query: str) -> list:
    """
    根据用户输入内容智能推断最合适的数据库字段（严格依据实际字段名）。
    """
    # 反应方程式结构
    if '->' in user_query or '→' in user_query:
        return ['reaction_equation']
    # EC号
    if re.match(r'\d+\.\d+\.\d+\.\d+', user_query):
        return ['ec_number']
    # 酶名关键词
    if any(kw in user_query.lower() for kw in ['酶', 'ase', 'protein']):
        return ['enzyme_name', 'enzyme_synonyms']
    # 反应类型
    if '可逆' in user_query or '不可逆' in user_query or '类型' in user_query:
        return ['reaction_type_reversible']
    # 底物/产物
    if '底物' in user_query or '产物' in user_query or 'substrate' in user_query.lower() or 'product' in user_query.lower():
        return ['participant_name', 'role']
    # 基因名
    if '基因' in user_query or 'gene' in user_query.lower():
        return ['gene_name']
    # 物种
    if '物种' in user_query or 'organism' in user_query.lower():
        return ['organism']
    # 备注
    if '备注' in user_query or 'note' in user_query.lower():
        return ['notes']
    # fallback: 所有主要字段
    return [
        'reaction_equation', 'reaction_type_reversible', 'notes',
        'enzyme_name', 'enzyme_synonyms', 'gene_name', 'organism', 'ec_number',
        'participant_name', 'role'
    ]

def get_database_statistics() -> str:
    """
    获取数据库统计信息。
    """
    if not DB: return "数据库未加载。"
    
    result = "# 数据库统计信息\n\n"
    
    for table_name, df in DB.items():
        result += f"## {table_name}\n"
        result += f"- **记录数**: {len(df)}\n"
        result += f"- **列数**: {len(df.columns)}\n"
        result += f"- **列名**: {', '.join(df.columns.tolist())}\n\n"
    
    return result

def find_similar_reactions(
    target_reaction_id: str = None,
    similarity_criteria: str = None,
    max_results: int = 10
) -> str:
    """
    查找相似反应。参数均可选。
    
    :param target_reaction_id: str，可选
    :param similarity_criteria: str，可选
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    
    # 解析目标反应ID
    if target_reaction_id and ':' not in target_reaction_id:
        return "反应ID格式错误，应为 'literature_id:reaction_id'"
    
    lit_id = target_reaction_id.split(':', 1)[0] if target_reaction_id else None
    react_id = target_reaction_id.split(':', 1)[1] if target_reaction_id else None
    
    # 获取目标反应信息
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    
    target_reaction = core_df[(core_df['literature_id'] == lit_id) & (core_df['reaction_id'] == react_id)] if lit_id and react_id else pd.DataFrame()
    target_enzyme = enzymes_df[(enzymes_df['literature_id'] == lit_id) & (enzymes_df['reaction_id'] == react_id)] if lit_id and react_id else pd.DataFrame()
    
    if target_reaction.empty:
        return f"未找到目标反应 {target_reaction_id}"
    
    # 根据相似性标准查找
    if similarity_criteria == "enzyme":
        if not target_enzyme.empty:
            enzyme_name = target_enzyme.iloc[0]['enzyme_name']
            similar = enzymes_df[
                _enzyme_name_or_synonym_match(enzymes_df, enzyme_name.split('_')[0]) &
                (enzymes_df['literature_id'] != lit_id)
            ]
        else:
            return "目标反应无酶信息"
    elif similarity_criteria == "ec_number":
        if not target_enzyme.empty:
            ec_number = str(target_enzyme.iloc[0].get('ec_number', ''))
            if ec_number:
                # 支持主类/子类模糊匹配（如1.1.1.1主类为1.1，子类为1.1.1）
                ec_main = '.'.join(ec_number.split('.')[:2])  # 主类
                ec_sub = '.'.join(ec_number.split('.')[:3])  # 子类
                similar = enzymes_df[
                    (enzymes_df['ec_number'].astype(str).str.startswith(ec_main)) &
                    (enzymes_df['literature_id'] != lit_id)
                ]
                # 如果主类匹配结果太多，可进一步用子类筛选
                if len(similar) > max_results * 2 and len(ec_sub) > 0:
                    similar = enzymes_df[
                        (enzymes_df['ec_number'].astype(str).str.startswith(ec_sub)) &
                        (enzymes_df['literature_id'] != lit_id)
                    ]
            else:
                return "目标反应无EC号信息"
        else:
            return "目标反应无酶信息"
    else:
        return "不支持的相似性标准"
    
    if similar.empty:
        return f"未找到相似反应"
    
    # 合并数据
    merged_df = pd.merge(similar, core_df, on=['literature_id', 'reaction_id'])
    merged_df = pd.merge(merged_df, enzymes_df, on=['literature_id', 'reaction_id'])
    
    # 限制结果数量
    max_results = min(max_results, QUERY_CONFIG["max_results"])
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 相似反应查询结果\n\n"
    result += f"**目标反应**: {target_reaction_id if target_reaction_id else '全部'}\n"
    result += f"**相似性标准**: {similarity_criteria if similarity_criteria else '全部'}\n"
    result += f"**找到相似反应数**: {len(result_df)}\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **EC号**: {row['ec_number']}\n"
        result += f"- **物种**: {row['organism']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **反应是否可逆**: {row['reaction_type_reversible']}\n\n"
        result += f"- **温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        result += f"- **pH**: {row.get('ph', 'N/A')}\n"
        result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"
    
    return result

def analyze_reaction_patterns(
    pattern_type: str = None,
    min_occurrences: int = 1
) -> str:
    """
    分析反应模式。参数均可选。
    
    :param pattern_type: str，可选
    :param min_occurrences: int
    """
    if not DB: return "数据库未加载。"
    
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    
    if core_df.empty or enzymes_df.empty:
        return "核心数据表未加载。"
    
    # 合并数据
    merged_df = pd.merge(core_df, enzymes_df, on=['literature_id', 'reaction_id'])
    
    result = f"# 反应模式分析\n\n"
    result += f"**分析类型**: {pattern_type if pattern_type else '全部'}\n"
    result += f"**最小出现次数**: {min_occurrences}\n\n"
    
    if pattern_type == "enzyme_frequency":
        # 酶使用频率分析
        enzyme_counts = merged_df['enzyme_name'].value_counts()
        frequent_enzymes = enzyme_counts[enzyme_counts >= min_occurrences]
        
        result += "## 常用酶分析\n\n"
        for enzyme, count in frequent_enzymes.items():
            result += f"- **{enzyme}**: {count}次使用\n"
    
    elif pattern_type == "organism_frequency":
        # 物种使用频率分析
        organism_counts = merged_df['organism'].value_counts()
        frequent_organisms = organism_counts[organism_counts >= min_occurrences]
        
        result += "## 常用物种分析\n\n"
        for organism, count in frequent_organisms.items():
            result += f"- **{organism}**: {count}次使用\n"
    
    elif pattern_type == "reaction_type_frequency":
        # 反应类型频率分析
        type_counts = merged_df['reaction_type'].value_counts()
        frequent_types = type_counts[type_counts >= min_occurrences]
        
        result += "## 反应类型分析\n\n"
        for rtype, count in frequent_types.items():
            result += f"- **{rtype}**: {count}次出现\n"
    
    else:
        return "不支持的模式分析类型"
    
    return result

def find_kinetic_parameters(
    literature_id: str = None,
    reaction_id: str = None,
    parameter_type: str = None,
    max_results: int = 10
) -> str:
    """
    查询并展示指定反应的动力学参数（如kcat、Km、Vmax、kcat_km、specific_activity等）。
    支持按文献、反应、参数类型筛选。参数均可选。
    :param literature_id: str，可选
    :param reaction_id: str，可选
    :param parameter_type: str，可选（如'kcat','Km','Vmax', 'kcat_km', 'specific_activity'等）
    :param max_results: int
    """
    if not DB: return "数据库未加载。"
    kinetic_df = DB.get('6_kinetic_parameters', pd.DataFrame())
    if kinetic_df.empty:
        return "动力学参数数据表未加载。"

    # 条件筛选
    df = kinetic_df.copy()
    if literature_id:
        df = df[df['literature_id'] == literature_id]
    if reaction_id:
        df = df[df['reaction_id'] == reaction_id]
    if parameter_type:
        df = df[df['parameter_type'].str.lower() == parameter_type.lower()]
    if df.empty:
        return "未找到匹配的动力学参数数据。"

    # 限制结果数量
    df = df.head(max_results)

    # 分组展示
    result = f"# 动力学参数查询结果\n\n"
    group_cols = ['literature_id', 'reaction_id', 'source_type', 'mutation_description']
    grouped = df.groupby(group_cols)
    for group_keys, group_df in grouped:
        lit, rid, src, mut = group_keys
        result += f"## 文献: {lit} 反应: {rid} 类型: {src}"
        if mut and str(mut).strip():
            result += f" 突变: {mut}"
        result += "\n"
        for _, row in group_df.iterrows():
            result += f"- **参数类型**: {row['parameter_type']}"
            if row['substrate_name'] and str(row['substrate_name']).strip():
                result += f" | **底物**: {row['substrate_name']}"
            result += f" | **数值**: {row['value']} {row['unit']}"
            if row['error_margin'] and str(row['error_margin']).strip():
                result += f" (误差: {row['error_margin']})"
            if row['details'] and str(row['details']).strip():
                result += f" | 说明: {row['details']}"
            result += "\n"
        result += "\n"
    return result

# --- 创建FunctionTool实例 ---
get_reaction_summary_tool = FunctionTool(func=get_reaction_summary)
find_reactions_by_enzyme_tool = FunctionTool(func=find_reactions_by_enzyme)
find_inhibition_data_tool = FunctionTool(func=find_inhibition_data)
find_reactions_by_organism_tool = FunctionTool(func=find_reactions_by_organism)
find_reactions_by_condition_tool = FunctionTool(func=find_reactions_by_condition)
find_reactions_with_pdb_id_tool = FunctionTool(func=find_reactions_with_pdb_id)
find_top_reactions_by_performance_tool = FunctionTool(func=find_top_reactions_by_performance)
find_kinetic_parameters_tool = FunctionTool(func=find_kinetic_parameters)
find_conditions_by_enzyme_tool = FunctionTool(func=find_conditions_by_enzyme)
find_enzymes_by_participant_tool = FunctionTool(func=find_enzymes_by_participant)
smart_search_reactions_tool = FunctionTool(func=smart_search_reactions)
get_database_statistics_tool = FunctionTool(func=get_database_statistics)
find_similar_reactions_tool = FunctionTool(func=find_similar_reactions)
analyze_reaction_patterns_tool = FunctionTool(func=analyze_reaction_patterns)

