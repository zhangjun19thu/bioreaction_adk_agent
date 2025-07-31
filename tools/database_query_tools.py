from google.adk.tools import FunctionTool
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from .database_loader import DB

from ..config import QUERY_CONFIG, ANALYSIS_CONFIG
import re
import logging
logging.basicConfig(filename="test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

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
        # 转化率
        cr = activity_info.get('conversion_rate', 'N/A')
        cr_unit = activity_info.get('conversion_rate_unit', '')
        cr_error = activity_info.get('conversion_rate_error', '')
        summary += f"**转化率**: {cr} {cr_unit} {(f'(误差: {cr_error})' if cr_error else '')}\n"
        # 产率
        py = activity_info.get('product_yield', 'N/A')
        py_unit = activity_info.get('product_yield_unit', '')
        py_error = activity_info.get('product_yield_error', '')
        summary += f"**产率**: {py} {py_unit} {(f'(误差: {py_error})' if py_error else '')}\n"
        # 选择性
        summary += f"**对映选择性**: {activity_info.get('regioselectivity', 'N/A')}\n"
        summary += f"**立体选择性**: {activity_info.get('stereoselectivity', 'N/A')}\n"
    
        # 对映体过量
        ee = activity_info.get('enantiomeric_excess', None)
        ee_unit = activity_info.get('enantiomeric_excess_unit', '')
        ee_error = '' # 如有enantiomeric_excess_error字段可补充
        if ee is not None and ee != '' and ee != 'N/A':
            summary += f"**对映体过量**: {ee} {ee_unit} {(f'(误差: {ee_error})' if ee_error else '')}\n\n"

    # 实验条件
    if not reaction_conditions.empty:
        condition_info = reaction_conditions.iloc[0]
        summary += f"**温度**: {condition_info.get('temperature_celsius', 'N/A')}°C\n"
        summary += f"**pH**: {condition_info.get('ph', 'N/A')}\n"
        # summary += f"**反应时间**: {condition_info.get('reaction_time_hours', 'N/A')}小时\n\n"
        summary += f"**pH补充说明**: {condition_info.get('ph_details', 'N/A')}\n"
        summary += f"**实验类型**: {condition_info.get('assay_type', 'N/A')}\n"
        summary += f"**实验细节**: {condition_info.get('assay_details', 'N/A')}\n"
        summary += f"**缓冲液/溶剂**: {condition_info.get('solvent_buffer', 'N/A')}\n"
        summary += f"**表达宿主**: {condition_info.get('expression_host', 'N/A')}\n"
        summary += f"**表达载体**: {condition_info.get('expression_vector', 'N/A')}\n"
        summary += f"**诱导条件**: {condition_info.get('expression_induction', 'N/A')}\n"

    # 反应参与分子
    if not reaction_participants.empty:
        summary += "**反应参与分子**:\n"
        for _, participant in reaction_participants.iterrows():
            summary += f"- {participant.get('participant_name', 'N/A')} ({participant.get('role', 'N/A')})\n"
    
    summary += '\n'
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

def find_reactions_by_enzyme(**kwargs) -> str:

    """
    根据酶名称或物种查找相关反应。参数均可选。
    :param enzyme_name: str，可选，不严格要求
    :param organism: str，可选,不严格要求
    :param max_results: int
    """
    enzyme_name = kwargs.get('enzyme_name', None)
    organism = kwargs.get('organism', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
    
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

    result_df = merged_df.head(max_results)
    # 格式化输出
    result = f"# 酶相关反应查询结果\n\n"
    result += f"**查询条件**: 酶={enzyme_name if enzyme_name else '全部'}, 物种={organism if organism else '全部'}\n"
    result += f"**输出反应数**: {len(result_df)} (共找到{len(merged_df)}个反应)\n\n"
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **物种**: {row['organism']}\n"
        result += f"- **酶EC号**: {row['ec_number']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **反应是否可逆**: {row['reaction_type_reversible']}\n\n"
    return result

def find_inhibition_data(**kwargs) -> str:
    """
    根据抑制剂名或酶名字查找抑制剂相关的数据。参数均可选。
    
    :param inhibitor_name: str，可选，不严格要求
    :param enzyme_name: str，可选，不严格要求
    :param max_results: int
    """
    inhibitor_name = kwargs.get('inhibitor_name', None)
    enzyme_name = kwargs.get('enzyme_name', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
    if not DB: return "数据库未加载。"
    
    inhibitors_df = DB.get('8_inhibitors_main', pd.DataFrame())
    inhibition_params_df = DB.get('9_inhibition_params', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    
    if inhibitors_df.empty or inhibition_params_df.empty:
        return "抑制剂数据表未加载。"
    
    # 先合并酶信息
    merged_inhibitors = pd.merge(inhibitors_df, enzymes_df, on=['literature_id', 'reaction_id'], how='left', suffixes=('', '_enzyme'))
    # 构建查询条件
    query_conditions = []
    if inhibitor_name:
        query_conditions.append(merged_inhibitors['inhibitor_name'].str.contains(inhibitor_name, case=False, na=False))
    if enzyme_name:
        # 支持酶名和同义词模糊匹配
        enzyme_match = _enzyme_name_or_synonym_match(merged_inhibitors, enzyme_name) if 'enzyme_synonyms' in merged_inhibitors.columns else merged_inhibitors['enzyme_name'].str.contains(enzyme_name, case=False, na=False)
        query_conditions.append(enzyme_match)
    
    if not query_conditions:
        return "请提供抑制剂名称或酶名称。"
    
    # 应用查询条件
    filtered_inhibitors = merged_inhibitors[pd.concat(query_conditions, axis=1).all(axis=1)]
    
    if filtered_inhibitors.empty:
        return f"未找到匹配的抑制剂数据。"
    
    # 合并抑制参数
    merged_df = pd.merge(filtered_inhibitors, inhibition_params_df, on=['literature_id', 'reaction_id','inhibitor_name'], how='left')
    
    result_df = merged_df.head(max_results)
    # print(result_df.columns)
    # 格式化输出
    result = f"# 抑制剂数据查询结果\n\n"
    result += f"**查询条件**: 抑制剂={inhibitor_name if inhibitor_name else '全部'}, 酶={enzyme_name if enzyme_name else '全部'}\n"
    result += f"**输出记录数**: {len(result_df)} (共找到{len(merged_df)}条记录)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **抑制剂**: {row.get('inhibitor_name', 'N/A')}\n"
        result += f"- **酶**: {row.get('enzyme_name', 'N/A')}\n"
        result += f"- **抑制类型**: {row.get('inhibition_type', 'N/A')}\n"
        result += f"- **定性效应**: {row.get('activity_qualitative', 'N/A')} and {row.get('inhibition_qualitative', 'N/A')} \n"
        # 输出所有参数类型及数值
        if pd.notnull(row.get('parameter_type')) and pd.notnull(row.get('value')):
            param_type = str(row.get('parameter_type', '')).strip()
            value = row.get('value', 'N/A')
            unit = row.get('unit', '')
            error = row.get('error_margin', '')
            param_str = f"- **{param_type}**: {value} {unit}"
            if error and str(error).strip():
                param_str += f" (误差: {error})"
            result += param_str + "\n"
        result += f"- **热力学信息**: {row.get('thermodynamics', 'N/A')}\n"
        result += f"- **说明补充**: {row.get('details', 'N/A')} and {row.get('notes', 'N/A')} \n\n"
    return result

def find_reactions_by_organism(**kwargs) -> str:
    """
    根据物种和酶EC号查找反应。参数均可选。
    
    :param organism: str，可选
    :param ec_number: str，可选
    :param max_results: int
    """
    organism = kwargs.get('organism', None)
    ec_number = kwargs.get('ec_number', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
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
    
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 物种+EC号反应查询结果\n\n"
    result += f"**查询条件**: 物种={organism if organism else '全部'}, EC号={ec_number if ec_number else '全部'}\n"
    result += f"**输出反应数**: {len(result_df)} (共找到{len(merged_df)}个反应)\n\n"
    
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

def find_reactions_by_condition(**kwargs) -> str:
    """
    根据实验条件查找反应。参数均可选。
    
    :param temperature_range: str (例如: "20-37", ">50", "<20")，可选
    :param ph_range: str (例如: "7-9", ">9", "<5")，可选
    :param max_results: int
    """
    temperature_range = kwargs.get('temperature_range', None)
    ph_range = kwargs.get('ph_range', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
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
    
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 条件查询结果\n\n"
    result += f"**查询条件**: 温度={temperature_range if temperature_range else '全部'}, pH={ph_range if ph_range else '全部'}\n"
    result += f"**输出记录数**: {len(result_df)} (共找到{len(merged_df)}个记录)\n\n"
    
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

def find_reactions_with_pdb_id(**kwargs) -> str:
    """
    查找具有PDB ID的反应。参数均可选。
    
    :param pdb_id: str，可选
    :param max_results: int
    """
    pdb_id = kwargs.get('pdb_id', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
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

    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# PDB ID查询结果\n\n"
    result += f"**查询PDB ID**: {pdb_id if pdb_id else '全部'}\n"
    result += f"**输出记录数**: {len(result_df)} (共找到{len(merged_df)}个记录)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **PDB ID**: {row['pdb_id']}\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **物种**: {row['organism']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **EC号**: {row['ec_number']}\n"

        # result += f"- **温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        # result += f"- **pH**: {row.get('ph', 'N/A')}\n"
        # result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        # result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        # result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        # result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        # result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        # result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        # result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"    
    return result

def find_top_reactions_by_performance(**kwargs) -> str:
    """
    根据性能指标（如conversion_rate、product_yield等，来源于4_activity_performance.csv）查找表现最好的反应。
    不处理动力学参数（如kcat、Km、Vmax等），动力学参数请用find_kinetic_parameters。
    参数均可选。
    
    :param metric: str，如'conversion_rate'、'product_yield'等，可选
    :param top_n: int
    :param min_data_points: int
    """
    metric = kwargs.get('metric', None)
    top_n = kwargs.get('top_n', 5)
    min_data_points = kwargs.get('min_data_points', 5)
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
    for i, (_, row) in enumerate(merged_df.iterrows(), 1):
        result += f"## 第{i}名: {row['literature_id']}:{row['reaction_id']}\n"
        value = row[metric] if metric else row['conversion_rate'] # 默认值
        # 新增：始终输出unit和error（仅对conversion_rate、product_yield、enantiomeric_excess）
        if metric in ['conversion_rate', 'product_yield', 'enantiomeric_excess']:
            unit_col = f"{metric}_unit"
            error_col = f"{metric}_error"
            unit = row[unit_col] if unit_col in row and pd.notnull(row[unit_col]) else ''
            error = row[error_col] if error_col in row and pd.notnull(row[error_col]) else ''
            result += f"- **{metric}**: {value} {unit} {(f'(误差: {error})' if error else '')}\n"
        else:
            result += f"- **{metric}**: {value}\n"
        result += f"- **酶**: {row['enzyme_name']}\n"
        result += f"- **物种**: {row['organism']}\n"
        result += f"- **反应**: {row['reaction_equation']}\n"
        result += f"- **EC号**: {row['ec_number']}\n"
        
        # result += f"- **温度**: {row.get('temperature_celsius', 'N/A')}°C\n"
        # result += f"- **pH**: {row.get('ph', 'N/A')}\n"
        # result += f"- **pH补充说明**: {row.get('ph_details', 'N/A')}\n"
        # result += f"- **实验类型**: {row.get('assay_type', 'N/A')}\n"
        # result += f"- **实验细节**: {row.get('assay_details', 'N/A')}\n"
        # result += f"- **缓冲液/溶剂**: {row.get('solvent_buffer', 'N/A')}\n"
        # result += f"- **表达宿主**: {row.get('expression_host', 'N/A')}\n"
        # result += f"- **表达载体**: {row.get('expression_vector', 'N/A')}\n"
        # result += f"- **诱导条件**: {row.get('expression_induction', 'N/A')}\n"
        result += "\n"
    return result

def find_conditions_by_enzyme(**kwargs) -> str:
    """
    查找特定酶的实验条件。参数均可选。
    
    :param enzyme_name: str，可选
    :param max_results: int
    """
    enzyme_name = kwargs.get('enzyme_name', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
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
    
    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 酶条件查询结果\n\n"
    result += f"**目标酶**: {enzyme_name if enzyme_name else '全部'}\n"
    result += f"**输出记录数**: {len(result_df)} (共找到记录{len(merged_df)}个)\n\n"
    
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

def find_enzymes_by_participant(**kwargs) -> str:
    """
    根据反应参与分子查找相关酶。参数均可选。
    
    :param participant_name: str，可选
    :param max_results: int
    """
    participant_name = kwargs.get('participant_name', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
    if not DB: return "数据库未加载。"
    
    participants_df = DB.get('5_reaction_participants', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    
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
    merged_df = pd.merge(merged_df, conditions_df, on=['literature_id', 'reaction_id'])

    result_df = merged_df.head(max_results)
    
    # 格式化输出
    result = f"# 参与者酶查询结果\n\n"
    result += f"**目标参与者**: {participant_name if participant_name else '全部'}\n"
    result += f"**输出记录数**: {len(result_df)} (共找到记录{len(merged_df)}个)\n\n"
    
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
    search_query: str,
    search_fields: List[str],
    max_results: int
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
                # case=False，理论上不区分大小写;去除正则影响，使用regex=False
                search_conditions.append(merged_df[field].astype(str).str.contains(search_query, case=False, na=False,regex=False))
    
    if not search_conditions:
        return "未找到有效的搜索字段。"
    
    # 应用搜索条件（OR逻辑）
    combined_condition = pd.concat(search_conditions, axis=1).any(axis=1)
    filtered_df = merged_df[combined_condition]
    
    if filtered_df.empty:
        return f"未找到匹配查询 '{search_query}' 的反应。"
    
    result_df = filtered_df.head(max_results)
    
    # 格式化输出
    result = f"# 智能搜索结果\n\n"
    result += f"**搜索查询**: {search_query if search_query else '全部'}\n"
    result += f"**搜索字段**: {', '.join(valid_fields) if valid_fields else '全部'}\n"
    result += f"**输出记录数**: {len(result_df)} (共找到记录{len(filtered_df)}个)\n\n"
    
    for _, row in result_df.iterrows():
        result += f"## 文献id:{row['literature_id']},反应id:{row['reaction_id']}\n"
        result += f"- **酶**: {row.get('enzyme_name', 'N/A')}\n"
        result += f"- **物种**: {row.get('organism', 'N/A')}\n"
        result += f"- **酶EC号**: {row.get('ec_number', 'N/A')}\n"
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

def find_similar_reactions(**kwargs) -> str:
    """
    根据反应id及相似性标准查找制定反应相似的反应。
    
    :param target_reaction_id: str
    :param similarity_criteria: str
    :param max_results: int,可选
    """
    target_reaction_id = kwargs.get('target_reaction_id', None)
    similarity_criteria = kwargs.get('similarity_criteria', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
    if not DB: return "数据库未加载。"

    # 合并所有相关表
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    merged_df = pd.merge(enzymes_df, core_df, on=['literature_id', 'reaction_id'])
    merged_df = pd.merge(merged_df, conditions_df, on=['literature_id', 'reaction_id'])
    
    # 解析目标反应
    if target_reaction_id and ':' not in target_reaction_id:
        return "反应ID格式错误，应为 'literature_id:reaction_id'"
    lit_id = target_reaction_id.split(':', 1)[0] if target_reaction_id else None
    react_id = target_reaction_id.split(':', 1)[1] if target_reaction_id else None

    target_row = merged_df[(merged_df['literature_id'] == lit_id) & (merged_df['reaction_id'] == react_id)]
    if target_row.empty:
        return f"未找到目标反应 {target_reaction_id}"

    # 根据相似性标准筛选
    # 支持多种酶相关的相似性标准，区分酶名与EC号
    if any(x in str(similarity_criteria).lower() for x in ["enzyme", "酶", "enzyme_name", "酶名"]):
        enzyme_name = target_row.iloc[0]['enzyme_name']
        similar = merged_df[
            (merged_df['enzyme_name'].str.contains(enzyme_name.split('_')[0], case=False, na=False)) &
            ~((merged_df['literature_id'] == lit_id) & (merged_df['reaction_id'] == react_id))
        ]
    elif any(x in str(similarity_criteria).lower() for x in ["ec_number", "ec号", "ec", "酶分类"]):
        ec_number = str(target_row.iloc[0].get('ec_number', ''))
        if ec_number:
            ec_main = '.'.join(ec_number.split('.')[:2])
            similar = merged_df[
                (merged_df['ec_number'].astype(str).str.startswith(ec_main)) &
                ~((merged_df['literature_id'] == lit_id) & (merged_df['reaction_id'] == react_id))
            ]
        else:
            return "目标反应无EC号信息"
    else:
        return "不支持的相似性标准"

    if similar.empty:
        return f"未找到相似反应"

    result_df = similar.head(max_results)

    # 格式化输出
    result = f"# 相似反应查询结果\n\n"
    result += f"**目标反应**: {target_reaction_id if target_reaction_id else '全部'}\n"
    result += f"**相似性标准**: {similarity_criteria if similarity_criteria else '全部'}\n"
    result += f"**输出相似反应数**: {len(result_df)}，共找到记录{len(similar)}条\n\n"

    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']}\n"
        result += f"- **酶**: {row.get('enzyme_name', 'N/A')}\n"
        result += f"- **EC号**: {row.get('ec_number', 'N/A')}\n"
        result += f"- **物种**: {row.get('organism', 'N/A')}\n"
        result += f"- **反应**: {row.get('reaction_equation', 'N/A')}\n"
        result += f"- **反应是否可逆**: {row.get('reaction_type_reversible', 'N/A')}\n\n"
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
        
    MAX_OUTPUT_LEN = 1000  # 你可以根据实际情况调整
    if len(result) > MAX_OUTPUT_LEN:
        result = result[:MAX_OUTPUT_LEN] + "\n\n【内容过长，仅显示前部分】"
    return result

def analyze_reaction_patterns(
    pattern_type: str,
    min_occurrences: int
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
        type_counts = merged_df['reaction_type_reversible'].value_counts()
        frequent_types = type_counts[type_counts >= min_occurrences]
        
        result += "## 反应类型分析\n\n"
        for rtype, count in frequent_types.items():
            result += f"- **{rtype}**: {count}次出现\n"
    
    else:
        return "不支持的模式分析类型"
    
    return result

def find_kinetic_parameters(**kwargs) -> str:
    """
    查询并展示指定反应的动力学参数（如kcat、Km、Vmax、kcat_km、specific_activity等）。
    支持按文献、反应、参数类型、酶名筛选。参数均可选。
    
    :param literature_id: str，可选，举例: PMID32044030
    :param reaction_id: str，可选，举例: reaction_1
    :param parameter_type: str，可选（如'kcat','Km','Vmax', 'kcat_km', 'specific_activity'等）
    :param enzyme_name: str，可选（支持酶名检索）
    :param max_results: int
    """
    literature_id = kwargs.get('literature_id', None)
    reaction_id = kwargs.get('reaction_id', None)
    parameter_type = kwargs.get('parameter_type', None)
    enzyme_name = kwargs.get('enzyme_name', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
    
    if not DB: return "数据库未加载。"
    kinetic_df = DB.get('6_kinetic_parameters', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    if kinetic_df.empty:
        return "动力学参数数据表未加载。"
    # 条件筛选
    df = kinetic_df.copy()
    # 新增：支持酶名检索
    if enzyme_name and not enzymes_df.empty:
        enzyme_rows = enzymes_df[enzymes_df['enzyme_name'].str.contains(enzyme_name, case=False, na=False)]
        if enzyme_rows.empty:
            return f"未找到酶名为 '{enzyme_name}' 的相关反应。"
        # 获取所有相关literature_id和reaction_id
        id_pairs = enzyme_rows[['literature_id', 'reaction_id']].drop_duplicates()
        # 合并条件
        df = pd.merge(df, id_pairs, on=['literature_id', 'reaction_id'])
    if literature_id:
        df = df[df['literature_id'] == literature_id]
    if reaction_id:
        df = df[df['reaction_id'] == reaction_id]
    if parameter_type:
        df = df[df['parameter_type'].str.lower() == parameter_type.lower()]
    if df.empty:
        return "未找到匹配的动力学参数数据。"
    
    # 限制结果数量
    total = len(df)
    df = df.head(max_results)
    # 分组展示
    result = f"# 动力学参数查询结果\n\n"
    result += f"**输出记录数**: {len(df)}，共找到记录{total}条\n\n"
    
    group_cols = ['literature_id', 'reaction_id', 'source_type', 'mutation_description']
    grouped = df.groupby(group_cols)
    for group_keys, group_df in grouped:
        lit, rid, src, mut = group_keys
        result += f"## 文献: {lit} 反应: {rid} 类型: {src}"
        if src == "wild_type":
            result += f" 野生型: WT"
        else:
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

def find_mutant_performance(**kwargs) -> str:
    """
    查询突变体的性能表现，支持按酶名、文献ID、反应ID、突变描述等索引。
    整合7_mutants_characterized和2_enzymes表。

    :param enzyme_name: str，可选
    :param literature_id: str，可选
    :param reaction_id: str，可选
    :param mutation_description: str，可选
    :param max_results: int
    """
    enzyme_name = kwargs.get('enzyme_name', None)
    literature_id = kwargs.get('literature_id', None)
    reaction_id = kwargs.get('reaction_id', None)
    mutation_description = kwargs.get('mutation_description', None)
    max_results = kwargs.get('max_results', QUERY_CONFIG["max_results"])
    
    if not DB:
        return "数据库未加载。"
    mutants_df = DB.get('7_mutants_characterized', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    # kinetic_df = DB.get('6_kinetic_parameters', pd.DataFrame())
    if mutants_df.empty or enzymes_df.empty:
        return "突变体或酶信息表未加载。"
    # 合并酶名
    merged_df = pd.merge(mutants_df, enzymes_df, on=['literature_id', 'reaction_id'], how='left')
    # 条件筛选
    if enzyme_name:
        merged_df = merged_df[merged_df['enzyme_name'].str.contains(enzyme_name, case=False, na=False)]
    if literature_id:
        merged_df = merged_df[merged_df['literature_id'] == literature_id]
    if reaction_id:
        merged_df = merged_df[merged_df['reaction_id'] == reaction_id]
    if mutation_description:
        merged_df = merged_df[merged_df['mutation_description'].str.contains(mutation_description, case=False, na=False)]
    if merged_df.empty:
        return "未找到匹配的突变体性能数据。"
    # 限制结果数量
    result_df = merged_df.head(max_results)
    # 格式化输出
    result = f"# 突变体性能表现查询结果\n\n"
    result += f"**筛选条件**: 酶={enzyme_name if enzyme_name else '全部'}, 文献={literature_id if literature_id else '全部'}, 反应={reaction_id if reaction_id else '全部'}\n"
    result += f"**输出记录数**: {len(result_df)} (共找到记录数{len(merged_df)}个)\n\n"
    for _, row in result_df.iterrows():
        result += f"## {row['literature_id']}:{row['reaction_id']} | 酶: {row.get('enzyme_name', 'N/A')} | 突变: {row['mutation_description']}\n"
        result += f"- **定性活性**: {row.get('activity_qualitative', 'N/A')}\n"
        result += f"- **转化率**: {row.get('conversion_rate', 'N/A')} %\n"
        result += f"- **产率**: {row.get('product_yield', 'N/A')} {row.get('product_yield_unit', '')}\n"
        result += f"- **区域选择性**: {row.get('selectivity_regio', 'N/A')}\n"
        result += f"- **立体选择性**: {row.get('selectivity_stereo', 'N/A')}\n"
        result += f"- **对映体过量**: {row.get('enantiomeric_excess', 'N/A')} %\n"
        # # 动力学参数联查
        # if enzyme_name and not kinetic_df.empty:
        #     kin_rows = kinetic_df[(kinetic_df['literature_id'] == row['literature_id']) & (kinetic_df['reaction_id'] == row['reaction_id'])]
        #     if not kin_rows.empty:
        #         result += f"- **动力学参数**:\n"
        #         for _, kin in kin_rows.iterrows():
        #             param = kin.get('parameter_type', 'N/A')
        #             value = kin.get('value', 'N/A')
        #             unit = kin.get('unit', '')
        #             error = kin.get('error_margin', '')
        #             details = kin.get('details', '')
        #             result += f"    - {param}: {value} {unit} {(f'(误差: {error})' if error else '')} {(f'| 说明: {details}' if details else '')}\n"
        result += "\n"
    return result

# --- FunctionTool实例导出 ---
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
find_mutant_performance_tool = FunctionTool(func=find_mutant_performance)

