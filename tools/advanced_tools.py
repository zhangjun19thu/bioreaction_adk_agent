from google.adk.tools import FunctionTool
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from .database_loader import DB
from ..config import ANALYSIS_CONFIG, QUERY_CONFIG
import json

def _safe_numeric_conversion(series: pd.Series) -> pd.Series:
    """安全地将字符串转换为数值类型"""
    return pd.to_numeric(series, errors='coerce')

def _format_trend_analysis(trend_data: Dict, title: str) -> str:
    """格式化趋势分析结果"""
    result = f"## {title}\n\n"
    
    if trend_data.get('trend_type') == 'increasing':
        result += "📈 **上升趋势**\n"
    elif trend_data.get('trend_type') == 'decreasing':
        result += "📉 **下降趋势**\n"
    elif trend_data.get('trend_type') == 'stable':
        result += "➡️ **稳定趋势**\n"
    else:
        result += "📊 **复杂趋势**\n"
    
    if trend_data.get('correlation'):
        result += f"**相关性系数**: {trend_data['correlation']:.3f}\n"
    
    if trend_data.get('key_factors'):
        result += f"**关键影响因素**: {', '.join(trend_data['key_factors'])}\n"
    
    if trend_data.get('recommendations'):
        result += f"**优化建议**: {trend_data['recommendations']}\n"
    
    return result

def analyze_reaction_trends(
    enzyme_name: str,
    organism: str,
    metric: str,
    min_data_points: int
) -> str:
    """
    分析生物化学反应的趋势模式，识别关键影响因素和优化机会。
    
    :param enzyme_name: 可为 None
    :param organism: 可为 None
    :param metric: str
    :param min_data_points: int
    """
    if not DB: return "数据库未加载。"
    
    # 获取相关数据
    activity_df = DB.get('4_activity_performance', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    
    if activity_df.empty or enzymes_df.empty:
        return "核心数据表未加载。"
    
    # 合并数据
    merged_df = pd.merge(activity_df, enzymes_df, on=['literature_id', 'reaction_id'])
    merged_df = pd.merge(merged_df, conditions_df, on=['literature_id', 'reaction_id'], how='left')
    
    # 应用筛选条件
    if enzyme_name:
        merged_df = merged_df[
            merged_df['enzyme_name'].str.contains(enzyme_name, case=False, na=False) |
            merged_df['enzyme_synonyms'].str.contains(enzyme_name, case=False, na=False)
        ]
    
    if organism:
        merged_df = merged_df[merged_df['organism'].str.contains(organism, case=False, na=False)]
    
    if len(merged_df) < min_data_points:
        return f"数据点不足（{len(merged_df)} < {min_data_points}），无法进行可靠的趋势分析。"
    
    # 数值转换
    if metric in ['conversion_rate', 'product_yield']:
        merged_df[metric] = _safe_numeric_conversion(merged_df[metric])
        merged_df = merged_df.dropna(subset=[metric])
    
    # 趋势分析
    trends = {}
    
    # 1. 温度对性能的影响
    if 'temperature_celsius' in merged_df.columns and metric in ['conversion_rate', 'product_yield']:
        temp_perf = merged_df[['temperature_celsius', metric]].dropna()
        if len(temp_perf) >= 3:
            correlation = temp_perf.corr().iloc[0, 1]
            # 使用配置中的相关性阈值
            threshold = ANALYSIS_CONFIG["correlation_threshold"]
            trends['temperature_impact'] = {
                'trend_type': 'increasing' if correlation > threshold else 'decreasing' if correlation < -threshold else 'stable',
                'correlation': correlation,
                'key_factors': ['温度优化', '热稳定性'],
                'recommendations': f"温度与{metric}的相关性为{correlation:.3f}，建议优化温度条件"
            }
    
    # 2. pH对性能的影响
    if 'ph' in merged_df.columns and metric in ['conversion_rate', 'product_yield']:
        ph_perf = merged_df[['ph', metric]].dropna()
        if len(ph_perf) >= 3:
            correlation = ph_perf.corr().iloc[0, 1]
            threshold = ANALYSIS_CONFIG["correlation_threshold"]
            trends['ph_impact'] = {
                'trend_type': 'stable' if abs(correlation) < threshold else ('increasing' if correlation > 0 else 'decreasing'),
                'correlation': correlation,
                'key_factors': ['pH优化', '缓冲液选择'],
                'recommendations': f"pH与{metric}的相关性为{correlation:.3f}，建议调整pH条件"
            }
    
    # 3. 物种间性能对比
    if not organism and 'organism' in merged_df.columns:
        org_perf = merged_df.groupby('organism')[metric].agg(['mean', 'count']).reset_index()
        org_perf = org_perf[org_perf['count'] >= 2].sort_values('mean', ascending=False)
        if len(org_perf) >= 2:
            top_org = org_perf.iloc[0]['organism']
            trends['organism_comparison'] = {
                'trend_type': 'complex',
                'key_factors': [f'最佳物种: {top_org}'],
                'recommendations': f"推荐使用{top_org}作为宿主，平均{metric}为{org_perf.iloc[0]['mean']:.2f}"
            }
    
    # 4. 整体性能分布
    if metric in ['conversion_rate', 'product_yield']:
        values = merged_df[metric].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            trends['performance_distribution'] = {
                'trend_type': 'stable',
                'key_factors': [f'平均{metric}: {mean_val:.2f}', f'标准差: {std_val:.2f}'],
                'recommendations': f"当前{metric}平均值为{mean_val:.2f}，有{len(values[values > mean_val + std_val])}个高表现样本"
            }
    
    # 格式化输出
    result = f"# 生物化学反应趋势分析报告\n\n"
    result += f"**分析范围**: {len(merged_df)} 个反应\n"
    result += f"**主要指标**: {metric}\n"
    if enzyme_name:
        result += f"**目标酶**: {enzyme_name}\n"
    if organism:
        result += f"**目标物种**: {organism}\n"
    result += "\n"
    
    for trend_name, trend_data in trends.items():
        if trend_name == 'temperature_impact':
            result += _format_trend_analysis(trend_data, "温度影响分析")
        elif trend_name == 'ph_impact':
            result += _format_trend_analysis(trend_data, "pH影响分析")
        elif trend_name == 'organism_comparison':
            result += _format_trend_analysis(trend_data, "物种性能对比")
        elif trend_name == 'performance_distribution':
            result += _format_trend_analysis(trend_data, "性能分布分析")
        result += "\n"
    
    return result

def compare_reactions(
    reaction_ids: List[str],
    comparison_metrics: List[str]
) -> str:
    """
    对比多个反应的性能指标和实验条件。
    
    :param reaction_ids: List[str]
    :param comparison_metrics: List[str]
    """
    if not DB: return "数据库未加载。"
    
    if len(reaction_ids) < 2:
        return "至少需要2个反应进行对比分析。"
    
    # 解析反应ID
    parsed_reactions = []
    for reaction_id in reaction_ids:
        if ':' in reaction_id:
            lit_id, react_id = reaction_id.split(':', 1)
            parsed_reactions.append((lit_id, react_id))
        else:
            return f"反应ID格式错误: {reaction_id}，应为 'literature_id:reaction_id'"
    
    # 获取数据
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    activity_df = DB.get('4_activity_performance', pd.DataFrame())
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    
    # 筛选目标反应
    comparison_data = []
    for lit_id, react_id in parsed_reactions:
        reaction_data = {
            'literature_id': lit_id,
            'reaction_id': react_id
        }
        
        # 核心信息
        core_info = core_df[(core_df['literature_id'] == lit_id) & (core_df['reaction_id'] == react_id)]
        if not core_info.empty:
            reaction_data.update(core_info.iloc[0].to_dict())
        
        # 酶信息
        enzyme_info = enzymes_df[(enzymes_df['literature_id'] == lit_id) & (enzymes_df['reaction_id'] == react_id)]
        if not enzyme_info.empty:
            reaction_data.update(enzyme_info.iloc[0].to_dict())
        
        # 性能信息
        activity_info = activity_df[(activity_df['literature_id'] == lit_id) & (activity_df['reaction_id'] == react_id)]
        if not activity_info.empty:
            reaction_data.update(activity_info.iloc[0].to_dict())
        
        # 条件信息
        condition_info = conditions_df[(conditions_df['literature_id'] == lit_id) & (conditions_df['reaction_id'] == react_id)]
        if not condition_info.empty:
            reaction_data.update(condition_info.iloc[0].to_dict())
        
        comparison_data.append(reaction_data)
    
    if not comparison_data:
        return "未找到指定的反应数据。"
    
    # 生成对比报告
    result = "# 反应对比分析报告\n\n"
    
    # 基本信息对比
    result += "## 基本信息对比\n\n"
    result += "| 反应ID | 酶名称 | 物种 | 反应方程式 |\n"
    result += "|--------|--------|------|------------|\n"
    for data in comparison_data:
        result += f"| {data.get('literature_id', '')}:{data.get('reaction_id', '')} | "
        result += f"{data.get('enzyme_name', 'N/A')} | "
        result += f"{data.get('organism', 'N/A')} | "
        result += f"{data.get('reaction_equation', 'N/A')} |\n"
    result += "\n"
    
    # 性能指标对比
    result += "## 性能指标对比\n\n"
    result += "| 反应ID | 转化率 | 产率 | 温度(°C) | pH |\n"
    result += "|--------|--------|------|----------|----|\n"
    for data in comparison_data:
        # 转化率
        cr = data.get('conversion_rate', 'N/A')
        cr_unit = data.get('conversion_rate_unit', '')
        cr_error = data.get('conversion_rate_error', '')
        cr_str = f"{cr} {cr_unit} {(f'(误差: {cr_error})' if cr_error else '')}" if cr != 'N/A' else 'N/A'
        # 产率
        py = data.get('product_yield', 'N/A')
        py_unit = data.get('product_yield_unit', '')
        py_error = data.get('product_yield_error', '')
        py_str = f"{py} {py_unit} {(f'(误差: {py_error})' if py_error else '')}" if py != 'N/A' else 'N/A'
        result += f"| {data.get('literature_id', '')}:{data.get('reaction_id', '')} | "
        result += f"{cr_str} | "
        result += f"{py_str} | "
        result += f"{data.get('temperature_celsius', 'N/A')} | "
        result += f"{data.get('ph', 'N/A')} |\n"
        result += "\n"
    
    # 关键差异分析
    result += "## 关键差异分析\n\n"
    
    # 转化率对比
    print(comparison_data)
    conversion_rates = [
        float(data.get('conversion_rate')) for data in comparison_data
        if data.get('conversion_rate') not in [None, '', 'N/A']
        and str(data.get('conversion_rate')).replace('.', '', 1).isdigit()
    ]
    if len(conversion_rates) >= 2:
        max_rate = max(conversion_rates)
        min_rate = min(conversion_rates)
        # 用类型一致的 float 比较
        best_reaction = next(
            (
                data for data in comparison_data
                if (
                    data.get('conversion_rate') not in [None, '', 'N/A'] and
                    abs(float(data.get('conversion_rate')) - max_rate) < 1e-8
                )
            ),
            None
        )
        if best_reaction:
            result += f"**转化率差异**: 最高 {max_rate}，最低 {min_rate}\n"
            result += f"**最佳反应**: {best_reaction.get('literature_id')}:{best_reaction.get('reaction_id')}\n"
            result += f"**关键因素**: 酶({best_reaction.get('enzyme_name')})，物种({best_reaction.get('organism')})\n\n"
        else:
            result += f"**转化率差异**: 最高 {max_rate}，最低 {min_rate}\n"
            result += f"**未能定位到最佳反应的详细信息**\n\n"
            
    # 条件差异
    temperatures = [data.get('temperature_celsius') for data in comparison_data if data.get('temperature_celsius')]
    if len(temperatures) >= 2:
        temp_range = max(temperatures) - min(temperatures)
        result += f"**温度范围**: {min(temperatures)} - {max(temperatures)}°C (差异: {temp_range}°C)\n"
    
    phs = [data.get('ph') for data in comparison_data if data.get('ph')]
    if len(phs) >= 2:
        ph_range = max(phs) - min(phs)
        result += f"**pH范围**: {min(phs)} - {max(phs)} (差异: {ph_range})\n\n"
    
    return result

def suggest_optimization(
    literature_id: str,
    reaction_id: str,
    target_metric: str,
    optimization_type: str
) -> str:
    """
    基于现有数据和相似反应，为特定反应提供优化建议。
    
    :param literature_id: str
    :param reaction_id: str
    :param target_metric: str
    :param optimization_type: str
    """
    if not DB: return "数据库未加载。"
    
    # 获取目标反应信息
    core_df = DB.get('1_reactions_core', pd.DataFrame())
    enzymes_df = DB.get('2_enzymes', pd.DataFrame())
    activity_df = DB.get('4_activity_performance', pd.DataFrame())
    conditions_df = DB.get('3_experimental_conditions', pd.DataFrame())
    
    target_reaction = core_df[(core_df['literature_id'] == literature_id) & (core_df['reaction_id'] == reaction_id)]
    if target_reaction.empty:
        return f"未找到反应 {literature_id}:{reaction_id}"
    
    target_enzyme = enzymes_df[(enzymes_df['literature_id'] == literature_id) & (enzymes_df['reaction_id'] == reaction_id)]
    target_activity = activity_df[(activity_df['literature_id'] == literature_id) & (activity_df['reaction_id'] == reaction_id)]
    target_condition = conditions_df[(conditions_df['literature_id'] == literature_id) & (conditions_df['reaction_id'] == reaction_id)]
    
    # 获取当前性能
    current_performance = target_activity[target_metric].iloc[0] if not target_activity.empty else None
    
    result = f"# 反应优化建议报告\n\n"
    result += f"**目标反应**: {literature_id}:{reaction_id}\n"
    result += f"**优化目标**: {target_metric}\n"
    result += f"**当前性能**: {current_performance}\n\n"
    
    # 基于优化类型提供建议
    if optimization_type == 'condition':
        result += _suggest_condition_optimization(target_condition, target_metric, activity_df, conditions_df)
    elif optimization_type == 'enzyme':
        result += _suggest_enzyme_optimization(target_enzyme, target_metric, enzymes_df, activity_df)
    elif optimization_type == 'organism':
        result += _suggest_organism_optimization(target_enzyme, target_metric, enzymes_df, activity_df)
    else:
        result += "不支持的优化类型。"
    
    return result

def _suggest_condition_optimization(target_condition, target_metric, activity_df, conditions_df):
    """提供条件优化建议"""
    result = "## 实验条件优化建议\n\n"
    
    # 温度优化
    if 'temperature_celsius' in target_condition.columns:
        current_temp = target_condition['temperature_celsius'].iloc[0]
        # 使用配置中的温度范围
        temp_ranges = ANALYSIS_CONFIG["temperature_ranges"]
        room_temp_range = temp_ranges["room"]
        similar_conditions = conditions_df[
            (conditions_df['temperature_celsius'].between(room_temp_range[0], room_temp_range[1])) &
            (conditions_df['temperature_celsius'] != current_temp)
        ]
        
        if not similar_conditions.empty:
            merged = pd.merge(similar_conditions, activity_df, on=['literature_id', 'reaction_id'])
            if target_metric in merged.columns:
                best_temp = merged.loc[merged[target_metric].idxmax(), 'temperature_celsius']
                result += f"**温度建议**: 当前 {current_temp}°C，建议尝试 {best_temp}°C\n"
    
    # pH优化
    if 'ph' in target_condition.columns:
        current_ph = target_condition['ph'].iloc[0]
        # 使用配置中的pH范围
        ph_ranges = ANALYSIS_CONFIG["ph_ranges"]
        neutral_range = ph_ranges["neutral_weak_basic"]
        similar_ph = conditions_df[
            (conditions_df['ph'].between(neutral_range[0], neutral_range[1])) &
            (conditions_df['ph'] != current_ph)
        ]
        
        if not similar_ph.empty:
            merged = pd.merge(similar_ph, activity_df, on=['literature_id', 'reaction_id'])
            if target_metric in merged.columns:
                best_ph = merged.loc[merged[target_metric].idxmax(), 'ph']
                result += f"**pH建议**: 当前 {current_ph}，建议尝试 {best_ph}\n"
    
    return result

def _suggest_enzyme_optimization(target_enzyme, target_metric, enzymes_df, activity_df):
    """提供酶优化建议"""
    result = "## 酶优化建议\n\n"
    
    if not target_enzyme.empty:
        current_enzyme = target_enzyme['enzyme_name'].iloc[0]
        similar_enzymes = enzymes_df[
            enzymes_df['enzyme_name'].str.contains(current_enzyme.split('_')[0], case=False, na=False)
        ]
        
        if not similar_enzymes.empty:
            merged = pd.merge(similar_enzymes, activity_df, on=['literature_id', 'reaction_id'])
            if target_metric in merged.columns:
                best_enzyme = merged.loc[merged[target_metric].idxmax(), 'enzyme_name']
                result += f"**酶建议**: 当前 {current_enzyme}，建议尝试 {best_enzyme}\n"
    
    return result

def _suggest_organism_optimization(target_enzyme, target_metric, enzymes_df, activity_df):
    """提供物种优化建议"""
    result = "## 物种优化建议\n\n"
    
    if not target_enzyme.empty:
        current_organism = target_enzyme['organism'].iloc[0]
        similar_organisms = enzymes_df[
            enzymes_df['organism'].str.contains(current_organism.split()[0], case=False, na=False)
        ]
        
        if not similar_organisms.empty:
            merged = pd.merge(similar_organisms, activity_df, on=['literature_id', 'reaction_id'])
            if target_metric in merged.columns:
                best_organism = merged.loc[merged[target_metric].idxmax(), 'organism']
                result += f"**物种建议**: 当前 {current_organism}，建议尝试 {best_organism}\n"
    
    return result


# --- 创建FunctionTool实例 ---
analyze_reaction_trends_tool = FunctionTool(func=analyze_reaction_trends)
compare_reactions_tool = FunctionTool(func=compare_reactions)
suggest_optimization_tool = FunctionTool(func=suggest_optimization)
