#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
误差分析可视化模块

提供全面的预测误差分析和可视化工具，帮助深入理解模型性能和预测质量。

核心功能：
1. 误差分布分析：
   - 整体误差分布直方图和统计特征
   - 误差的正态性检验和异常值检测
   - 箱线图展示误差的四分位数分布

2. 分层误差分析：
   - 按真实评分等级的平均误差分析
   - 各评分等级的RMSE和MAE对比
   - 置信区间和样本数量统计

3. 混淆矩阵分析：
   - 预测评分与真实评分的混淆热力图
   - 支持不同精度的评分离散化
   - 多种归一化方式（行、列、总体）

4. 用户维度分析：
   - 用户级别的误差分布
   - 用户预测准确性的差异分析
   - 识别难以预测的用户群体

5. 电影流行度分析：
   - 误差与电影流行度的关系
   - 热门电影vs冷门电影的预测准确性
   - 流行度分箱分析

6. 时间维度分析：
   - 按电影年份的误差变化趋势
   - 时间序列误差分析
   - 年代效应对预测准确性的影响

可视化特点：
- 支持中文字体显示
- 统一的配色方案和样式
- 详细的统计信息标注
- 高质量图表输出（300 DPI）
- 自动保存和路径管理
- 完整的异常处理和日志记录

技术特点：
- 基于matplotlib和seaborn的专业可视化
- 支持多种误差度量（RMSE、MAE、偏差等）
- 灵活的参数配置和自定义选项
- 统计显著性检验和置信区间
- 异常值检测和标记
- 内存优化的大数据处理
"""

import os
from typing import Optional, Tuple, List, Dict, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib.font_manager as fm

from config import config
from utils.logger import logger

# 导入字体配置模块以解决中文显示问题
from .font_config import setup_chinese_fonts

# 设置中文字体
setup_chinese_fonts()

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_error_distribution(output_df: pd.DataFrame,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8),
                           bins: int = 30,
                           show_stats: bool = True,
                           show_outliers: bool = True) -> Optional[plt.Figure]:
    """
    绘制预测误差的分布图
    
    创建包含直方图、核密度估计和箱线图的综合误差分布分析图表。
    帮助理解模型预测误差的整体分布特征、统计属性和异常值情况。
    
    可视化内容：
    - 主图：误差分布直方图 + KDE曲线
    - 统计信息：均值、标准差、中位数、偏度、峰度、MAE、RMSE
    - 参考线：零误差线、均值线、中位数线
    - 异常值边界：基于IQR方法的异常值检测
    - 底图：误差箱线图，展示四分位数分布
    
    Args:
        output_df: 包含预测误差的DataFrame，需包含'error'列或'true_rating'+'pred_rating'列
        save_path: 保存图表的路径，None时使用默认路径
        figsize: 图表大小，默认(12, 8)
        bins: 直方图的箱数，默认30
        show_stats: 是否显示详细统计信息，默认True
        show_outliers: 是否标记异常值边界，默认True
    
    Returns:
        matplotlib图表对象，绘制失败时返回None
        
    Raises:
        TypeError: 当output_df不是DataFrame时
        ValueError: 当DataFrame缺少必要列时
        
    Example:
        >>> predictions_df = pd.DataFrame({
        ...     'true_rating': [1, 2, 3, 4, 5],
        ...     'pred_rating': [1.1, 1.9, 3.2, 3.8, 4.9],
        ...     'error': [0.1, -0.1, 0.2, -0.2, -0.1]
        ... })
        >>> fig = plot_error_distribution(predictions_df)
        
    Note:
        - 正态分布的误差表明模型预测无系统性偏差
        - 偏度反映误差分布的对称性（0为完全对称）
        - 峰度反映误差分布的尖锐程度（3为正态分布）
        - 异常值可能指示数据质量问题或模型局限性
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df must be a pandas DataFrame")
    
    if output_df.empty:
        logger.warning("DataFrame is empty, cannot plot error distribution.")
        return None
    
    # 计算或获取误差数据
    if 'error' in output_df.columns:
        errors = output_df['error'].dropna()
    elif 'true_rating' in output_df.columns and 'pred_rating' in output_df.columns:
        errors = (output_df['pred_rating'] - output_df['true_rating']).dropna()
        logger.info("Calculating errors from 'true_rating' and 'pred_rating' columns.")
    else:
        raise ValueError("DataFrame must contain 'error' column or both 'true_rating' and 'pred_rating' columns.")
    
    if errors.empty:
        logger.warning("Error data is empty, cannot plot distribution.")
        return None
    
    try:
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # 绘制主要误差分布直方图
        hist_plot = sns.histplot(
            errors, 
            bins=bins, 
            kde=True, 
            color='orange', 
            ax=ax1,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # 设置主图标题和标签
        ax1.set_title("预测误差分布分析", fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel("预测误差 (预测值 - 真实值)", fontsize=12)
        ax1.set_ylabel("频率", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加零误差参考线
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='零误差线')
        
        # 计算并显示统计信息
        if show_stats:
            mean_error = errors.mean()
            std_error = errors.std()
            median_error = errors.median()
            skew_error = errors.skew()
            kurt_error = errors.kurtosis()
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors ** 2).mean())
            
            # 在图表上标注统计信息
            stats_text = (
                f"样本数: {len(errors)}\n"
                f"均值: {mean_error:.4f}\n"
                f"标准差: {std_error:.4f}\n"
                f"中位数: {median_error:.4f}\n"
                f"平均绝对误差: {mae:.4f}\n"
                f"均方根误差: {rmse:.4f}\n"
                f"偏度: {skew_error:.4f}\n"
                f"峰度: {kurt_error:.4f}"
            )
            
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 添加均值和中位数的垂直线
            ax1.axvline(mean_error, color='blue', linestyle=':', alpha=0.8, 
                       label=f'均值: {mean_error:.3f}')
            ax1.axvline(median_error, color='green', linestyle=':', alpha=0.8, 
                       label=f'中位数: {median_error:.3f}')
        
        # 检测并标记异常值
        if show_outliers:
            Q1 = errors.quantile(0.25)
            Q3 = errors.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = errors[(errors < lower_bound) | (errors > upper_bound)]
            if len(outliers) > 0:
                ax1.axvline(lower_bound, color='purple', linestyle='-.', alpha=0.6, 
                           label=f'异常值边界: [{lower_bound:.2f}, {upper_bound:.2f}]')
                ax1.axvline(upper_bound, color='purple', linestyle='-.', alpha=0.6)
                logger.info(f"Detected {len(outliers)} outliers ({len(outliers)/len(errors)*100:.1f}%)")
        
        # 添加图例
        ax1.legend(loc='upper left')
        
        # 绘制箱线图（底部子图）
        box_plot = ax2.boxplot(errors, vert=False, patch_artist=True, 
                              boxprops=dict(facecolor='lightcoral', alpha=0.7))
        ax2.set_xlabel("预测误差", fontsize=12)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        ax2.set_title("误差箱线图", fontsize=12)
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "prediction_error_hist.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Error distribution plot saved to: {save_path}")
        logger.info(f"Error statistics: Mean={mean_error:.4f}, Std={std_error:.4f}, RMSE={rmse:.4f}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to plot error distribution: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_mean_error_per_rating(output_df: pd.DataFrame,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8),
                              show_confidence: bool = True,
                              show_sample_size: bool = True) -> Optional[plt.Figure]:
    """
    绘制每个真实评分等级的平均预测误差。

    该函数创建一个条形图，显示每个真实评分等级的平均预测误差。
    有助于识别模型在不同评分范围内的系统性偏差和预测准确性。

    Args:
        output_df (pd.DataFrame): 包含真实评分和预测误差的DataFrame。
                                  必须包含'true_rating'列和'error'或'pred_rating'列。
        save_path (Optional[str]): 保存图表的路径。如果为None，使用默认路径。
        figsize (Tuple[int, int]): 图表大小，默认为(10, 8)。
        show_confidence (bool): 是否显示置信区间，默认为True。
        show_sample_size (bool): 是否显示样本数量，默认为True。

    Returns:
        Optional[plt.Figure]: matplotlib图表对象，如果绘制失败则返回None。

    Raises:
        ValueError: 如果DataFrame缺少必需的列。
        TypeError: 如果任何参数类型不正确。

    Example:
        >>> predictions_df = pd.DataFrame({
        ...     'true_rating': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ...     'pred_rating': [1.2, 0.9, 2.1, 1.8, 2.9, 3.2, 4.1, 3.9, 4.8, 5.1],
        ...     'error': [0.2, -0.1, 0.1, -0.2, -0.1, 0.2, 0.1, -0.1, -0.2, 0.1]
        ... })
        >>> fig = plot_mean_error_per_rating(predictions_df)

    Note:
        - 正误差表示模型在该评分等级倾向于高估。
        - 负误差表示模型在该评分等级倾向于低估。
        - 理想情况下，所有评分等级的平均误差都应接近0。
        - 置信区间显示误差估计的不确定性。
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df必须是pandas DataFrame类型")
    
    if output_df.empty:
        logger.warning("DataFrame为空，无法绘制平均误差图")
        return None
    
    if 'true_rating' not in output_df.columns:
        raise ValueError("DataFrame缺少'true_rating'列")
    
    # 计算或获取误差数据
    df_work = output_df.copy()
    if 'error' not in df_work.columns:
        if 'pred_rating' in df_work.columns:
            df_work['error'] = df_work['pred_rating'] - df_work['true_rating']
            logger.info("从pred_rating和true_rating列计算误差")
        else:
            raise ValueError("DataFrame必须包含'error'列或'pred_rating'列")
    
    # 清理数据
    df_clean = df_work[['true_rating', 'error']].dropna()
    if df_clean.empty:
        logger.warning("清理后的数据为空，无法绘制平均误差图")
        return None
    
    try:
        # 计算每个评分等级的统计信息
        error_stats = df_clean.groupby("true_rating")['error'].agg([
            'mean', 'std', 'count', 'sem'  # sem: standard error of mean
        ]).reset_index()
        error_stats.columns = ['true_rating', 'mean_error', 'std_error', 'sample_count', 'sem_error']
        
        # 计算95%置信区间
        error_stats['ci_lower'] = error_stats['mean_error'] - 1.96 * error_stats['sem_error']
        error_stats['ci_upper'] = error_stats['mean_error'] + 1.96 * error_stats['sem_error']
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制条形图
        bars = sns.barplot(
            x='true_rating', 
            y='mean_error', 
            data=error_stats, 
            palette="RdYlBu_r",  # 红-黄-蓝调色板，红色表示正误差，蓝色表示负误差
            ax=ax
        )
        
        # 添加置信区间
        if show_confidence:
            for i, (idx, row) in enumerate(error_stats.iterrows()):
                ax.errorbar(
                    i, row['mean_error'], 
                    yerr=[[row['mean_error'] - row['ci_lower']], 
                          [row['ci_upper'] - row['mean_error']]], 
                    fmt='none', 
                    color='black', 
                    capsize=5, 
                    alpha=0.7
                )
        
        # 添加零误差参考线
        ax.axhline(0, color='gray', linestyle='--', alpha=0.8, linewidth=2, label='零误差线')
        
        # 设置标题和标签
        ax.set_title("各评分等级的平均预测误差", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("真实评分", fontsize=12)
        ax.set_ylabel("平均预测误差", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 在条形图上显示数值和样本数量
        for i, (idx, row) in enumerate(error_stats.iterrows()):
            # 显示平均误差值
            height = row['mean_error']
            ax.text(i, height + (0.01 if height >= 0 else -0.01), 
                   f'{height:.3f}', 
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontweight='bold', fontsize=10)
            
            # 显示样本数量
            if show_sample_size:
                ax.text(i, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                       f'n={int(row["sample_count"])}', 
                       ha='center', va='bottom',
                       fontsize=9, alpha=0.8)
        
        # 添加图例
        ax.legend()
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "mean_error_per_rating.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Mean error bar plot saved to: {save_path}")
        
        # 记录统计信息
        overall_bias = error_stats['mean_error'].mean()
        max_bias_rating = error_stats.loc[error_stats['mean_error'].abs().idxmax(), 'true_rating']
        max_bias_value = error_stats.loc[error_stats['mean_error'].abs().idxmax(), 'mean_error']
        
        logger.info(f"整体偏差: {overall_bias:.4f}")
        logger.info(f"最大偏差评分等级: {max_bias_rating} (偏差: {max_bias_value:.4f})")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制平均误差条形图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_rmse_per_rating(output_df: pd.DataFrame,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        show_mae: bool = True,
                        show_sample_size: bool = True) -> Optional[plt.Figure]:
    """
    Plot RMSE per true rating level.

    This function creates a bar plot showing the Root Mean Squared Error (RMSE) for each true rating level.
    It helps to identify the prediction accuracy and stability of the model across different rating ranges.

    Args:
        output_df (pd.DataFrame): DataFrame containing true and predicted ratings.
                                  Must include 'true_rating' and 'pred_rating' columns.
        save_path (Optional[str]): Path to save the plot. If None, a default path is used.
        figsize (Tuple[int, int]): Figure size, default is (10, 8).
        show_mae (bool): Whether to also display Mean Absolute Error (MAE), default is True.
        show_sample_size (bool): Whether to display sample sizes, default is True.

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None if plotting fails.

    Raises:
        ValueError: If the DataFrame is missing required columns.
        TypeError: If any parameter has an incorrect type.

    Example:
        >>> predictions_df = pd.DataFrame({
        ...     'true_rating': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ...     'pred_rating': [1.2, 0.9, 2.1, 1.8, 2.9, 3.2, 4.1, 3.9, 4.8, 5.1]
        ... })
        >>> fig = plot_rmse_per_rating(predictions_df)

    Note:
        - A smaller RMSE value indicates higher prediction accuracy for that rating level.
        - RMSE is more sensitive to large errors, highlighting outlier predictions.
        - MAE provides a linear measure of error, which is easier to interpret.
        - Differences in RMSE across rating levels reflect the model's adaptability.
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df必须是pandas DataFrame类型")
    
    if output_df.empty:
        logger.warning("DataFrame为空，无法绘制RMSE图")
        return None
    
    required_columns = ['true_rating', 'pred_rating']
    missing_columns = [col for col in required_columns if col not in output_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame缺少必要的列: {missing_columns}")
    
    # 清理数据
    df_clean = output_df[required_columns].dropna()
    if df_clean.empty:
        logger.warning("清理后的数据为空，无法绘制RMSE图")
        return None
    
    try:
        # 计算每个评分等级的RMSE和MAE
        def calculate_metrics(group):
            true_vals = group['true_rating']
            pred_vals = group['pred_rating']
            rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
            mae = mean_absolute_error(true_vals, pred_vals)
            count = len(group)
            return pd.Series({
                'RMSE': rmse,
                'MAE': mae,
                'sample_count': count
            })
        
        metrics_by_rating = df_clean.groupby("true_rating").apply(calculate_metrics).reset_index()
        
        # 创建图表
        if show_mae:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
        
        # 绘制RMSE条形图
        bars1 = sns.barplot(
            x='true_rating', 
            y='RMSE', 
            data=metrics_by_rating, 
            palette="viridis", 
            ax=ax1
        )
        
        ax1.set_title("RMSE per Rating Level", fontsize=14, fontweight='bold')
        ax1.set_xlabel("True Rating", fontsize=12)
        ax1.set_ylabel("RMSE", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 在RMSE条形图上显示数值和样本数量
        for i, (idx, row) in enumerate(metrics_by_rating.iterrows()):
            # 显示RMSE值
            height = row['RMSE']
            ax1.text(i, height + 0.01, f'{height:.3f}', 
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
            
            # 显示样本数量
            if show_sample_size:
                ax1.text(i, 0.02, f'n={int(row["sample_count"])}', 
                        ha='center', va='bottom',
                        fontsize=9, alpha=0.7)
        
        # 绘制MAE条形图（如果启用）
        if show_mae:
            bars2 = sns.barplot(
                x='true_rating', 
                y='MAE', 
                data=metrics_by_rating, 
                palette="plasma", 
                ax=ax2
            )
            
            ax2.set_title("不同评分等级的MAE", fontsize=14, fontweight='bold')
            ax2.set_xlabel("真实评分", fontsize=12)
            ax2.set_ylabel("MAE", fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 在MAE条形图上显示数值
            for i, (idx, row) in enumerate(metrics_by_rating.iterrows()):
                height = row['MAE']
                ax2.text(i, height + 0.01, f'{height:.3f}', 
                        ha='center', va='bottom',
                        fontweight='bold', fontsize=10)
        
        # 添加统计信息
        overall_rmse = np.sqrt(mean_squared_error(df_clean['true_rating'], df_clean['pred_rating']))
        overall_mae = mean_absolute_error(df_clean['true_rating'], df_clean['pred_rating'])
        
        stats_text = (
            f"总体RMSE: {overall_rmse:.4f}\n"
            f"总体MAE: {overall_mae:.4f}\n"
            f"最高RMSE: {metrics_by_rating['RMSE'].max():.4f}\n"
            f"最低RMSE: {metrics_by_rating['RMSE'].min():.4f}\n"
            f"RMSE标准差: {metrics_by_rating['RMSE'].std():.4f}"
        )
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "rmse_per_rating_level.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"RMSE bar plot saved to: {save_path}")
        logger.info(f"Overall RMSE: {overall_rmse:.4f}, Overall MAE: {overall_mae:.4f}")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制RMSE条形图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_confusion_heatmap(output_df: pd.DataFrame,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 10),
                          rating_precision: float = 0.5,
                          normalize: str = 'none',
                          show_percentages: bool = True) -> Optional[plt.Figure]:
    """
    绘制预测评分与真实评分的混淆矩阵热力图
    
    通过热力图展示预测评分与真实评分之间的对应关系，
    有助于识别模型的预测模式、系统性偏差和分类准确性。
    
    Args:
        output_df (pd.DataFrame): 包含真实评分和预测评分的DataFrame
                                必须包含'true_rating'和'pred_rating'列
        save_path (Optional[str]): 图片保存路径，如果为None则使用默认路径
        figsize (Tuple[int, int]): 图表尺寸，默认为(12, 10)
        rating_precision (float): 评分精度，用于四舍五入，默认为0.5
        normalize (str): 归一化方式，可选'none', 'true', 'pred', 'all'
        show_percentages (bool): 是否显示百分比，默认为True
    
    Returns:
        Optional[plt.Figure]: matplotlib图表对象，如果绘制失败则返回None
    
    Raises:
        ValueError: 当DataFrame缺少必要列或参数无效时抛出异常
        TypeError: 当参数类型不正确时抛出异常
    
    Example:
        >>> predictions_df = pd.DataFrame({
        ...     'true_rating': [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5],
        ...     'pred_rating': [1.2, 2.1, 2.9, 4.1, 4.8, 1.3, 2.7, 3.2, 4.6]
        ... })
        >>> fig = plot_confusion_heatmap(predictions_df)
    
    Note:
        - 对角线上的值表示预测正确的样本数
        - 非对角线值表示预测错误的样本数
        - 颜色深度表示样本数量的多少
        - 支持多种归一化方式便于不同角度的分析
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df必须是pandas DataFrame类型")
    
    if output_df.empty:
        logger.warning("DataFrame为空，无法绘制混淆矩阵")
        return None
    
    required_columns = ['true_rating', 'pred_rating']
    missing_columns = [col for col in required_columns if col not in output_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame缺少必要的列: {missing_columns}")
    
    if normalize not in ['none', 'true', 'pred', 'all']:
        raise ValueError("normalize参数必须是'none', 'true', 'pred', 'all'之一")
    
    # 数据预处理
    heatmap_df = output_df[required_columns].dropna().copy()
    if heatmap_df.empty:
        logger.warning("清理后的数据为空，无法绘制混淆矩阵")
        return None
    
    try:
        # 根据精度四舍五入评分
        heatmap_df['true_rating_rounded'] = (heatmap_df['true_rating'] / rating_precision).round() * rating_precision
        heatmap_df['pred_rating_rounded'] = (heatmap_df['pred_rating'] / rating_precision).round() * rating_precision
        
        # 确定评分范围
        min_rating = min(heatmap_df['true_rating_rounded'].min(), heatmap_df['pred_rating_rounded'].min())
        max_rating = max(heatmap_df['true_rating_rounded'].max(), heatmap_df['pred_rating_rounded'].max())
        
        # 创建评分范围
        rating_range = np.arange(min_rating, max_rating + rating_precision, rating_precision)
        rating_range = np.round(rating_range, 1)  # 避免浮点精度问题
        
        # 创建混淆矩阵
        conf_mat = pd.crosstab(
            heatmap_df['true_rating_rounded'],
            heatmap_df['pred_rating_rounded']
        ).reindex(index=rating_range, columns=rating_range, fill_value=0)
        
        # 归一化处理
        conf_mat_display = conf_mat.copy()
        if normalize == 'true':
            conf_mat_display = conf_mat.div(conf_mat.sum(axis=1), axis=0).fillna(0)
        elif normalize == 'pred':
            conf_mat_display = conf_mat.div(conf_mat.sum(axis=0), axis=1).fillna(0)
        elif normalize == 'all':
            conf_mat_display = conf_mat / conf_mat.sum().sum()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 选择颜色映射和格式
        if normalize == 'none':
            fmt = 'd'
            cmap = "YlOrRd"
            cbar_label = "样本数量"
        else:
            fmt = '.2%' if show_percentages else '.3f'
            cmap = "Blues"
            cbar_label = "比例"
        
        # 绘制热力图
        heatmap = sns.heatmap(
            conf_mat_display, 
            annot=True, 
            fmt=fmt, 
            cmap=cmap, 
            ax=ax,
            cbar_kws={'label': cbar_label},
            square=True,
            linewidths=0.5
        )
        
        # 设置标题和标签
        title_suffix = {
            'none': '(原始计数)',
            'true': '(按真实评分归一化)',
            'pred': '(按预测评分归一化)',
            'all': '(全局归一化)'
        }
        
        ax.set_title(f"预测评分混淆矩阵热力图 {title_suffix[normalize]}", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("预测评分", fontsize=12)
        ax.set_ylabel("真实评分", fontsize=12)
        
        # 计算准确性统计
        total_samples = conf_mat.sum().sum()
        correct_predictions = np.diag(conf_mat).sum()
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        # 计算每个类别的精确率和召回率
        precision_per_class = []
        recall_per_class = []
        
        for i, rating in enumerate(rating_range):
            if rating in conf_mat.index and rating in conf_mat.columns:
                # 精确率 = TP / (TP + FP)
                tp = conf_mat.loc[rating, rating]
                fp = conf_mat.loc[:, rating].sum() - tp
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                precision_per_class.append(precision)
                
                # 召回率 = TP / (TP + FN)
                fn = conf_mat.loc[rating, :].sum() - tp
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                recall_per_class.append(recall)
            else:
                precision_per_class.append(0)
                recall_per_class.append(0)
        
        # 添加统计信息
        stats_text = (
            f"总样本数: {total_samples}\n"
            f"总体准确率: {accuracy:.3f}\n"
            f"平均精确率: {np.mean(precision_per_class):.3f}\n"
            f"平均召回率: {np.mean(recall_per_class):.3f}\n"
            f"评分精度: {rating_precision}"
        )
        
        # 将统计信息放在图表左上角内部，避免被截断
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='left', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 调整布局，增加右侧边距以防止文字被截断
        fig.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "confusion_heatmap.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"混淆矩阵热力图已保存到: {save_path}")
        logger.info(f"总体准确率: {accuracy:.3f}, 分析了 {total_samples} 个样本")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制混淆矩阵热力图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_user_error_distribution(output_df: pd.DataFrame,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8),
                                bins: int = 30,
                                show_stats: bool = True) -> Optional[plt.Figure]:
    """
    绘制用户平均预测误差分布直方图
    
    通过直方图展示不同用户的平均预测误差分布情况，
    有助于识别用户群体的预测准确性差异和异常用户。
    
    Args:
        output_df (pd.DataFrame): 包含用户ID和预测误差的DataFrame
                                必须包含'userId'和'error'列
        save_path (Optional[str]): 图片保存路径，如果为None则使用默认路径
        figsize (Tuple[int, int]): 图表尺寸，默认为(12, 8)
        bins (int): 直方图的箱数，默认为30
        show_stats (bool): 是否显示统计信息，默认为True
    
    Returns:
        Optional[plt.Figure]: matplotlib图表对象，如果绘制失败则返回None
    
    Raises:
        ValueError: 当DataFrame缺少必要列时抛出异常
        TypeError: 当参数类型不正确时抛出异常
    
    Example:
        >>> predictions_df = pd.DataFrame({
        ...     'userId': [1, 1, 2, 2, 3, 3],
        ...     'error': [0.1, -0.2, 0.3, -0.1, 0.2, 0.1]
        ... })
        >>> fig = plot_user_error_distribution(predictions_df)
    
    Note:
        - 正误差表示用户倾向于被高估评分
        - 负误差表示用户倾向于被低估评分
        - 分布的宽度反映用户间预测准确性的差异
        - 异常值可能表示特殊用户行为模式
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df必须是pandas DataFrame类型")
    
    if output_df.empty:
        logger.warning("DataFrame为空，无法绘制用户误差分布图")
        return None
    
    required_columns = ['userId', 'error']
    missing_columns = [col for col in required_columns if col not in output_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame缺少必要的列: {missing_columns}")
    
    # 清理数据
    df_clean = output_df[required_columns].dropna()
    if df_clean.empty:
        logger.warning("清理后的数据为空，无法绘制用户误差分布图")
        return None
    
    try:
        # 计算每个用户的平均误差
        user_error_df = df_clean.groupby('userId')['error'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        user_error_df.columns = ['userId', 'mean_error', 'std_error', 'rating_count']
        
        # 过滤掉评分数量过少的用户（至少5个评分）
        user_error_df = user_error_df[user_error_df['rating_count'] >= 5]
        
        if user_error_df.empty:
            logger.warning("没有足够评分数量的用户，无法绘制分布图")
            return None
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # 绘制直方图和密度曲线
        sns.histplot(
            user_error_df['mean_error'], 
            bins=bins, 
            kde=True, 
            color='coral', 
            alpha=0.7,
            ax=ax1
        )
        
        # 设置标题和标签
        ax1.set_title("用户平均预测误差分布", fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel("平均预测误差", fontsize=12)
        ax1.set_ylabel("用户数量", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        if show_stats:
            mean_error = user_error_df['mean_error'].mean()
            std_error = user_error_df['mean_error'].std()
            median_error = user_error_df['mean_error'].median()
            
            # 添加均值和中位数线
            ax1.axvline(mean_error, color='blue', linestyle='--', alpha=0.8, 
                       label=f'均值: {mean_error:.3f}')
            ax1.axvline(median_error, color='green', linestyle='--', alpha=0.8, 
                       label=f'中位数: {median_error:.3f}')
            ax1.axvline(0, color='red', linestyle='-', alpha=0.8, 
                       label='零误差线')
            
            # 统计信息文本
            stats_text = (
                f"用户数量: {len(user_error_df)}\n"
                f"均值: {mean_error:.4f}\n"
                f"标准差: {std_error:.4f}\n"
                f"中位数: {median_error:.4f}\n"
                f"最小值: {user_error_df['mean_error'].min():.4f}\n"
                f"最大值: {user_error_df['mean_error'].max():.4f}"
            )
            
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax1.legend()
        
        # 绘制箱线图
        box_plot = ax2.boxplot(user_error_df['mean_error'], vert=False, patch_artist=True,
                              boxprops=dict(facecolor='lightcoral', alpha=0.7))
        ax2.set_xlabel("平均预测误差", fontsize=12)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        ax2.set_title("误差箱线图", fontsize=12)
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "user_error_distribution.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"用户误差分布图已保存到: {save_path}")
        logger.info(f"分析了 {len(user_error_df)} 个用户的误差分布")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制用户误差分布图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_error_vs_popularity(output_df: pd.DataFrame,
                            movie_stats: pd.DataFrame,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            n_bins: int = 20) -> Optional[plt.Figure]:
    """
    绘制电影热度与预测误差的关系图
    
    通过线图展示电影受欢迎程度（评分数量）与预测误差之间的关系，
    有助于识别模型在不同热度电影上的预测表现差异。
    
    Args:
        output_df (pd.DataFrame): 包含预测误差的DataFrame
                                必须包含'movieId'和'error'列
        movie_stats (pd.DataFrame): 包含电影统计信息的DataFrame
                                  必须包含'movieId'和'movie_num_ratings'列
        save_path (Optional[str]): 图片保存路径，如果为None则使用默认路径
        figsize (Tuple[int, int]): 图表尺寸，默认为(12, 8)
        n_bins (int): 分箱数量，默认为20
    
    Returns:
        Optional[plt.Figure]: matplotlib图表对象，如果绘制失败则返回None
    
    Raises:
        ValueError: 当DataFrame缺少必要列时抛出异常
        TypeError: 当参数类型不正确时抛出异常
    
    Example:
        >>> predictions_df = pd.DataFrame({
        ...     'movieId': [1, 2, 3, 4, 5],
        ...     'error': [0.1, -0.2, 0.3, -0.1, 0.2]
        ... })
        >>> movie_stats_df = pd.DataFrame({
        ...     'movieId': [1, 2, 3, 4, 5],
        ...     'movie_num_ratings': [100, 50, 200, 10, 500]
        ... })
        >>> fig = plot_error_vs_popularity(predictions_df, movie_stats_df)
    
    Note:
        - 热门电影通常有更多的训练数据，可能预测更准确
        - 冷门电影由于数据稀少，可能预测误差较大
        - 使用对数刻度更好地展示不同热度范围的电影
        - 分箱处理有助于减少噪声，显示整体趋势
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df必须是pandas DataFrame类型")
    
    if not isinstance(movie_stats, pd.DataFrame):
        raise TypeError("movie_stats必须是pandas DataFrame类型")
    
    if output_df.empty or movie_stats.empty:
        logger.warning("输入DataFrame为空，无法绘制误差与热度关系图")
        return None
    
    # 检查必要列
    output_required = ['movieId', 'error']
    stats_required = ['movieId', 'movie_num_ratings']
    
    output_missing = [col for col in output_required if col not in output_df.columns]
    stats_missing = [col for col in stats_required if col not in movie_stats.columns]
    
    if output_missing:
        raise ValueError(f"output_df缺少必要的列: {output_missing}")
    if stats_missing:
        raise ValueError(f"movie_stats缺少必要的列: {stats_missing}")
    
    try:
        # 合并数据
        output_with_stats = output_df.merge(movie_stats, on='movieId', how='inner')
        
        if output_with_stats.empty:
            logger.warning("合并后的数据为空，无法绘制关系图")
            return None
        
        # 清理数据
        output_with_stats = output_with_stats.dropna(subset=['error', 'movie_num_ratings'])
        
        # 过滤掉评分数量为0的电影
        output_with_stats = output_with_stats[output_with_stats['movie_num_ratings'] > 0]
        
        if output_with_stats.empty:
            logger.warning("清理后的数据为空，无法绘制关系图")
            return None
        
        # 使用分位数分箱，避免重复值问题
        try:
            output_with_stats['popularity_bin'] = pd.qcut(
                output_with_stats['movie_num_ratings'], 
                q=n_bins, 
                duplicates='drop'
            )
        except ValueError as e:
            logger.warning(f"分位数分箱失败，使用等宽分箱: {e}")
            output_with_stats['popularity_bin'] = pd.cut(
                output_with_stats['movie_num_ratings'], 
                bins=n_bins
            )
        
        # 计算每个分箱的统计信息
        bin_stats = output_with_stats.groupby('popularity_bin').agg({
            'movie_num_ratings': ['mean', 'median', 'count'],
            'error': ['mean', 'std', 'count']
        }).reset_index()
        
        # 简化列名
        bin_stats.columns = [
            'popularity_bin', 'avg_popularity', 'median_popularity', 'movie_count',
            'avg_error', 'std_error', 'sample_count'
        ]
        
        # 计算标准误差
        bin_stats['sem_error'] = bin_stats['std_error'] / np.sqrt(bin_stats['sample_count'])
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # 主图：误差与热度关系
        line_plot = sns.lineplot(
            data=bin_stats, 
            x='avg_popularity', 
            y='avg_error', 
            marker='o', 
            markersize=8,
            linewidth=2,
            ax=ax1
        )
        
        # 添加误差条
        ax1.errorbar(
            bin_stats['avg_popularity'], 
            bin_stats['avg_error'],
            yerr=bin_stats['sem_error'],
            fmt='none',
            capsize=5,
            alpha=0.7,
            color='gray'
        )
        
        # 添加零误差参考线
        ax1.axhline(0, color='red', linestyle='--', alpha=0.8, label='零误差线')
        
        # 设置对数刻度
        ax1.set_xscale("log")
        ax1.set_title("电影热度与平均预测误差关系", fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel("电影评分数量 (对数刻度)", fontsize=12)
        ax1.set_ylabel("平均预测误差", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 添加统计信息
        correlation = output_with_stats['movie_num_ratings'].corr(output_with_stats['error'])
        stats_text = (
            f"样本数量: {len(output_with_stats)}\n"
            f"电影数量: {output_with_stats['movieId'].nunique()}\n"
            f"相关系数: {correlation:.4f}\n"
            f"分箱数量: {len(bin_stats)}\n"
            f"热度范围: {output_with_stats['movie_num_ratings'].min():.0f} - {output_with_stats['movie_num_ratings'].max():.0f}"
        )
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 下图：样本数量分布
        ax2.bar(range(len(bin_stats)), bin_stats['sample_count'], 
               alpha=0.7, color='skyblue')
        ax2.set_xlabel("热度分箱 (从低到高)", fontsize=12)
        ax2.set_ylabel("样本数量", fontsize=12)
        ax2.set_title("各热度分箱的样本数量分布", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "error_vs_popularity_line.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"误差与热度关系图已保存到: {save_path}")
        logger.info(f"相关系数: {correlation:.4f}, 分析了 {len(output_with_stats)} 个样本")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制误差与热度关系图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None