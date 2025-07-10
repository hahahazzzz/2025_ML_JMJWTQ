#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本可视化图表模块

该模块提供了电影推荐系统的基础可视化功能，主要包括：
1. 评分分布可视化：真实评分与预测评分的对比分析
2. 预测结果分析：预测评分的分布特征和统计特性
3. 模型性能可视化：通过图表直观展示模型预测效果
4. 数据质量检查：通过可视化发现数据异常和模式

主要功能：
- 箱线图分析：展示不同真实评分下的预测评分分布
- 直方图分析：展示预测评分的整体分布特征
- 散点图分析：展示真实评分与预测评分的相关性
- 统计图表：提供详细的数值统计信息

可视化特性：
- 使用专业的配色方案和图表样式
- 自动保存高质量图片文件
- 支持中文标签和注释
- 提供详细的图例和标题
- 自动调整图表布局和尺寸

使用方式：
    from visualization.basic_plots import plot_boxplot_true_vs_pred, plot_predicted_rating_hist
    
    # 绘制箱线图
    plot_boxplot_true_vs_pred(predictions_df)
    
    # 绘制直方图
    plot_predicted_rating_hist(predictions_df)

输出文件：
    所有图表自动保存到配置指定的输出目录中：
    - boxplot_true_vs_pred.png: 真实vs预测评分箱线图
    - predicted_rating_hist.png: 预测评分分布直方图
    - scatter_true_vs_pred.png: 真实vs预测评分散点图

依赖库：
    - matplotlib: 基础绘图功能
    - seaborn: 高级统计图表
    - pandas: 数据处理
    - numpy: 数值计算

作者: 电影推荐系统开发团队
创建时间: 2024
最后修改: 2024
"""

import os
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from config import config
from utils.logger import logger

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_boxplot_true_vs_pred(output_df: pd.DataFrame, 
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8),
                              show_stats: bool = True) -> Optional[plt.Figure]:
    """
    绘制真实评分与预测评分的箱线图
    
    通过箱线图展示不同真实评分等级下预测评分的分布特征，
    包括中位数、四分位数、异常值等统计信息。有助于分析模型
    在不同评分等级上的预测准确性和一致性。
    
    Args:
        output_df (pd.DataFrame): 包含真实评分和预测评分的DataFrame
                                必须包含'true_rating'和'pred_rating'列
        save_path (Optional[str]): 图片保存路径，如果为None则使用默认路径
        figsize (Tuple[int, int]): 图表尺寸，默认为(10, 8)
        show_stats (bool): 是否在图表上显示统计信息，默认为True
    
    Returns:
        Optional[plt.Figure]: matplotlib图表对象，如果绘制失败则返回None
    
    Raises:
        ValueError: 当DataFrame缺少必要列时抛出异常
        TypeError: 当参数类型不正确时抛出异常
    
    Example:
        >>> predictions_df = pd.DataFrame({
        ...     'true_rating': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        ...     'pred_rating': [1.2, 2.1, 2.9, 4.1, 4.8, 0.9, 2.3, 3.2, 3.9, 5.1]
        ... })
        >>> fig = plot_boxplot_true_vs_pred(predictions_df)
    
    Note:
        - 箱线图显示每个真实评分等级的预测评分分布
        - 箱体表示25%-75%分位数范围
        - 中线表示中位数
        - 须线表示1.5倍IQR范围
        - 圆点表示异常值
        - 自动保存为高分辨率PNG格式
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df必须是pandas DataFrame类型")
    
    if output_df.empty:
        logger.warning("DataFrame为空，无法绘制箱线图")
        return None
    
    required_columns = ['true_rating', 'pred_rating']
    missing_columns = [col for col in required_columns if col not in output_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame缺少必要的列: {missing_columns}")
    
    # 数据预处理
    df_clean = output_df[required_columns].dropna()
    if df_clean.empty:
        logger.warning("清理后的数据为空，无法绘制箱线图")
        return None
    
    try:
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制箱线图
        box_plot = sns.boxplot(
            x='true_rating', 
            y='pred_rating', 
            data=df_clean, 
            ax=ax, 
            palette='Set3',
            showfliers=True,
            notch=True  # 显示置信区间
        )
        
        # 设置标题和标签
        ax.set_title("真实评分与预测评分分布对比", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("真实评分", fontsize=12)
        ax.set_ylabel("预测评分", fontsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 显示统计信息
        if show_stats:
            # 计算每个真实评分的统计信息
            stats_text = []
            for rating in sorted(df_clean['true_rating'].unique()):
                rating_data = df_clean[df_clean['true_rating'] == rating]['pred_rating']
                mean_val = rating_data.mean()
                std_val = rating_data.std()
                stats_text.append(f"评分{rating}: μ={mean_val:.2f}, σ={std_val:.2f}")
            
            # 在图表上添加统计信息
            stats_str = '\n'.join(stats_text)
            ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 设置y轴范围
        y_min, y_max = df_clean['pred_rating'].min(), df_clean['pred_rating'].max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "boxplot_true_vs_pred.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"箱线图已保存到: {save_path}")
        logger.info(f"分析了 {len(df_clean)} 个预测样本，涵盖 {df_clean['true_rating'].nunique()} 个评分等级")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制箱线图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_predicted_rating_hist(output_df: pd.DataFrame,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8),
                              bins: int = 20,
                              show_kde: bool = True,
                              show_stats: bool = True) -> Optional[plt.Figure]:
    """
    绘制预测评分的直方图
    
    通过直方图展示预测评分的分布特征，包括频率分布、概率密度曲线
    和关键统计指标。有助于理解模型预测结果的整体分布模式，
    识别预测偏差和异常值。
    
    Args:
        output_df (pd.DataFrame): 包含预测评分的DataFrame
                                必须包含'pred_rating'列
        save_path (Optional[str]): 图片保存路径，如果为None则使用默认路径
        figsize (Tuple[int, int]): 图表尺寸，默认为(10, 8)
        bins (int): 直方图的箱数，默认为20
        show_kde (bool): 是否显示核密度估计曲线，默认为True
        show_stats (bool): 是否显示统计信息，默认为True
    
    Returns:
        Optional[plt.Figure]: matplotlib图表对象，如果绘制失败则返回None
    
    Raises:
        ValueError: 当DataFrame缺少必要列时抛出异常
        TypeError: 当参数类型不正确时抛出异常
    
    Example:
        >>> predictions_df = pd.DataFrame({
        ...     'pred_rating': [1.2, 2.1, 2.9, 4.1, 4.8, 3.5, 2.8, 4.2, 3.1, 4.9]
        ... })
        >>> fig = plot_predicted_rating_hist(predictions_df, bins=15, show_kde=True)
    
    Note:
        - 直方图显示预测评分的频率分布
        - KDE曲线显示平滑的概率密度估计
        - 统计信息包括均值、标准差、偏度、峰度等
        - 自动标注关键统计值的位置
        - 支持自定义箱数和样式
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df必须是pandas DataFrame类型")
    
    if output_df.empty:
        logger.warning("DataFrame为空，无法绘制直方图")
        return None
    
    if 'pred_rating' not in output_df.columns:
        raise ValueError("DataFrame缺少'pred_rating'列")
    
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins必须是正整数")
    
    # 数据预处理
    pred_ratings = output_df['pred_rating'].dropna()
    if pred_ratings.empty:
        logger.warning("预测评分数据为空，无法绘制直方图")
        return None
    
    try:
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制直方图
        hist_plot = sns.histplot(
            pred_ratings, 
            bins=bins, 
            kde=show_kde, 
            color='skyblue', 
            ax=ax,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # 设置标题和标签
        ax.set_title("预测评分分布直方图", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("预测评分", fontsize=12)
        ax.set_ylabel("频次", fontsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 计算并显示统计信息
        if show_stats:
            mean_val = pred_ratings.mean()
            std_val = pred_ratings.std()
            median_val = pred_ratings.median()
            skew_val = pred_ratings.skew()
            kurt_val = pred_ratings.kurtosis()
            
            # 在图表上标注统计信息
            stats_text = (
                f"样本数: {len(pred_ratings)}\n"
                f"均值: {mean_val:.3f}\n"
                f"标准差: {std_val:.3f}\n"
                f"中位数: {median_val:.3f}\n"
                f"偏度: {skew_val:.3f}\n"
                f"峰度: {kurt_val:.3f}"
            )
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            # 添加均值和中位数的垂直线
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'均值: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, 
                      label=f'中位数: {median_val:.2f}')
            
            # 添加图例
            ax.legend(loc='upper left')
        
        # 设置x轴范围
        x_min, x_max = pred_ratings.min(), pred_ratings.max()
        x_range = x_max - x_min
        ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "predicted_rating_hist.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"预测评分直方图已保存到: {save_path}")
        logger.info(f"分析了 {len(pred_ratings)} 个预测评分，范围: [{x_min:.2f}, {x_max:.2f}]")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制预测评分直方图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None