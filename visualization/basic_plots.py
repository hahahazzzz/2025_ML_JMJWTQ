#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础可视化图表模块

为电影推荐系统提供基础可视化功能，包括评分分布、预测结果分析等。

核心功能：
1. 预测结果可视化：箱线图对比真实评分与预测评分
2. 分布分析：预测评分的直方图和核密度估计
3. 统计信息展示：均值、标准差、偏度、峰度等统计指标
4. 图表美化：统一的样式、中文字体支持、高质量输出

可视化特点：
- 支持中文字体显示，解决中文乱码问题
- 提供详细的统计信息标注
- 高质量图片输出（300 DPI）
- 完整的参数验证和错误处理
- 灵活的图表配置选项

图表类型：
1. 箱线图（Boxplot）：
   - 显示不同真实评分下的预测评分分布
   - 包含四分位数、异常值检测
   - 支持统计信息标注

2. 直方图（Histogram）：
   - 展示预测评分的整体分布
   - 可选核密度估计曲线
   - 标注均值、中位数等关键统计量

技术特点：
- 基于matplotlib和seaborn构建
- 支持自定义图表尺寸和样式
- 自动创建保存目录
- 完整的日志记录
- 异常安全的图表生成
"""

import os
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm

from config import config
from utils.logger import logger

# 导入字体配置模块以解决中文显示问题
from .font_config import setup_chinese_fonts

# 设置中文字体
setup_chinese_fonts()

sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_boxplot_true_vs_pred(output_df: pd.DataFrame, 
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8),
                              show_stats: bool = True) -> Optional[plt.Figure]:
    """
    绘制预测评分与真实评分的箱线图对比
    
    通过箱线图展示不同真实评分等级下预测评分的分布情况，
    帮助分析模型在各个评分等级上的预测性能。
    
    可视化内容：
    1. 每个真实评分等级的预测评分分布
    2. 四分位数、中位数、异常值
    3. 各评分等级的统计信息（均值、标准差）
    4. 预测偏差的可视化分析
    
    Args:
        output_df (pd.DataFrame): 包含真实评分和预测评分的DataFrame
            必须包含列：'true_rating', 'pred_rating'
        save_path (Optional[str]): 图片保存路径，默认保存到配置的输出目录
        figsize (Tuple[int, int]): 图表尺寸(宽, 高)，单位英寸
        show_stats (bool): 是否在图表上显示统计信息
    
    Returns:
        Optional[plt.Figure]: 成功时返回matplotlib图表对象，失败时返回None
    
    Raises:
        TypeError: 当output_df不是DataFrame时
        ValueError: 当DataFrame缺少必需列时
    
    Example:
        >>> output_df = pd.DataFrame({
        ...     'true_rating': [1, 2, 3, 4, 5],
        ...     'pred_rating': [1.2, 2.1, 2.8, 4.1, 4.9]
        ... })
        >>> fig = plot_boxplot_true_vs_pred(output_df)
    
    Note:
        - 图表会自动过滤缺失值
        - 统计信息包括每个评分等级的均值和标准差
        - 支持异常值检测和显示
    """
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df must be a pandas DataFrame")
    
    if output_df.empty:
        logger.warning("DataFrame is empty, cannot plot boxplot.")
        return None
    
    required_columns = ['true_rating', 'pred_rating']
    missing_columns = [col for col in required_columns if col not in output_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    
    df_clean = output_df[required_columns].dropna()
    if df_clean.empty:
        logger.warning("Cleaned data is empty, cannot plot boxplot.")
        return None
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        box_plot = sns.boxplot(
            x='true_rating', 
            y='pred_rating', 
            data=df_clean, 
            ax=ax, 
            palette='Set3',
            showfliers=True,
            notch=True
        )
        
        ax.set_title("预测评分与真实评分分布对比", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("真实评分", fontsize=12)
        ax.set_ylabel("预测评分", fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        # 显示统计信息
        if show_stats:
            # 计算每个真实评分的统计信息
            stats_text = []
            for rating in sorted(df_clean['true_rating'].unique()):
                rating_data = df_clean[df_clean['true_rating'] == rating]['pred_rating']
                mean_val = rating_data.mean()
                std_val = rating_data.std()
                stats_text.append(f"Rating {rating}: μ={mean_val:.2f}, σ={std_val:.2f}")
            
            # 将统计信息添加到图表中
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
        logger.info(f"Boxplot saved to: {save_path}")
        logger.info(f"Analyzed {len(df_clean)} prediction samples, covering {df_clean['true_rating'].nunique()} rating levels.")
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to plot boxplot: {e}")
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
    绘制预测评分的直方图分布
    
    通过直方图和核密度估计展示预测评分的整体分布特征，
    帮助分析模型预测结果的统计特性和分布规律。
    
    可视化内容：
    1. 预测评分的频率分布直方图
    2. 核密度估计曲线（可选）
    3. 关键统计量标注（均值、中位数）
    4. 详细统计信息（样本数、标准差、偏度、峰度）
    
    Args:
        output_df (pd.DataFrame): 包含预测评分的DataFrame
            必须包含列：'pred_rating'
        save_path (Optional[str]): 图片保存路径，默认保存到配置的输出目录
        figsize (Tuple[int, int]): 图表尺寸(宽, 高)，单位英寸
        bins (int): 直方图分箱数量，必须为正整数
        show_kde (bool): 是否显示核密度估计曲线
        show_stats (bool): 是否显示详细统计信息
    
    Returns:
        Optional[plt.Figure]: 成功时返回matplotlib图表对象，失败时返回None
    
    Raises:
        TypeError: 当output_df不是DataFrame时
        ValueError: 当DataFrame缺少'pred_rating'列或bins不是正整数时
    
    Example:
        >>> output_df = pd.DataFrame({
        ...     'pred_rating': np.random.normal(3.5, 1.0, 1000)
        ... })
        >>> fig = plot_predicted_rating_hist(output_df, bins=30)
    
    Note:
        - 自动过滤缺失值
        - 统计信息包括样本数、均值、标准差、中位数、偏度、峰度
        - 在图表上标注均值和中位数的位置
        - 支持自定义分箱数量以适应不同数据规模
    """
    # 参数验证
    if not isinstance(output_df, pd.DataFrame):
        raise TypeError("output_df must be a pandas DataFrame")
    
    if output_df.empty:
        logger.warning("DataFrame is empty, cannot plot histogram.")
        return None
    
    if 'pred_rating' not in output_df.columns:
        raise ValueError("DataFrame is missing 'pred_rating' column.")
    
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer.")
    
    # 数据预处理
    pred_ratings = output_df['pred_rating'].dropna()
    if pred_ratings.empty:
        logger.warning("Predicted rating data is empty, cannot plot histogram.")
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
        ax.set_ylabel("频率", fontsize=12)
        
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
        logger.info(f"Predicted rating histogram saved to: {save_path}")
        logger.info(f"Analyzed {len(pred_ratings)} predicted ratings, range: [{x_min:.2f}, {x_max:.2f}]")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制预测评分直方图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None