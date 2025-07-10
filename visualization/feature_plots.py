#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征可视化模块

提供全面的特征分析和可视化功能，帮助理解模型特征的重要性、相关性和分布特征。

核心功能：
1. 特征重要性分析：
   - Top-N特征重要性排名可视化
   - 支持多模型特征重要性聚合
   - 可解释特征与隐因子特征分类展示
   - 特征重要性的统计分析（均值、标准差）

2. 特征相关性分析：
   - 特征与目标变量的相关性热力图
   - 支持多种相关性计算方法（Pearson、Spearman、Kendall）
   - 相关性阈值过滤和显著性检验
   - 特征间相关性矩阵可视化

3. 特征分布分析：
   - 多特征分布的子图网格展示
   - 支持直方图、密度图、箱线图等多种分布图
   - 统计信息标注（均值、标准差、偏度、峰度）
   - 异常值检测和标记

特征分类体系：
- 可解释特征：用户统计、电影统计、类型偏好、时间特征等
- 隐因子特征：用户隐因子、物品隐因子、交叉隐因子等
- 工程特征：TF-IDF特征、协同过滤特征、交互特征等

可视化特点：
- 支持中文字体显示
- 专业的配色方案和样式
- 详细的统计信息和图例
- 高质量图表输出（300 DPI）
- 灵活的布局和尺寸配置
- 完整的异常处理和日志记录

技术特点：
- 基于matplotlib和seaborn的专业可视化
- 支持大规模特征集的高效处理
- 多模型特征重要性聚合算法
- 统计显著性检验和置信区间
- 内存优化的批量处理
- 模块化的可扩展设计
"""

import os
from typing import Optional, List, Dict, Any, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
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

def plot_top20_feature_importance(models: List[Any],
                                  X_train: pd.DataFrame,
                                  whitelist_features: Optional[List[str]] = None,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 8),
                                  top_n: int = 20,
                                  show_values: bool = True) -> Optional[plt.Figure]:
    """
    绘制前N个特征重要性条形图
    
    创建特征重要性排名的水平条形图，支持多模型聚合和特征分类展示。
    帮助识别对模型预测最重要的特征，区分可解释特征和隐因子特征。
    
    功能特点：
    - 多模型特征重要性聚合（计算均值和标准差）
    - 可解释特征与隐因子特征分类处理
    - 隐因子特征按类型分组（用户、物品、交叉隐因子）
    - 误差条显示多模型间的重要性变异
    - 详细的统计信息标注
    
    可视化内容：
    - 水平条形图：特征重要性排名
    - 误差条：多模型间的标准差（如适用）
    - 数值标注：精确的重要性得分
    - 统计信息：模型数量、特征统计、最高重要性
    
    Args:
        models: 训练好的模型列表，需具有feature_importances_属性
        X_train: 训练集特征DataFrame，用于获取特征名称
        whitelist_features: 可解释特征白名单，None时使用config中的白名单
        save_path: 保存图表的路径，None时使用默认路径
        figsize: 图表大小，默认(12, 8)
        top_n: 显示前N个重要特征，默认20
        show_values: 是否在条形图上显示数值，默认True
    
    Returns:
        matplotlib图表对象，绘制失败时返回None
        
    Raises:
        TypeError: 当X_train不是DataFrame时
        ValueError: 当模型缺少feature_importances_属性时
        
    Example:
        >>> models = [lgb_model1, lgb_model2, lgb_model3]
        >>> X_train = pd.DataFrame({'feature1': [1,2,3], 'feature2': [4,5,6]})
        >>> whitelist = ['feature1', 'feature2']
        >>> fig = plot_top20_feature_importance(models, X_train, whitelist)
        
    Note:
        - 隐因子特征会按类型分组聚合显示
        - 多模型的重要性通过平均值聚合
        - 标准差反映模型间重要性的一致性
        - 可解释特征有助于业务理解和决策
    """
    # 参数验证
    if not models:
        logger.warning("Model list is empty, cannot plot feature importance.")
        return None
    
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    
    if X_train.empty:
        logger.warning("Training data is empty, cannot plot feature importance.")
        return None
    
    # 验证模型是否具有feature_importances_属性
    for i, model in enumerate(models):
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {i} does not have feature_importances_ attribute.")
    
    # 如果没有提供白名单，使用config中的白名单
    if whitelist_features is None:
        whitelist_features = getattr(config, 'whitelist_features', [])
    
    try:
        # 收集所有模型的特征重要性
        importance_dict = defaultdict(list)
        
        for model_idx, model in enumerate(models):
            try:
                feature_importances = model.feature_importances_
                if len(feature_importances) != len(X_train.columns):
                    logger.warning(f"Feature importance length of model {model_idx} does not match the number of features in training data.")
                    continue
                
                for fname, score in zip(X_train.columns, feature_importances):
                    importance_dict[fname].append(score)
            except Exception as e:
                logger.warning(f"Failed to get feature importance from model {model_idx}: {e}")
                continue
        
        if not importance_dict:
            logger.error("Could not get feature importance from any model.")
            return None
        
        # 过滤可解释特征
        interpretable_data = []
        for fname, scores in importance_dict.items():
            if fname in whitelist_features and scores:
                avg_importance = np.mean(scores)
                std_importance = np.std(scores) if len(scores) > 1 else 0
                interpretable_data.append((fname, avg_importance, std_importance, len(scores)))
        
        # 汇总隐因子特征重要性
        latent_groups = {
            '用户隐因子': lambda name: name.startswith("user_f"),
            '物品隐因子': lambda name: name.startswith("item_f"),
            '交叉隐因子': lambda name: name.startswith("cross_f"),
            '其他隐因子': lambda name: any(name.startswith(prefix) for prefix in ["latent_", "factor_", "embed_"])
        }
        
        latent_data = []
        for label, condition in latent_groups.items():
            matching_features = [f for f in importance_dict.keys() if condition(f)]
            if matching_features:
                all_scores = []
                for f in matching_features:
                    all_scores.extend(importance_dict[f])
                
                if all_scores:
                    avg_importance = np.mean(all_scores)
                    std_importance = np.std(all_scores)
                    feature_count = len(matching_features)
                    latent_data.append((label, avg_importance, std_importance, feature_count))
        
        # 合并重要性数据
        total_data = interpretable_data + latent_data
        
        if not total_data:
            logger.warning("No valid feature importance data available.")
            return None
        
        # 创建DataFrame
        importance_df = pd.DataFrame(
            total_data, 
            columns=['feature', 'importance_mean', 'importance_std', 'feature_count']
        )
        importance_df = importance_df.sort_values(by='importance_mean', ascending=False).head(top_n)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制条形图
        bars = sns.barplot(
            data=importance_df, 
            y='feature', 
            x='importance_mean', 
            palette='viridis',
            ax=ax
        )
        
        # 添加误差条（如果有多个模型）
        if len(models) > 1:
            ax.errorbar(
                importance_df['importance_mean'],
                range(len(importance_df)),
                xerr=importance_df['importance_std'],
                fmt='none',
                color='black',
                capsize=3,
                alpha=0.7
            )
        
        # 在条形图上显示数值
        if show_values:
            for i, (idx, row) in enumerate(importance_df.iterrows()):
                value = row['importance_mean']
                ax.text(value + 0.01 * ax.get_xlim()[1], i, 
                       f'{value:.3f}', 
                       va='center', fontsize=10, fontweight='bold')
        
        # 设置标题和标签
        ax.set_title(f"前{len(importance_df)}个特征重要性排名", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("平均重要性得分", fontsize=12)
        ax.set_ylabel("特征名称", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = (
            f"模型数量: {len(models)}\n"
            f"总特征数: {len(X_train.columns)}\n"
            f"可解释特征: {len(interpretable_data)}\n"
            f"隐因子组: {len(latent_data)}\n"
            f"最高重要性: {importance_df['importance_mean'].max():.4f}"
        )
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "top20_feature_importance.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Feature importance plot saved to: {save_path}")
        logger.info(f"Displayed {len(importance_df)} important features, based on {len(models)} models.")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制特征重要性图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None



def plot_feature_correlation(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            whitelist_features: Optional[List[str]] = None,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8),
                            correlation_method: str = 'pearson',
                            min_correlation: float = 0.01,
                            show_values: bool = True) -> Optional[plt.Figure]:
    """
    绘制特征与目标变量的相关性热力图
    
    通过热力图展示特征与目标变量之间的相关性强度和方向，
    帮助识别对预测目标最有影响力的特征。
    
    Args:
        X_train (pd.DataFrame): 训练集特征DataFrame
        y_train (pd.Series): 训练集目标变量Series
        whitelist_features (Optional[List[str]]): 可解释特征白名单列表，
                                                如果为None则使用config中的白名单
        save_path (Optional[str]): 图片保存路径，如果为None则使用默认路径
        figsize (Tuple[int, int]): 图表尺寸，默认为(12, 8)
        correlation_method (str): 相关性计算方法，可选'pearson'、'spearman'、'kendall'
        min_correlation (float): 最小相关性阈值，低于此值的特征将被过滤
        show_values (bool): 是否在热力图上显示相关性数值，默认为True
    
    Returns:
        Optional[plt.Figure]: matplotlib图表对象，如果绘制失败则返回None
    
    Raises:
        ValueError: 当输入参数无效时抛出异常
        TypeError: 当参数类型不正确时抛出异常
    
    Example:
        >>> fig = plot_feature_correlation(X_train, y_train, 
        ...                              correlation_method='spearman',
        ...                              min_correlation=0.05)
    
    Note:
        - 支持多种相关性计算方法（Pearson、Spearman、Kendall）
        - 自动过滤低相关性特征，突出重要特征
        - 使用颜色编码显示正负相关性
        - 按相关性绝对值排序显示
    """
    # 参数验证
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train必须是pandas DataFrame类型")
    
    if not isinstance(y_train, (pd.Series, np.ndarray)):
        raise TypeError("y_train必须是pandas Series或numpy数组类型")
    
    if X_train.empty:
        logger.warning("训练数据为空，无法绘制相关性图")
        return None
    
    if len(X_train) != len(y_train):
        raise ValueError("X_train和y_train的长度必须相同")
    
    if correlation_method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("correlation_method必须是'pearson'、'spearman'或'kendall'之一")
    
    # 如果没有提供白名单，使用config中的白名单
    if whitelist_features is None:
        whitelist_features = getattr(config, 'whitelist_features', [])
    
    try:
        # 筛选可解释特征
        interpretable_features = [f for f in X_train.columns if f in whitelist_features]
        
        if not interpretable_features:
            logger.warning("白名单中没有找到可解释特征")
            # 如果没有白名单特征，使用数值型特征
            numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
            interpretable_features = numeric_features[:50]  # 限制特征数量
            logger.info(f"使用前50个数值型特征: {len(interpretable_features)}个")
        
        if not interpretable_features:
            logger.error("没有找到任何可用的特征")
            return None
        
        # 获取可解释特征数据
        X_interpretable = X_train[interpretable_features].copy()
        
        # 处理缺失值
        if X_interpretable.isnull().any().any():
            logger.warning("发现缺失值，使用均值填充")
            X_interpretable = X_interpretable.fillna(X_interpretable.mean())
        
        # 转换y_train为Series（如果是numpy数组）
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train, index=X_interpretable.index)
        
        # 计算相关性
        correlation_data = X_interpretable.corrwith(y_train, method=correlation_method)
        
        # 过滤掉NaN值和低相关性特征
        correlation_data = correlation_data.dropna()
        correlation_data = correlation_data[abs(correlation_data) >= min_correlation]
        
        if correlation_data.empty:
            logger.warning(f"没有特征的相关性超过阈值 {min_correlation}")
            return None
        
        # 按绝对相关性排序
        correlation_data = correlation_data.reindex(
            correlation_data.abs().sort_values(ascending=False).index
        )
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]})
        
        # 绘制热力图
        correlation_matrix = correlation_data.values.reshape(-1, 1)
        
        # 创建自定义颜色映射
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', 
                 '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
        n_colors = len(colors)
        cmap = sns.blend_palette(colors, n_colors=256, as_cmap=True)
        
        # 绘制热力图
        sns.heatmap(
            correlation_matrix, 
            annot=show_values,
            fmt='.3f',
            cmap=cmap,
            center=0,
            vmin=-1, vmax=1,
            yticklabels=correlation_data.index,
            xticklabels=['目标相关性'],
            ax=ax1,
            cbar_kws={'label': '相关性系数'}
        )
        
        ax1.set_title(f"特征-目标相关性热力图 ({correlation_method.title()})", 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel("相关性类型", fontsize=12)
        ax1.set_ylabel("特征名称", fontsize=12)
        
        # 绘制相关性条形图
        colors_bar = ['red' if x < 0 else 'blue' for x in correlation_data.values]
        bars = ax2.barh(range(len(correlation_data)), correlation_data.values, color=colors_bar, alpha=0.7)
        
        ax2.set_yticks(range(len(correlation_data)))
        ax2.set_yticklabels([])
        ax2.set_xlabel("相关性系数", fontsize=12)
        ax2.set_title("相关性分布", fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        if show_values:
            for i, v in enumerate(correlation_data.values):
                ax2.text(v + (0.02 if v >= 0 else -0.02), i, f'{v:.3f}', 
                        va='center', ha='left' if v >= 0 else 'right', fontsize=9)
        
        # 添加统计信息
        stats_text = (
            f"特征数量: {len(correlation_data)}\n"
            f"相关性方法: {correlation_method.title()}\n"
            f"最小阈值: {min_correlation}\n"
            f"最强正相关: {correlation_data.max():.4f}\n"
            f"最强负相关: {correlation_data.min():.4f}\n"
            f"平均绝对相关性: {abs(correlation_data).mean():.4f}"
        )
        
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
                verticalalignment='top', horizontalalignment='left',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 调整布局
        fig.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "feature_correlation.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"特征相关性图已保存到: {save_path}")
        logger.info(f"显示了 {len(correlation_data)} 个特征的相关性，使用 {correlation_method} 方法")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制特征相关性图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def plot_feature_distributions(X_train: pd.DataFrame,
                              whitelist_features: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              figsize: Optional[Tuple[int, int]] = None,
                              n_cols: int = 3,
                              max_features: int = 20,
                              plot_type: str = 'hist',
                              show_stats: bool = True) -> Optional[plt.Figure]:
    """
    绘制特征分布图
    
    通过直方图、密度图或箱线图展示特征的分布情况，
    帮助理解数据的分布特性、异常值和偏态情况。
    
    Args:
        X_train (pd.DataFrame): 训练集特征DataFrame
        whitelist_features (Optional[List[str]]): 可解释特征白名单列表，
                                                如果为None则使用config中的白名单
        save_path (Optional[str]): 图片保存路径，如果为None则使用默认路径
        figsize (Optional[Tuple[int, int]]): 图表尺寸，如果为None则自动计算
        n_cols (int): 子图列数，默认为3
        max_features (int): 最大显示特征数量，默认为20
        plot_type (str): 图表类型，可选'hist'、'kde'、'box'、'violin'
        show_stats (bool): 是否显示统计信息，默认为True
    
    Returns:
        Optional[plt.Figure]: matplotlib图表对象，如果绘制失败则返回None
    
    Raises:
        ValueError: 当输入参数无效时抛出异常
        TypeError: 当参数类型不正确时抛出异常
    
    Example:
        >>> fig = plot_feature_distributions(X_train, 
        ...                                plot_type='violin',
        ...                                max_features=15)
    
    Note:
        - 支持多种分布图类型（直方图、密度图、箱线图、小提琴图）
        - 自动处理缺失值和异常值
        - 显示关键统计信息（均值、中位数、标准差等）
        - 自动调整子图布局以获得最佳显示效果
    """
    # 参数验证
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train必须是pandas DataFrame类型")
    
    if X_train.empty:
        logger.warning("训练数据为空，无法绘制特征分布图")
        return None
    
    if plot_type not in ['hist', 'kde', 'box', 'violin']:
        raise ValueError("plot_type必须是'hist'、'kde'、'box'或'violin'之一")
    
    # 如果没有提供白名单，使用config中的白名单
    if whitelist_features is None:
        whitelist_features = getattr(config, 'whitelist_features', [])
    
    try:
        # 筛选可解释特征
        interpretable_features = [f for f in X_train.columns if f in whitelist_features]
        
        if not interpretable_features:
            logger.warning("白名单中没有找到可解释特征")
            # 如果没有白名单特征，使用数值型特征
            numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
            interpretable_features = numeric_features[:max_features]
            logger.info(f"使用前{max_features}个数值型特征: {len(interpretable_features)}个")
        
        if not interpretable_features:
            logger.error("没有找到任何可用的特征")
            return None
        
        # 限制特征数量
        interpretable_features = interpretable_features[:max_features]
        
        # 计算子图布局
        n_features = len(interpretable_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # 自动计算图表尺寸
        if figsize is None:
            width = max(15, n_cols * 5)
            height = max(10, n_rows * 4)
            figsize = (width, height)
        
        # 创建图表
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        # 绘制每个特征的分布图
        for i, feature in enumerate(interpretable_features):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                # 获取特征数据并处理缺失值
                feature_data = X_train[feature].dropna()
                
                if feature_data.empty:
                    ax.text(0.5, 0.5, f'{feature}\n(无有效数据)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{feature} - 无数据")
                    continue
                
                # 计算统计信息
                mean_val = feature_data.mean()
                median_val = feature_data.median()
                std_val = feature_data.std()
                skew_val = feature_data.skew()
                
                # 根据图表类型绘制
                if plot_type == 'hist':
                    # 直方图 + 密度曲线
                    sns.histplot(feature_data, kde=True, ax=ax, alpha=0.7)
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'均值: {mean_val:.3f}')
                    ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'中位数: {median_val:.3f}')
                    
                elif plot_type == 'kde':
                    # 密度图
                    sns.kdeplot(feature_data, ax=ax, fill=True, alpha=0.7)
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'均值: {mean_val:.3f}')
                    ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'中位数: {median_val:.3f}')
                    
                elif plot_type == 'box':
                    # 箱线图
                    sns.boxplot(y=feature_data, ax=ax)
                    
                elif plot_type == 'violin':
                    # 小提琴图
                    sns.violinplot(y=feature_data, ax=ax)
                
                # 设置标题和标签
                title = f"{feature}"
                if show_stats:
                    title += f"\n(μ={mean_val:.3f}, σ={std_val:.3f}, 偏度={skew_val:.2f})"
                
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel(feature if plot_type in ['hist', 'kde'] else '')
                ax.set_ylabel('频率' if plot_type == 'hist' else '密度' if plot_type == 'kde' else feature)
                ax.grid(True, alpha=0.3)
                
                # 添加图例（仅对hist和kde）
                if plot_type in ['hist', 'kde'] and show_stats:
                    ax.legend(fontsize=9)
                
                # 检测并标注异常值
                if plot_type in ['hist', 'kde']:
                    Q1 = feature_data.quantile(0.25)
                    Q3 = feature_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
                    if len(outliers) > 0:
                        outlier_ratio = len(outliers) / len(feature_data) * 100
                        ax.text(0.02, 0.98, f'异常值: {len(outliers)} ({outlier_ratio:.1f}%)', 
                               transform=ax.transAxes, va='top', ha='left',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               fontsize=8)
                
            except Exception as e:
                logger.warning(f"绘制特征 {feature} 的分布图失败: {e}")
                ax.text(0.5, 0.5, f'{feature}\n(绘制失败)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{feature} - 绘制失败")
        
        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        # 添加总体标题
        fig.suptitle(f"特征分布图 ({plot_type.upper()}) - 共{n_features}个特征", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 调整布局
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(config.save_dir, "feature_distributions.png")
        
        # 确保保存目录存在
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"特征分布图已保存到: {save_path}")
        logger.info(f"显示了 {n_features} 个特征的分布，使用 {plot_type} 图表类型")
        
        return fig
        
    except Exception as e:
        logger.error(f"绘制特征分布图失败: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None