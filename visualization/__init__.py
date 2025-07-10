# visualization/__init__.py
"""
Visualization Package - 机器学习模型可视化工具包

本包提供了一套完整的机器学习模型可视化工具，专门用于推荐系统和评分预测任务的
结果分析和模型诊断。包含基本图表、误差分析和特征分析三大类可视化功能。

主要功能模块:
    - basic_plots: 基础可视化图表（箱线图、直方图等）
    - error_analysis: 误差分析图表（误差分布、混淆矩阵等）
    - feature_plots: 特征分析图表（特征重要性、相关性等）

特性:
    - 统一的API设计，易于使用和集成
    - 丰富的参数配置，支持高度自定义
    - 完善的错误处理和日志记录
    - 高质量的图表输出，支持多种格式
    - 详细的统计信息展示
    - 自动化的异常值检测和标注

使用场景:
    - 模型性能评估和诊断
    - 预测结果质量分析
    - 特征工程效果验证
    - 数据分布探索分析
    - 模型可解释性研究

Author: Assistant
Date: 2024-12
Version: 1.0.0

Example:
    >>> from visualization import plot_error_distribution, plot_feature_correlation
    >>> fig1 = plot_error_distribution(y_true, y_pred)
    >>> fig2 = plot_feature_correlation(X_train, y_train)
"""

from .basic_plots import (
    plot_boxplot_true_vs_pred,
    plot_predicted_rating_hist
)

from .error_analysis import (
    plot_error_distribution,
    plot_mean_error_per_rating,
    plot_rmse_per_rating,
    plot_confusion_heatmap,
    plot_user_error_distribution,
    plot_error_vs_popularity,
    plot_error_by_year
)

from .feature_plots import (
    plot_top20_feature_importance,
    plot_feature_correlation,
    plot_feature_distributions
)

__all__ = [
    # 基本图表
    'plot_boxplot_true_vs_pred',
    'plot_predicted_rating_hist',
    
    # 误差分析图表
    'plot_error_distribution',
    'plot_mean_error_per_rating',
    'plot_rmse_per_rating',
    'plot_confusion_heatmap',
    'plot_user_error_distribution',
    'plot_error_vs_popularity',
    'plot_error_by_year',
    
    # 特征相关图表
    'plot_top20_feature_importance',
    'plot_feature_correlation',
    'plot_feature_distributions'
]