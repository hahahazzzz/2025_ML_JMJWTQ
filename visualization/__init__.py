# visualization/__init__.py
"""
Visualization Package - 机器学习模型可视化工具包

提供基础图表、误差分析和特征分析三大类可视化功能
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
    plot_error_vs_popularity
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
    
    # 特征相关图表
    'plot_top20_feature_importance',
    'plot_feature_correlation',
    'plot_feature_distributions'
]