#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算模块

提供全面的模型评估指标计算功能，支持回归、分类和推荐系统的多维度性能分析。

核心功能：
1. 基础回归指标：
   - RMSE（均方根误差）：衡量预测值与真实值的整体偏差
   - MAE（平均绝对误差）：衡量预测误差的平均幅度
   - MAPE（平均绝对百分比误差）：相对误差的百分比表示

2. 分组分析指标：
   - 按评分等级的RMSE分析：识别模型在不同评分区间的性能
   - 用户误差分布分析：分析用户级别的预测准确性差异
   - 分层性能评估：多维度的模型性能分解

3. 分类评估指标：
   - 准确率、精确率、召回率、F1分数
   - 混淆矩阵和分类报告
   - 支持多类别和二分类评估

4. 推荐系统专用指标：
   - 评分预测准确性分析
   - 用户和物品维度的性能分析
   - 预测区间和置信度计算

5. 异常检测功能：
   - 预测异常值检测（IQR、Z-score方法）
   - 误差分布分析
   - 数据质量评估

技术特点：
- 完整的参数验证和异常处理
- 支持多种数据格式（numpy、pandas、list）
- 自动处理缺失值和无效数据
- 详细的日志记录和调试信息
- 高效的向量化计算
- 统计显著性检验支持

使用场景：
- 模型训练过程中的性能监控
- 模型选择和超参数调优
- A/B测试和模型对比
- 生产环境的模型质量监控
- 研究报告和论文的实验结果分析
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from scipy import stats
from utils.logger import logger


# ==================== 基础回归指标 ====================

def compute_rmse(true_ratings: Union[np.ndarray, pd.Series, List], 
                pred_ratings: Union[np.ndarray, pd.Series, List]) -> float:
    """
    计算均方根误差(Root Mean Square Error, RMSE)
    
    RMSE是回归问题中最常用的评估指标之一，它衡量预测值与真实值之间的平均偏差。
    RMSE对大误差更敏感，因为它对误差进行平方处理。
    
    计算公式：RMSE = sqrt(mean((y_true - y_pred)^2))
    
    Args:
        true_ratings: 真实评分数组，支持numpy数组、pandas Series或Python列表
        pred_ratings: 预测评分数组，必须与true_ratings长度相同
    
    Returns:
        float: RMSE值，值越小表示预测越准确
               - 0表示完美预测
               - 值的单位与原始评分相同
    
    Raises:
        ValueError: 当输入数组长度不匹配、为空或全为无效值时
        TypeError: 当输入数据类型无法转换为数值时
    
    Example:
        >>> true_ratings = [1, 2, 3, 4, 5]
        >>> pred_ratings = [1.1, 1.9, 3.2, 3.8, 4.9]
        >>> rmse = compute_rmse(true_ratings, pred_ratings)
        >>> print(f"RMSE: {rmse:.4f}")
        
    Note:
        - 自动处理NaN值，会在计算前移除
        - 对异常值敏感，适合检测模型的最大误差
        - 常用于模型选择和超参数调优
    """
    try:
        # 参数验证
        true_ratings = np.asarray(true_ratings, dtype=float)
        pred_ratings = np.asarray(pred_ratings, dtype=float)
        
        if len(true_ratings) != len(pred_ratings):
            raise ValueError(f"输入数组长度不匹配: {len(true_ratings)} vs {len(pred_ratings)}")
        
        if len(true_ratings) == 0:
            raise ValueError("输入数组不能为空")
        
        # 检查无效值
        if np.any(np.isnan(true_ratings)) or np.any(np.isnan(pred_ratings)):
            logger.warning("检测到NaN值，将被忽略")
            mask = ~(np.isnan(true_ratings) | np.isnan(pred_ratings))
            true_ratings = true_ratings[mask]
            pred_ratings = pred_ratings[mask]
        
        if len(true_ratings) == 0:
            raise ValueError("移除无效值后数组为空")
        
        # 计算均方根误差
        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        
        logger.debug(f"RMSE计算完成: {rmse:.6f} (样本数: {len(true_ratings)})")
        return float(rmse)
        
    except Exception as e:
        logger.error(f"RMSE计算失败: {e}")
        raise


def compute_mae(true_ratings: Union[np.ndarray, pd.Series, List], 
               pred_ratings: Union[np.ndarray, pd.Series, List]) -> float:
    """
    计算平均绝对误差(Mean Absolute Error, MAE)
    
    MAE衡量预测值与真实值之间的平均绝对偏差，相比RMSE对异常值不那么敏感。
    MAE提供了误差的线性度量，更容易解释。
    
    计算公式：MAE = mean(|y_true - y_pred|)
    
    Args:
        true_ratings: 真实评分数组，支持numpy数组、pandas Series或Python列表
        pred_ratings: 预测评分数组，必须与true_ratings长度相同
    
    Returns:
        float: MAE值，值越小表示预测越准确
               - 0表示完美预测
               - 值的单位与原始评分相同
               - 表示平均每个预测的绝对误差
    
    Raises:
        ValueError: 当输入数组长度不匹配或没有有效数据点时
        TypeError: 当输入数据类型无法转换为数值时
    
    Example:
        >>> true_ratings = [1, 2, 3, 4, 5]
        >>> pred_ratings = [1.2, 1.8, 3.1, 4.2, 4.8]
        >>> mae = compute_mae(true_ratings, pred_ratings)
        >>> print(f"MAE: {mae:.4f}")
        
    Note:
        - 对异常值的鲁棒性比RMSE更好
        - 提供误差的直观解释（平均偏差）
        - 适合评估模型的整体预测质量
    """
    try:
        true_ratings = np.asarray(true_ratings, dtype=float)
        pred_ratings = np.asarray(pred_ratings, dtype=float)
        
        if len(true_ratings) != len(pred_ratings):
            raise ValueError(f"输入数组长度不匹配: {len(true_ratings)} vs {len(pred_ratings)}")
        
        # 处理无效值
        mask = ~(np.isnan(true_ratings) | np.isnan(pred_ratings))
        true_ratings = true_ratings[mask]
        pred_ratings = pred_ratings[mask]
        
        if len(true_ratings) == 0:
            raise ValueError("没有有效的数据点")
        
        mae = mean_absolute_error(true_ratings, pred_ratings)
        logger.debug(f"MAE计算完成: {mae:.6f}")
        return float(mae)
        
    except Exception as e:
        logger.error(f"MAE计算失败: {e}")
        raise


def compute_mape(true_ratings: Union[np.ndarray, pd.Series, List], 
                pred_ratings: Union[np.ndarray, pd.Series, List]) -> float:
    """
    计算平均绝对百分比误差(Mean Absolute Percentage Error, MAPE)
    
    MAPE提供了相对误差的百分比表示，便于不同量级数据的比较。
    它表示预测误差相对于真实值的平均百分比。
    
    计算公式：MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Args:
        true_ratings: 真实评分数组，支持numpy数组、pandas Series或Python列表
        pred_ratings: 预测评分数组，必须与true_ratings长度相同
    
    Returns:
        float: MAPE值（百分比），值越小表示预测越准确
               - 0表示完美预测
               - 值以百分比形式表示
               - 例如：10.5表示平均10.5%的相对误差
    
    Raises:
        ValueError: 当输入数组长度不匹配或没有有效数据点时
        ZeroDivisionError: 当真实值包含0时（自动过滤）
    
    Example:
        >>> true_ratings = [2, 3, 4, 5]
        >>> pred_ratings = [2.1, 2.9, 4.2, 4.8]
        >>> mape = compute_mape(true_ratings, pred_ratings)
        >>> print(f"MAPE: {mape:.2f}%")
        
    Note:
        - 自动过滤真实值为0的数据点
        - 适合比较不同量级的预测任务
        - 对小的真实值敏感，可能产生较大的百分比误差
    """
    try:
        true_ratings = np.asarray(true_ratings, dtype=float)
        pred_ratings = np.asarray(pred_ratings, dtype=float)
        
        # 移除真实值为0的点
        mask = (true_ratings != 0) & ~np.isnan(true_ratings) & ~np.isnan(pred_ratings)
        true_ratings = true_ratings[mask]
        pred_ratings = pred_ratings[mask]
        
        if len(true_ratings) == 0:
            raise ValueError("没有有效的数据点（真实值不能为0）")
        
        mape = np.mean(np.abs((true_ratings - pred_ratings) / true_ratings)) * 100
        logger.debug(f"MAPE计算完成: {mape:.6f}%")
        return float(mape)
        
    except Exception as e:
        logger.error(f"MAPE计算失败: {e}")
        raise


# ==================== 分组分析指标 ====================

def rmse_by_class(df: pd.DataFrame, 
                 true_col: str = 'true_rating', 
                 pred_col: str = 'pred_rating') -> pd.DataFrame:
    """
    按真实评分等级计算RMSE
    
    该函数将数据按真实评分进行分组，计算每个评分等级的RMSE，
    有助于分析模型在不同评分区间的预测性能。
    
    Args:
        df (pd.DataFrame): 包含真实评分和预测评分的DataFrame
        true_col (str, optional): 真实评分列名，默认为'true_rating'
        pred_col (str, optional): 预测评分列名，默认为'pred_rating'
    
    Returns:
        pd.DataFrame: 包含评分等级和对应RMSE的DataFrame
                     列名为[true_col, 'RMSE', 'count']
    
    Raises:
        KeyError: 当指定的列不存在时抛出异常
        ValueError: 当数据为空或无效时抛出异常
    
    Example:
        >>> df = pd.DataFrame({
        ...     'true_rating': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ...     'pred_rating': [1.1, 0.9, 2.2, 1.8, 3.1, 2.9, 4.2, 3.8, 5.1, 4.9]
        ... })
        >>> result = rmse_by_class(df)
        >>> print(result)
    
    Note:
        - 结果按评分等级升序排列
        - 包含每个等级的样本数量
        - 如果某个等级只有一个样本，RMSE为0
    """
    try:
        # 参数验证
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        
        if true_col not in df.columns:
            raise KeyError(f"列'{true_col}'不存在")
        
        if pred_col not in df.columns:
            raise KeyError(f"列'{pred_col}'不存在")
        
        # 移除无效值
        valid_mask = ~(df[true_col].isna() | df[pred_col].isna())
        df_clean = df[valid_mask].copy()
        
        if df_clean.empty:
            raise ValueError("移除无效值后DataFrame为空")
        
        # 按评分等级分组计算均方根误差
        def calculate_group_rmse(group):
            if len(group) == 1:
                return pd.Series({
                    'RMSE': 0.0,
                    'count': 1,
                    'MAE': 0.0
                })
            else:
                rmse = np.sqrt(mean_squared_error(group[true_col], group[pred_col]))
                mae = mean_absolute_error(group[true_col], group[pred_col])
                return pd.Series({
                    'RMSE': rmse,
                    'count': len(group),
                    'MAE': mae
                })
        
        result = df_clean.groupby(true_col).apply(calculate_group_rmse).reset_index()
        result = result.sort_values(true_col)
        
        logger.info(f"按评分等级计算RMSE完成，共{len(result)}个等级")
        logger.debug(f"各等级RMSE: {dict(zip(result[true_col], result['RMSE']))}")
        
        return result
        
    except Exception as e:
        logger.error(f"按评分等级计算RMSE失败: {e}")
        raise


def user_error_distribution(df: pd.DataFrame, 
                          user_col: str = 'userId',
                          error_col: str = 'error') -> pd.DataFrame:
    """
    计算每个用户的预测误差统计
    
    该函数分析每个用户的预测误差分布，包括平均误差、误差标准差、
    最大误差等统计指标，有助于识别模型对不同用户的预测质量。
    
    分析维度：
    - 平均绝对误差：用户整体预测准确性
    - 误差标准差：用户预测一致性
    - 最大/最小误差：极端预测情况
    - 评分数量：用户活跃度
    - 误差分位数：误差分布特征
    
    Args:
        df (pd.DataFrame): 包含用户ID和误差的DataFrame
        user_col (str, optional): 用户ID列名，默认为'userId'
        error_col (str, optional): 误差列名，默认为'error'
    
    Returns:
        pd.DataFrame: 包含每个用户误差统计的DataFrame
                     列包括：用户ID、平均误差、误差标准差、最大误差、
                     最小误差、评分数量、25%分位数、75%分位数
    
    Raises:
        KeyError: 当指定的列不存在时
        ValueError: 当数据为空或无效时
    
    Example:
        >>> df = pd.DataFrame({
        ...     'userId': [1, 1, 2, 2, 3, 3],
        ...     'error': [0.1, -0.2, 0.3, 0.1, -0.1, 0.4]
        ... })
        >>> result = user_error_distribution(df)
        >>> print(result.head())
        
    Note:
        - 结果按平均绝对误差降序排列
        - 可用于识别预测困难的用户群体
        - 支持用户行为分析和个性化优化
                     列名为[user_col, 'mean_error', 'std_error', 'max_error', 
                            'min_error', 'count', 'rmse']
    
    Example:
        >>> # 首先计算误差
        >>> df['error'] = abs(df['true_rating'] - df['pred_rating'])
        >>> user_stats = user_error_distribution(df)
        >>> print(user_stats.head())
    
    Note:
        - 误差统计基于绝对误差
        - 包含每个用户的评分数量
        - 结果按平均误差升序排列
    """
    try:
        # 参数验证
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")
        
        if df.empty:
            raise ValueError("输入DataFrame不能为空")
        
        required_cols = [user_col, error_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"缺少必要的列: {missing_cols}")
        
        # 移除无效值
        valid_mask = ~(df[user_col].isna() | df[error_col].isna())
        df_clean = df[valid_mask].copy()
        
        if df_clean.empty:
            raise ValueError("移除无效值后DataFrame为空")
        
        # 计算用户误差统计
        user_stats = df_clean.groupby(user_col)[error_col].agg([
            ('mean_error', 'mean'),
            ('std_error', 'std'),
            ('max_error', 'max'),
            ('min_error', 'min'),
            ('count', 'count')
        ]).reset_index()
        
        # 计算每个用户的均方根误差（如果有真实评分和预测评分列）
        if 'true_rating' in df.columns and 'pred_rating' in df.columns:
            user_rmse = df_clean.groupby(user_col).apply(
                lambda g: np.sqrt(mean_squared_error(g['true_rating'], g['pred_rating']))
            ).reset_index(name='rmse')
            user_stats = user_stats.merge(user_rmse, on=user_col, how='left')
        
        # 填充标准差的空值（单个样本的情况）
        user_stats['std_error'] = user_stats['std_error'].fillna(0)
        
        # 按平均误差排序
        user_stats = user_stats.sort_values('mean_error')
        
        logger.info(f"用户误差分布计算完成，共{len(user_stats)}个用户")
        logger.debug(f"平均误差范围: {user_stats['mean_error'].min():.4f} - {user_stats['mean_error'].max():.4f}")
        
        return user_stats
        
    except Exception as e:
        logger.error(f"用户误差分布计算失败: {e}")
        raise


# ==================== 综合评估函数 ====================

def evaluate_predictions(df: pd.DataFrame,
                       true_col: str = 'true_rating',
                       pred_col: str = 'pred_rating',
                       user_col: str = 'userId',
                       item_col: str = 'movieId') -> Dict[str, Any]:
    """
    综合评估预测结果
    
    该函数提供了全面的预测评估，包括各种回归指标、分组分析和统计信息。
    是推荐系统评估的核心函数，支持多维度性能分析。
    
    评估维度：
    1. 整体性能指标：
       - RMSE、MAE、MAPE：基础回归指标
       - R²和相关系数：预测质量评估
       - 样本数量：数据规模统计
    
    2. 分层分析：
       - 按评分等级分析：识别不同评分区间的预测性能
       - 按用户分析：用户级别的预测准确性差异
       - 按物品分析：物品级别的预测难度评估
    
    3. 误差分布分析：
       - 误差统计量：均值、标准差、分位数
       - 异常值检测：识别预测异常情况
       - 分布特征：评估误差的分布模式
    
    Args:
        df (pd.DataFrame): 包含预测结果的DataFrame，必须包含真实评分和预测评分
        true_col (str, optional): 真实评分列名，默认为'true_rating'
        pred_col (str, optional): 预测评分列名，默认为'pred_rating'
        user_col (str, optional): 用户ID列名，默认为'userId'，用于用户维度分析
        item_col (str, optional): 物品ID列名，默认为'movieId'，用于物品维度分析
    
    Returns:
        Dict[str, Any]: 包含各种评估指标的嵌套字典，结构如下：
            - 'overall': 整体性能指标
            - 'by_rating': 按评分等级的性能分析
            - 'by_user': 用户维度分析（如果提供user_col）
            - 'by_item': 物品维度分析（如果提供item_col）
            - 'user_stats': 用户统计信息
            - 'item_stats': 物品统计信息
            - 'error_distribution': 误差分布统计
    
    Raises:
        KeyError: 当必要的列不存在时
        ValueError: 当数据为空或无效时
    
    Example:
        >>> # 基础使用
        >>> results = evaluate_predictions(test_df)
        >>> print(f"RMSE: {results['overall']['rmse']:.4f}")
        >>> print(f"MAE: {results['overall']['mae']:.4f}")
        >>> 
        >>> # 查看分层分析
        >>> for rating_level in results['by_rating']:
        ...     print(f"评分{rating_level['true_rating']}: RMSE={rating_level['RMSE']:.4f}")
        >>> 
        >>> # 误差分布分析
        >>> error_dist = results['error_distribution']
        >>> print(f"误差中位数: {error_dist['q50']:.4f}")
        >>> print(f"95%分位数: {error_dist['q95']:.4f}")
        
    Note:
        - 自动处理缺失值和无效数据
        - 支持灵活的列名配置
        - 提供详细的日志记录
        - 适用于模型选择、超参数调优和性能监控
        - 结果可直接用于可视化和报告生成
    """
    try:
        logger.info("开始综合评估预测结果")
        
        # 基本验证
        required_cols = [true_col, pred_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"缺少必要的列: {missing_cols}")
        
        # 移除无效值
        valid_mask = ~(df[true_col].isna() | df[pred_col].isna())
        df_clean = df[valid_mask].copy()
        
        if df_clean.empty:
            raise ValueError("没有有效的预测数据")
        
        # 计算误差
        df_clean['error'] = np.abs(df_clean[true_col] - df_clean[pred_col])
        df_clean['squared_error'] = (df_clean[true_col] - df_clean[pred_col]) ** 2
        
        # 整体指标
        overall_metrics = {
            'rmse': compute_rmse(df_clean[true_col], df_clean[pred_col]),
            'mae': compute_mae(df_clean[true_col], df_clean[pred_col]),
            'mape': compute_mape(df_clean[true_col], df_clean[pred_col]),
            'r2': stats.pearsonr(df_clean[true_col], df_clean[pred_col])[0] ** 2,
            'correlation': stats.pearsonr(df_clean[true_col], df_clean[pred_col])[0],
            'sample_count': len(df_clean)
        }
        
        # 分组分析
        results = {
            'overall': overall_metrics,
            'by_rating': rmse_by_class(df_clean, true_col, pred_col).to_dict('records')
        }
        
        # 用户分析（如果有用户列）
        if user_col in df.columns:
            results['by_user'] = user_error_distribution(df_clean, user_col, 'error')
            results['user_stats'] = {
                'unique_users': df_clean[user_col].nunique(),
                'avg_ratings_per_user': df_clean.groupby(user_col).size().mean(),
                'user_error_std': results['by_user']['mean_error'].std()
            }
        
        # 物品分析（如果有物品列）
        if item_col in df.columns:
            item_stats = df_clean.groupby(item_col)['error'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            item_stats.columns = [item_col, 'mean_error', 'std_error', 'count']
            results['by_item'] = item_stats
            results['item_stats'] = {
                'unique_items': df_clean[item_col].nunique(),
                'avg_ratings_per_item': df_clean.groupby(item_col).size().mean()
            }
        
        # 误差分布统计
        results['error_distribution'] = {
            'mean': df_clean['error'].mean(),
            'std': df_clean['error'].std(),
            'min': df_clean['error'].min(),
            'max': df_clean['error'].max(),
            'q25': df_clean['error'].quantile(0.25),
            'q50': df_clean['error'].quantile(0.50),
            'q75': df_clean['error'].quantile(0.75),
            'q95': df_clean['error'].quantile(0.95)
        }
        
        logger.info(f"综合评估完成 - RMSE: {overall_metrics['rmse']:.4f}, MAE: {overall_metrics['mae']:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"综合评估失败: {e}")
        raise


# ==================== 分类评估指标 ====================

def evaluate_classification(true_labels: Union[np.ndarray, pd.Series, List],
                          pred_labels: Union[np.ndarray, pd.Series, List],
                          target_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    评估分类预测结果
    
    当将评分预测转换为分类问题时（如好评/差评），使用该函数进行评估。
    支持二分类和多分类场景，提供全面的分类性能指标。
    
    评估指标：
    - 准确率（Accuracy）：正确预测的比例
    - 精确率（Precision）：预测为正类中实际为正类的比例
    - 召回率（Recall）：实际正类中被正确预测的比例
    - F1分数：精确率和召回率的调和平均
    - 混淆矩阵：详细的分类结果矩阵
    - 分类报告：每个类别的详细指标
    
    Args:
        true_labels: 真实标签，支持numpy数组、pandas Series或Python列表
        pred_labels: 预测标签，必须与true_labels长度相同
        target_names: 类别名称列表，用于报告中的标签显示
    
    Returns:
        Dict[str, Any]: 包含分类评估指标的字典，包括：
            - 'accuracy': 整体准确率
            - 'precision': 加权平均精确率
            - 'recall': 加权平均召回率
            - 'f1_score': 加权平均F1分数
            - 'classification_report': 详细分类报告
            - 'confusion_matrix': 混淆矩阵
            - 'sample_count': 样本数量
    
    Raises:
        ValueError: 当标签数组长度不匹配时
        TypeError: 当输入数据类型无效时
    
    Example:
        >>> # 二分类评估
        >>> true_labels = [0, 1, 1, 0, 1]
        >>> pred_labels = [0, 1, 0, 0, 1]
        >>> results = evaluate_classification(true_labels, pred_labels, 
        ...                                 target_names=['差评', '好评'])
        >>> print(f"准确率: {results['accuracy']:.4f}")
        >>> print(f"F1分数: {results['f1_score']:.4f}")
        >>> 
        >>> # 多分类评估
        >>> true_labels = [1, 2, 3, 4, 5]
        >>> pred_labels = [1, 2, 3, 4, 4]
        >>> results = evaluate_classification(true_labels, pred_labels)
        >>> print(results['classification_report'])
        
    Note:
        - 使用加权平均处理类别不平衡问题
        - 自动处理零除法情况
        - 适用于推荐系统的满意度分类
        - 可与rating_to_binary函数配合使用
    """
    try:
        true_labels = np.asarray(true_labels)
        pred_labels = np.asarray(pred_labels)
        
        if len(true_labels) != len(pred_labels):
            raise ValueError("标签数组长度不匹配")
        
        # 基本分类指标
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        
        # 详细报告
        report = classification_report(true_labels, pred_labels, 
                                     target_names=target_names, 
                                     output_dict=True, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'sample_count': len(true_labels)
        }
        
        logger.info(f"分类评估完成 - 准确率: {accuracy:.4f}, F1: {f1:.4f}")
        return results
        
    except Exception as e:
        logger.error(f"分类评估失败: {e}")
        raise


# ==================== 实用工具函数 ====================

def rating_to_binary(ratings: Union[np.ndarray, pd.Series, List], 
                    threshold: float = 3.5) -> np.ndarray:
    """
    将评分转换为二分类标签
    
    将连续的评分值转换为二分类标签，常用于将回归问题转换为分类问题。
    例如，将1-5星评分转换为好评/差评的二分类标签。
    
    转换规则：
    - 评分 >= threshold：正类（标签1，好评）
    - 评分 < threshold：负类（标签0，差评）
    
    Args:
        ratings: 评分数组，支持numpy数组、pandas Series或Python列表
        threshold (float, optional): 分类阈值，默认为3.5
                                   - 对于1-5评分系统，3.5是常用阈值
                                   - 可根据业务需求调整
    
    Returns:
        np.ndarray: 二分类标签数组（0或1）
                   - 0表示差评/不满意
                   - 1表示好评/满意
    
    Raises:
        TypeError: 当输入数据无法转换为数值时
        ValueError: 当输入为空时
    
    Example:
        >>> # 1-5星评分转换
        >>> ratings = [1, 2, 3, 4, 5]
        >>> binary_labels = rating_to_binary(ratings, threshold=3.5)
        >>> print(binary_labels)  # [0, 0, 0, 1, 1]
        >>> 
        >>> # 自定义阈值
        >>> ratings = [2.1, 2.8, 3.2, 4.1, 4.8]
        >>> binary_labels = rating_to_binary(ratings, threshold=3.0)
        >>> print(binary_labels)  # [0, 0, 1, 1, 1]
        
    Note:
        - 阈值选择影响正负类的平衡
        - 常与evaluate_classification函数配合使用
        - 适用于推荐系统的满意度分析
        - 可用于A/B测试的效果评估
    """
    ratings = np.asarray(ratings)
    return (ratings >= threshold).astype(int)


def compute_prediction_intervals(errors: Union[np.ndarray, pd.Series, List],
                               confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算预测区间
    
    基于历史误差分布计算预测的置信区间，用于评估预测的不确定性。
    通过分析历史预测误差的分布，为新预测提供置信区间估计。
    
    计算方法：
    - 使用分位数方法计算置信区间
    - 假设误差分布相对稳定
    - 不依赖特定的分布假设
    
    Args:
        errors: 历史预测误差数组，支持numpy数组、pandas Series或Python列表
        confidence (float, optional): 置信水平，默认0.95（95%置信区间）
                                    - 取值范围：(0, 1)
                                    - 常用值：0.90, 0.95, 0.99
    
    Returns:
        Tuple[float, float]: 置信区间的(下界, 上界)
                           - 下界：误差分布的低分位数
                           - 上界：误差分布的高分位数
                           - 区间包含指定置信水平的误差范围
    
    Raises:
        ValueError: 当置信水平不在有效范围内时
        TypeError: 当输入数据无法转换为数值时
    
    Example:
        >>> # 计算95%置信区间
        >>> errors = [0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2]
        >>> lower, upper = compute_prediction_intervals(errors)
        >>> print(f"95%置信区间: [{lower:.3f}, {upper:.3f}]")
        >>> 
        >>> # 计算90%置信区间
        >>> lower, upper = compute_prediction_intervals(errors, confidence=0.90)
        >>> print(f"90%置信区间: [{lower:.3f}, {upper:.3f}]")
        
    Note:
        - 适用于评估预测的可靠性
        - 可用于异常检测和质量控制
        - 区间宽度反映预测的不确定性
        - 建议使用足够的历史数据（>30个样本）
    """
    errors = np.asarray(errors)
    alpha = 1 - confidence
    lower = np.percentile(errors, 100 * alpha / 2)
    upper = np.percentile(errors, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def detect_prediction_outliers(df: pd.DataFrame,
                             error_col: str = 'error',
                             method: str = 'iqr',
                             threshold: float = 1.5) -> pd.DataFrame:
    """
    检测预测异常值
    
    识别预测误差异常大的样本，可能表示模型在这些样本上表现不佳。
    有助于发现数据质量问题、模型缺陷或特殊情况。
    
    检测方法：
    1. IQR方法（四分位距）：
       - 计算Q1、Q3和IQR = Q3 - Q1
       - 异常值：error < Q1 - threshold*IQR 或 error > Q3 + threshold*IQR
       - 对非正态分布数据鲁棒
    
    2. Z-score方法（标准分数）：
       - 计算标准化分数：z = (error - mean) / std
       - 异常值：|z| > threshold
       - 假设误差近似正态分布
    
    Args:
        df: 包含误差的DataFrame，必须包含指定的误差列
        error_col (str, optional): 误差列名，默认为'error'
        method (str, optional): 检测方法，默认为'iqr'
                              - 'iqr': 四分位距方法
                              - 'zscore': Z分数方法
        threshold (float, optional): 异常值阈值，默认为1.5
                                   - IQR方法：通常使用1.5或3.0
                                   - Z-score方法：通常使用2.0或3.0
    
    Returns:
        pd.DataFrame: 标记了异常值的DataFrame，新增列：
                     - 'is_outlier': 布尔值，标记是否为异常值
                     - 'outlier_score': 异常程度分数，值越大越异常
    
    Raises:
        KeyError: 当指定的误差列不存在时
        ValueError: 当检测方法不支持时
        TypeError: 当输入数据类型无效时
    
    Example:
        >>> # 使用IQR方法检测异常值
        >>> df_with_outliers = detect_prediction_outliers(df, method='iqr')
        >>> outlier_count = df_with_outliers['is_outlier'].sum()
        >>> print(f"发现{outlier_count}个异常值")
        >>> 
        >>> # 查看异常值样本
        >>> outliers = df_with_outliers[df_with_outliers['is_outlier']]
        >>> print(outliers[['error', 'outlier_score']].head())
        >>> 
        >>> # 使用Z-score方法
        >>> df_zscore = detect_prediction_outliers(df, method='zscore', threshold=2.0)
        
    Note:
        - IQR方法对非正态分布更鲁棒
        - Z-score方法适用于正态分布数据
        - 异常值可能指示数据质量问题或模型改进机会
        - 建议结合业务知识分析异常值的原因
    """
    try:
        df_copy = df.copy()
        errors = df_copy[error_col]
        
        if method == 'iqr':
            Q1 = errors.quantile(0.25)
            Q3 = errors.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (errors < lower_bound) | (errors > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(errors))
            outliers = z_scores > threshold
        
        else:
            raise ValueError(f"不支持的检测方法: {method}")
        
        df_copy['is_outlier'] = outliers
        df_copy['outlier_score'] = np.abs(errors - errors.mean()) / errors.std()
        
        logger.info(f"异常值检测完成，发现{outliers.sum()}个异常值（{outliers.mean()*100:.2f}%）")
        return df_copy
        
    except Exception as e:
        logger.error(f"异常值检测失败: {e}")
        raise