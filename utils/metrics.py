#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算模块

该模块提供了电影推荐系统的各种评估指标计算功能，主要包括：
1. 回归指标：RMSE、MAE、MAPE等
2. 分类指标：准确率、精确率、召回率、F1分数等
3. 排序指标：NDCG、MAP、MRR等
4. 分组分析：按用户、按电影、按评分等级的误差分析
5. 统计分析：误差分布、预测质量评估等

主要功能：
- 支持多种评估指标的计算
- 提供详细的分组分析功能
- 支持异常值检测和处理
- 提供可视化友好的结果格式
- 支持批量评估和增量评估

使用方式：
    from utils.metrics import compute_rmse, evaluate_predictions
    rmse = compute_rmse(true_ratings, pred_ratings)
    results = evaluate_predictions(df)

作者: 电影推荐系统开发团队
创建时间: 2024
最后修改: 2024
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
    RMSE对大误差更敏感，因为它对误差进行了平方处理。
    
    公式: RMSE = sqrt(mean((y_true - y_pred)^2))
    
    Args:
        true_ratings (Union[np.ndarray, pd.Series, List]): 真实评分数组
        pred_ratings (Union[np.ndarray, pd.Series, List]): 预测评分数组
    
    Returns:
        float: RMSE值，值越小表示预测越准确
    
    Raises:
        ValueError: 当输入数组长度不匹配或包含无效值时抛出异常
        TypeError: 当输入类型不正确时抛出异常
    
    Example:
        >>> true_ratings = [4.0, 3.5, 5.0, 2.0, 4.5]
        >>> pred_ratings = [3.8, 3.2, 4.9, 2.3, 4.2]
        >>> rmse = compute_rmse(true_ratings, pred_ratings)
        >>> print(f"RMSE: {rmse:.4f}")
        RMSE: 0.2449
    
    Note:
        - RMSE的单位与原始数据相同
        - RMSE值域为[0, +∞)，0表示完美预测
        - 对异常值敏感，一个很大的误差会显著影响RMSE
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
        
        # 计算RMSE
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
    
    MAE是另一个常用的回归评估指标，它计算预测值与真实值之间绝对误差的平均值。
    与RMSE相比，MAE对异常值不那么敏感。
    
    公式: MAE = mean(|y_true - y_pred|)
    
    Args:
        true_ratings (Union[np.ndarray, pd.Series, List]): 真实评分数组
        pred_ratings (Union[np.ndarray, pd.Series, List]): 预测评分数组
    
    Returns:
        float: MAE值，值越小表示预测越准确
    
    Example:
        >>> mae = compute_mae([4.0, 3.5, 5.0], [3.8, 3.2, 4.9])
        >>> print(f"MAE: {mae:.4f}")
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
    
    MAPE表示预测误差相对于真实值的百分比，便于理解预测的相对准确性。
    
    公式: MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Args:
        true_ratings: 真实评分数组
        pred_ratings: 预测评分数组
    
    Returns:
        float: MAPE值（百分比），值越小表示预测越准确
    
    Note:
        当真实值为0时，该点会被跳过以避免除零错误
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
        
        # 按评分等级分组计算RMSE
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
    
    Args:
        df (pd.DataFrame): 包含用户ID和误差的DataFrame
        user_col (str, optional): 用户ID列名，默认为'userId'
        error_col (str, optional): 误差列名，默认为'error'
    
    Returns:
        pd.DataFrame: 包含每个用户误差统计的DataFrame
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
        
        # 计算每个用户的RMSE（如果有真实评分和预测评分列）
        if 'true_rating' in df.columns and 'pred_rating' in df.columns:
            user_rmse = df_clean.groupby(user_col).apply(
                lambda g: np.sqrt(mean_squared_error(g['true_rating'], g['pred_rating']))
            ).reset_index(name='rmse')
            user_stats = user_stats.merge(user_rmse, on=user_col, how='left')
        
        # 填充标准差的NaN值（单个样本的情况）
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
    
    Args:
        df (pd.DataFrame): 包含预测结果的DataFrame
        true_col (str): 真实评分列名
        pred_col (str): 预测评分列名
        user_col (str): 用户ID列名
        item_col (str): 物品ID列名
    
    Returns:
        Dict[str, Any]: 包含各种评估指标的字典
    
    Example:
        >>> results = evaluate_predictions(test_df)
        >>> print(f"RMSE: {results['overall']['rmse']:.4f}")
        >>> print(f"MAE: {results['overall']['mae']:.4f}")
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
    
    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
        target_names: 类别名称列表
    
    Returns:
        Dict[str, Any]: 包含分类评估指标的字典
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
    
    Args:
        ratings: 评分数组
        threshold: 分类阈值，大于等于该值为正类
    
    Returns:
        np.ndarray: 二分类标签数组（0或1）
    """
    ratings = np.asarray(ratings)
    return (ratings >= threshold).astype(int)


def compute_prediction_intervals(errors: Union[np.ndarray, pd.Series, List],
                               confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算预测区间
    
    基于历史误差分布计算预测的置信区间。
    
    Args:
        errors: 历史预测误差数组
        confidence: 置信水平，默认0.95
    
    Returns:
        Tuple[float, float]: (下界, 上界)
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
    
    Args:
        df: 包含误差的DataFrame
        error_col: 误差列名
        method: 检测方法，'iqr'或'zscore'
        threshold: 异常值阈值
    
    Returns:
        pd.DataFrame: 标记了异常值的DataFrame
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