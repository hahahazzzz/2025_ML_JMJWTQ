#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具函数模块

提供序数分类和评分标签转换工具
"""

import numpy as np
import logging
from typing import Union, Tuple

# 设置日志
logger = logging.getLogger(__name__)


def generate_ordinal_targets(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    生成序数分类的目标标签矩阵
    
    Args:
        y: 原始评分标签数组
        num_classes: 总类别数量
    
    Returns:
        序数目标矩阵
    """
    # 参数验证
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    if len(y) == 0:
        raise ValueError("输入标签数组不能为空")
    
    if num_classes < 2:
        raise ValueError(f"类别数量必须至少为2，当前值: {num_classes}")
    
    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError(f"标签值必须在[0, {num_classes-1}]范围内，当前范围: [{y.min()}, {y.max()}]")
    
    logger.debug(f"生成序数目标矩阵: 样本数={len(y)}, 类别数={num_classes}")
    
    # 创建序数目标矩阵
    y_ordinal = np.zeros((len(y), num_classes - 1), dtype=np.int32)
    
    # 填充序数矩阵：对于每个阈值i，如果标签y > i，则对应位置为1
    for i in range(num_classes - 1):
        y_ordinal[:, i] = (y > i).astype(np.int32)
    
    logger.debug(f"序数目标矩阵生成完成，形状: {y_ordinal.shape}")
    return y_ordinal


def convert_ordinal_to_class(preds: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    将序数分类的预测概率转换为最终的类别预测
    
    Args:
        preds: 序数预测概率矩阵
        threshold: 二分类阈值
    
    Returns:
        类别预测数组
    """
    # 参数验证
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    
    if len(preds.shape) != 2:
        raise ValueError(f"预测矩阵必须是2维数组，当前维度: {len(preds.shape)}")
    
    if not 0 <= threshold <= 1:
        raise ValueError(f"阈值必须在[0, 1]范围内，当前值: {threshold}")
    
    logger.debug(f"转换序数预测为类别: 预测矩阵形状={preds.shape}, 阈值={threshold}")
    
    # 统计每行中大于阈值的元素个数
    class_preds = (preds > threshold).sum(axis=1)
    
    logger.debug(f"类别预测转换完成，预测范围: [{class_preds.min()}, {class_preds.max()}]")
    return class_preds


def rating_to_label(rating: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """
    将连续的评分值转换为离散的标签值
    
    Args:
        rating: 评分值
    
    Returns:
        对应的标签值
    """
    # 处理数组输入
    if isinstance(rating, np.ndarray):
        # 验证评分范围
        if np.any(rating < 0.5) or np.any(rating > 5.0):
            invalid_ratings = rating[(rating < 0.5) | (rating > 5.0)]
            raise ValueError(f"评分值必须在[0.5, 5.0]范围内，发现无效值: {invalid_ratings}")
        
        # 转换为标签
        labels = np.round((rating - 0.5) * 2).astype(int)
        # 确保标签在有效范围内
        labels = np.clip(labels, 0, 9)
        return labels
    
    # 处理单个值输入
    else:
        # 验证评分范围
        if not (0.5 <= rating <= 5.0):
            raise ValueError(f"评分值必须在[0.5, 5.0]范围内，当前值: {rating}")
        
        # 转换为标签并限制范围
        label = int(round((rating - 0.5) * 2))
        return max(0, min(9, label))


def label_to_rating(label: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    将离散的标签值转换为连续的评分值
    
    Args:
        label: 标签值
    
    Returns:
        对应的评分值
    """
    # 处理数组输入
    if isinstance(label, np.ndarray):
        # 验证标签范围
        if np.any(label < 0) or np.any(label > 9):
            invalid_labels = label[(label < 0) | (label > 9)]
            raise ValueError(f"标签值必须在[0, 9]范围内，发现无效值: {invalid_labels}")
        
        # 转换为评分
        ratings = 0.5 + 0.5 * label.astype(float)
        return ratings
    
    # 处理单个值输入
    else:
        # 验证标签范围
        if not (0 <= label <= 9):
            raise ValueError(f"标签值必须在[0, 9]范围内，当前值: {label}")
        
        # 转换为评分
        rating = 0.5 + 0.5 * label
        return rating


def validate_predictions(predictions: np.ndarray, 
                        expected_range: Tuple[float, float] = (0.5, 5.0)) -> bool:
    """
    验证预测结果的有效性
    
    Args:
        predictions: 预测的评分数组
        expected_range: 期望的评分范围
    
    Returns:
        如果所有预测都在有效范围内返回True，否则返回False
    """
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    min_val, max_val = expected_range
    
    # 检查是否有空值或无穷值
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        logger.warning("预测结果中包含NaN或无穷值")
        return False
    
    # 检查范围
    if np.any(predictions < min_val) or np.any(predictions > max_val):
        out_of_range = predictions[(predictions < min_val) | (predictions > max_val)]
        logger.warning(f"预测结果超出范围[{min_val}, {max_val}]: {out_of_range[:5]}...")
        return False
    
    logger.debug(f"预测结果验证通过: 范围[{predictions.min():.2f}, {predictions.max():.2f}]")
    return True