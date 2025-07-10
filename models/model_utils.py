#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具函数模块

该模块提供了电影推荐系统中模型相关的工具函数，主要包括：
1. 序数分类（Ordinal Classification）相关函数
2. 评分与标签之间的转换函数
3. 预测结果处理函数

序数分类是一种特殊的分类任务，其中类别之间存在自然的顺序关系。
在电影评分预测中，评分1.0 < 1.5 < 2.0 < ... < 5.0，因此适合使用序数分类方法。

作者: 电影推荐系统开发团队
创建时间: 2024
最后修改: 2024
"""

import numpy as np
import logging
from typing import Union, Tuple

# 设置日志
logger = logging.getLogger(__name__)


def generate_ordinal_targets(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    生成序数分类的目标标签矩阵
    
    该函数将原始的评分标签转换为序数分类所需的二进制矩阵格式。
    序数分类的核心思想是将一个K类问题转换为K-1个二分类问题。
    
    算法原理:
        对于评分r和阈值t_i，如果r > t_i，则对应位置为1，否则为0。
        例如：评分3.5对应标签7，则前7个阈值都为1，后面为0：[1,1,1,1,1,1,1,0,0]
    
    Args:
        y (np.ndarray): 原始评分标签数组，形状为(n_samples,)
                       标签值范围应为0到num_classes-1
        num_classes (int, optional): 总类别数量，默认为10
                                   对应评分0.5, 1.0, 1.5, ..., 5.0
    
    Returns:
        np.ndarray: 序数目标矩阵，形状为(n_samples, num_classes-1)
                   每行表示一个样本的序数编码
    
    Raises:
        ValueError: 当输入参数不合法时抛出异常
    
    Example:
        >>> y = np.array([0, 2, 5, 9])  # 对应评分0.5, 1.5, 3.0, 5.0
        >>> ordinal_targets = generate_ordinal_targets(y, num_classes=10)
        >>> print(ordinal_targets.shape)  # (4, 9)
        >>> print(ordinal_targets[1])     # [1, 1, 0, 0, 0, 0, 0, 0, 0] (标签2)
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
    
    该函数将模型输出的序数概率矩阵转换为具体的类别标签。
    转换方法是统计每个样本中超过阈值的预测数量。
    
    算法原理:
        对于预测概率矩阵的每一行，统计大于threshold的元素个数，
        该个数即为预测的类别标签。
    
    Args:
        preds (np.ndarray): 序数预测概率矩阵，形状为(n_samples, num_classes-1)
                          每个元素表示对应阈值的预测概率
        threshold (float, optional): 二分类阈值，默认为0.5
                                   概率大于该值被认为是正类
    
    Returns:
        np.ndarray: 类别预测数组，形状为(n_samples,)
                   每个元素为预测的类别标签(0到num_classes-1)
    
    Raises:
        ValueError: 当输入参数不合法时抛出异常
    
    Example:
        >>> preds = np.array([[0.9, 0.8, 0.3, 0.1],  # 预测类别2
        ...                   [0.9, 0.9, 0.9, 0.9]])  # 预测类别4
        >>> classes = convert_ordinal_to_class(preds, threshold=0.5)
        >>> print(classes)  # [2, 4]
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
    
    该函数将MovieLens数据集中的评分(0.5-5.0)转换为模型训练所需的标签(0-9)。
    转换公式: label = (rating - 0.5) * 2
    
    评分到标签的映射关系:
        0.5 -> 0, 1.0 -> 1, 1.5 -> 2, ..., 4.5 -> 8, 5.0 -> 9
    
    Args:
        rating (Union[float, np.ndarray]): 评分值，范围为[0.5, 5.0]
                                         可以是单个值或数组
    
    Returns:
        Union[int, np.ndarray]: 对应的标签值，范围为[0, 9]
                               返回类型与输入类型一致
    
    Raises:
        ValueError: 当评分值超出有效范围时抛出异常
    
    Example:
        >>> rating_to_label(3.5)  # 返回 6
        >>> rating_to_label(np.array([1.0, 2.5, 4.0]))  # 返回 [1, 4, 7]
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
    
    该函数将模型预测的标签(0-9)转换回MovieLens评分系统的评分(0.5-5.0)。
    转换公式: rating = 0.5 + 0.5 * label
    
    标签到评分的映射关系:
        0 -> 0.5, 1 -> 1.0, 2 -> 1.5, ..., 8 -> 4.5, 9 -> 5.0
    
    Args:
        label (Union[int, np.ndarray]): 标签值，范围为[0, 9]
                                      可以是单个值或数组
    
    Returns:
        Union[float, np.ndarray]: 对应的评分值，范围为[0.5, 5.0]
                                返回类型与输入类型一致
    
    Raises:
        ValueError: 当标签值超出有效范围时抛出异常
    
    Example:
        >>> label_to_rating(6)  # 返回 3.5
        >>> label_to_rating(np.array([1, 4, 7]))  # 返回 [1.0, 2.5, 4.0]
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
    
    检查预测的评分是否在合理范围内，用于模型输出的质量控制。
    
    Args:
        predictions (np.ndarray): 预测的评分数组
        expected_range (Tuple[float, float], optional): 期望的评分范围，默认为(0.5, 5.0)
    
    Returns:
        bool: 如果所有预测都在有效范围内返回True，否则返回False
    
    Example:
        >>> preds = np.array([1.0, 2.5, 4.0, 5.0])
        >>> validate_predictions(preds)  # 返回 True
    """
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    min_val, max_val = expected_range
    
    # 检查是否有NaN或无穷值
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