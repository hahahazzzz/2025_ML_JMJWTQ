#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练和评估模块

提供基于LightGBM的序数分类模型训练和预测功能

核心功能：
1. CORAL序数分类：将序数分类问题转换为多个二分类问题
2. 模型训练：训练多个LightGBM二分类器，每个预测评分是否>=某阈值
3. 预测推理：使用训练好的模型进行序数分类预测
4. 性能评估：计算准确率、RMSE、MAE等评估指标

序数分类原理：
- 对于10个类别（0-9），训练9个二分类器
- 第k个分类器预测样本是否属于类别k或更高
- 通过概率阈值将多个二分类结果转换为最终类别
- 保持了评分间的顺序关系，比传统多分类更适合评分预测

技术特点：
- 支持类别特征处理
- 提供详细的训练进度显示
- 包含完整的输入验证和错误处理
- 支持概率输出和自定义阈值
"""

import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Union
from lightgbm import LGBMClassifier
from tqdm import tqdm
from .model_utils import generate_ordinal_targets, convert_ordinal_to_class, validate_predictions

# 设置日志
logger = logging.getLogger(__name__)


class ProgressBarCallback:
    """
    LightGBM训练进度条回调类
    
    为LightGBM模型训练提供可视化进度条，显示训练迭代进度。
    
    Attributes:
        total (int): 总迭代次数
        pbar (tqdm): 进度条对象
    
    Note:
        - 自动在训练完成时关闭进度条
        - 提供详细的时间和速度信息
        - 支持自定义描述文本
    """
    
    def __init__(self, total: int, desc: str = "训练进度"):
        """
        初始化进度条回调
        
        Args:
            total: 总迭代次数，必须大于0
            desc: 进度条描述文本
            
        Raises:
            ValueError: 当total <= 0时抛出
        """
        if total <= 0:
            raise ValueError(f"总迭代次数必须大于0，当前值: {total}")
        
        self.total = total
        self.pbar = tqdm(
            total=total, 
            desc=desc, 
            unit="迭代",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        logger.debug(f"初始化进度条回调: 总迭代次数={total}")

    def __call__(self, env):
        """
        LightGBM回调函数接口
        
        Args:
            env: LightGBM环境对象，包含当前迭代信息
        """
        if env.iteration < self.total:
            self.pbar.update(1)
        
        if env.iteration + 1 == env.end_iteration:
            self.pbar.close()
            logger.debug(f"训练完成，总迭代次数: {env.iteration + 1}")


def train_models(X_train: np.ndarray, 
                y_train_raw: np.ndarray, 
                num_classes: int = 10, 
                n_estimators: int = 1000,
                learning_rate: float = 0.05,
                num_leaves: int = 63,
                seed: int = 42,
                categorical_features: Optional[List[str]] = None,
                verbose: bool = True) -> List[LGBMClassifier]:
    """
    训练CORAL风格的序数分类模型
    
    实现序数分类的核心函数，使用CORAL（Consistent Rank Logits）方法
    将序数分类问题转换为多个二分类问题。
    
    算法原理：
    1. 对于K个类别，训练K-1个二分类器
    2. 第i个分类器预测样本是否属于类别i或更高（y >= i）
    3. 每个分类器使用相同的特征，但目标变量不同
    4. 预测时通过概率阈值确定最终类别
    
    优势：
    - 保持类别间的顺序关系
    - 比传统多分类更适合评分预测
    - 可以利用序数信息提高预测精度
    
    Args:
        X_train (np.ndarray): 训练特征矩阵，形状为(n_samples, n_features)
        y_train_raw (np.ndarray): 训练标签，取值范围0到num_classes-1
        num_classes (int): 类别总数，对应评分0.5到5.0的10个等级
        n_estimators (int): 每个LightGBM模型的树数量
        learning_rate (float): 学习率，控制每棵树的贡献
        num_leaves (int): 每棵树的最大叶子节点数
        seed (int): 随机种子，确保结果可复现
        categorical_features (Optional[List[str]]): 类别特征名称列表
        verbose (bool): 是否显示详细训练信息
    
    Returns:
        List[LGBMClassifier]: 训练好的二分类器列表，长度为num_classes-1
    
    Raises:
        ValueError: 输入参数不合法时
        RuntimeError: 训练过程中出现错误时
    
    Example:
        >>> X_train = np.random.rand(1000, 50)
        >>> y_train = np.random.randint(0, 10, 1000)
        >>> models = train_models(X_train, y_train, num_classes=10)
        >>> print(f"训练了{len(models)}个模型")
    """
    if not isinstance(X_train, np.ndarray) or not isinstance(y_train_raw, np.ndarray):
        raise ValueError("输入的特征和标签必须是numpy数组")
    
    if len(X_train) != len(y_train_raw):
        raise ValueError(f"特征和标签的样本数量不匹配: {len(X_train)} vs {len(y_train_raw)}")
    
    if num_classes < 2:
        raise ValueError(f"类别数量必须至少为2，当前值: {num_classes}")
    
    if n_estimators <= 0 or learning_rate <= 0 or num_leaves <= 0:
        raise ValueError("模型参数必须为正数")
    
    if categorical_features is None:
        categorical_features = ['year_r', 'month_r', 'dayofweek_r']
    
    logger.info(f"开始训练序数分类模型: 样本数={len(X_train)}, 特征数={X_train.shape[1]}, 类别数={num_classes}")
    
    try:
        y_train_ordinal = generate_ordinal_targets(y_train_raw, num_classes)
        logger.info(f"序数目标矩阵生成完成，形状: {y_train_ordinal.shape}")
    except Exception as e:
        logger.error(f"生成序数目标时出错: {e}")
        raise RuntimeError(f"无法生成序数目标: {e}")
    
    models = []
    start_time = time.time()
    
    try:
        # 训练多个二分类器，每个预测评分是否>=某个阈值
        for k in range(num_classes - 1):
            threshold_rating = (k + 1) * 0.5  # 对应的评分阈值
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"训练第 {k + 1}/{num_classes - 1} 个子模型")
                print(f"任务: 预测评分是否 ≥ {threshold_rating:.1f} 星")
                print(f"正样本比例: {y_train_ordinal[:, k].mean():.3f}")
                print(f"{'='*60}")
            
            model_k = LGBMClassifier(
                objective='binary',
                random_state=seed,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                verbosity=-1,
                n_jobs=1,
                importance_type='gain'
            )
            
            logger.debug(f"开始训练第{k+1}个模型，目标变量统计: {np.bincount(y_train_ordinal[:, k])}")
            
            model_k.fit(
                X_train, 
                y_train_ordinal[:, k],
                categorical_feature=categorical_features
            )
            
            models.append(model_k)
            
            train_score = model_k.score(X_train, y_train_ordinal[:, k])
            logger.info(f"第{k+1}个模型训练完成，训练准确率: {train_score:.4f}")
    
    except Exception as e:
        logger.error(f"模型训练过程中出错: {e}")
        raise RuntimeError(f"模型训练失败: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"所有模型训练完成！")
        print(f"总用时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
        print(f"平均每个模型: {total_time/(num_classes-1):.2f} 秒")
        print(f"训练了 {len(models)} 个二分类器")
        print(f"{'='*60}")
    
    logger.info(f"序数分类模型训练完成: 用时{total_time:.2f}秒, 模型数量{len(models)}")
    
    return models


def predict(models: List[LGBMClassifier], 
           X_val: np.ndarray,
           threshold: float = 0.5,
           return_probabilities: bool = False,
           verbose: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    使用训练好的序数分类模型进行预测
    
    预测流程：
    1. 对每个二分类器获取正类概率
    2. 根据阈值将概率转换为二分类结果
    3. 通过序数逻辑将多个二分类结果合并为最终类别
    
    序数预测逻辑：
    - 如果所有分类器都预测为负类，则最终类别为0
    - 如果前k个分类器预测为正类，后续为负类，则最终类别为k
    - 如果所有分类器都预测为正类，则最终类别为最高类别
    
    Args:
        models (List[LGBMClassifier]): 训练好的二分类器列表
        X_val (np.ndarray): 验证集特征矩阵，形状为(n_samples, n_features)
        threshold (float): 二分类概率阈值，范围[0, 1]
        return_probabilities (bool): 是否同时返回概率矩阵
        verbose (bool): 是否显示预测过程信息
    
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - 如果return_probabilities=False：返回预测类别数组
            - 如果return_probabilities=True：返回(预测类别, 概率矩阵)元组
    
    Raises:
        ValueError: 输入参数不合法时
        RuntimeError: 预测过程中出现错误时
    
    Note:
        - 概率矩阵形状为(n_samples, n_models)
        - 预测类别范围为[0, n_models]
        - 阈值调整可以平衡精确率和召回率
    """
    if not models:
        raise ValueError("模型列表不能为空")
    
    if not isinstance(X_val, np.ndarray):
        raise ValueError("验证集特征必须是numpy数组")
    
    if not 0 <= threshold <= 1:
        raise ValueError(f"阈值必须在[0, 1]范围内，当前值: {threshold}")
    
    n_samples, n_features = X_val.shape
    n_models = len(models)
    
    if verbose:
        logger.info(f"开始预测: 样本数={n_samples}, 特征数={n_features}, 模型数={n_models}")
    
    try:
        # 收集所有模型的预测概率
        probabilities = np.zeros((n_samples, n_models))
        
        for k, model_k in enumerate(models):
            if verbose and k == 0:
                print(f"\n使用 {n_models} 个模型进行预测...")
            
            prob_k = model_k.predict_proba(X_val)[:, 1]  # 取正类概率
            probabilities[:, k] = prob_k
            
            if verbose:
                logger.debug(f"模型{k+1}预测完成，概率范围: [{prob_k.min():.3f}, {prob_k.max():.3f}]")
        
        # 将概率转换为序数分类结果
        class_predictions = convert_ordinal_to_class(probabilities, threshold)
        
        if not validate_predictions(class_predictions, expected_range=(0, n_models)):
            logger.warning("预测结果可能存在异常值")
        
        if verbose:
            unique_preds, counts = np.unique(class_predictions, return_counts=True)
            logger.info(f"预测完成，类别分布: {dict(zip(unique_preds, counts))}")
            print(f"预测完成！预测范围: [{class_predictions.min()}, {class_predictions.max()}]")
    
    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        raise RuntimeError(f"预测失败: {e}")
    
    if return_probabilities:
        return class_predictions, probabilities
    else:
        return class_predictions


def evaluate_models(models: List[LGBMClassifier], 
                   X_val: np.ndarray, 
                   y_val: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    评估序数分类模型的性能
    
    计算多种评估指标来全面评估模型性能，包括分类指标和回归指标。
    
    评估指标说明：
    1. 准确率(Accuracy)：预测类别完全正确的比例
    2. RMSE：将类别转换为评分后的均方根误差
    3. MAE：将类别转换为评分后的平均绝对误差
    
    Args:
        models (List[LGBMClassifier]): 训练好的模型列表
        X_val (np.ndarray): 验证集特征矩阵
        y_val (np.ndarray): 验证集真实标签
        threshold (float): 预测阈值
    
    Returns:
        dict: 包含以下评估指标的字典：
            - accuracy: 分类准确率
            - rmse: 均方根误差（基于评分）
            - mae: 平均绝对误差（基于评分）
            - n_samples: 验证样本数量
            - n_models: 模型数量
            - threshold: 使用的预测阈值
            - prediction_range: 预测值范围
            - true_range: 真实值范围
    
    Note:
        - RMSE和MAE通过label_to_rating函数将类别转换为评分
        - 评估结果会记录到日志中
        - 返回的字典可用于模型选择和超参数调优
    """
    y_pred, probabilities = predict(models, X_val, threshold, return_probabilities=True, verbose=False)
    
    accuracy = np.mean(y_pred == y_val)
    
    from .model_utils import label_to_rating
    y_val_ratings = label_to_rating(y_val)
    y_pred_ratings = label_to_rating(y_pred)
    rmse = np.sqrt(np.mean((y_val_ratings - y_pred_ratings) ** 2))
    
    mae = np.mean(np.abs(y_val_ratings - y_pred_ratings))
    
    evaluation_results = {
        'accuracy': accuracy,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(y_val),
        'n_models': len(models),
        'threshold': threshold,
        'prediction_range': (y_pred.min(), y_pred.max()),
        'true_range': (y_val.min(), y_val.max())
    }
    
    logger.info(f"模型评估完成: 准确率={accuracy:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return evaluation_results