#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练和评估模块

提供基于LightGBM的序数分类模型训练和预测功能
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
    """
    
    def __init__(self, total: int, desc: str = "训练进度"):
        """
        初始化进度条回调
        
        Args:
            total: 总的训练迭代次数
            desc: 进度条描述文本
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
        回调函数，在每次迭代时被调用
        """
        # 更新进度条
        if env.iteration < self.total:
            self.pbar.update(1)
        
        # 训练完成时关闭进度条
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
    训练多个LightGBM二分类模型以实现序数分类
    
    Args:
        X_train: 训练集特征矩阵
        y_train_raw: 训练集原始评分标签
        num_classes: 总类别数量
        n_estimators: 每个模型的树的数量
        learning_rate: 学习率
        num_leaves: 每棵树的叶子节点数
        seed: 随机种子
        categorical_features: 类别特征列表
        verbose: 是否显示详细训练信息
    
    Returns:
        训练好的模型列表
    """
    # ==================== 参数验证 ====================
    if not isinstance(X_train, np.ndarray) or not isinstance(y_train_raw, np.ndarray):
        raise ValueError("输入的特征和标签必须是numpy数组")
    
    if len(X_train) != len(y_train_raw):
        raise ValueError(f"特征和标签的样本数量不匹配: {len(X_train)} vs {len(y_train_raw)}")
    
    if num_classes < 2:
        raise ValueError(f"类别数量必须至少为2，当前值: {num_classes}")
    
    if n_estimators <= 0 or learning_rate <= 0 or num_leaves <= 0:
        raise ValueError("模型参数必须为正数")
    
    # 设置默认的类别特征
    if categorical_features is None:
        categorical_features = ['year_r', 'month_r', 'dayofweek_r']
    
    logger.info(f"开始训练序数分类模型: 样本数={len(X_train)}, 特征数={X_train.shape[1]}, 类别数={num_classes}")
    
    # ==================== 生成序数目标 ====================
    try:
        y_train_ordinal = generate_ordinal_targets(y_train_raw, num_classes)
        logger.info(f"序数目标矩阵生成完成，形状: {y_train_ordinal.shape}")
    except Exception as e:
        logger.error(f"生成序数目标时出错: {e}")
        raise RuntimeError(f"无法生成序数目标: {e}")
    
    # ==================== 模型训练 ====================
    models = []
    start_time = time.time()
    
    try:
        for k in range(num_classes - 1):
            threshold_rating = (k + 1) * 0.5  # 对应的评分阈值
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"训练第 {k + 1}/{num_classes - 1} 个子模型")
                print(f"任务: 预测评分是否 ≥ {threshold_rating:.1f} 星")
                print(f"正样本比例: {y_train_ordinal[:, k].mean():.3f}")
                print(f"{'='*60}")
            
            # 创建并配置模型
            model_k = LGBMClassifier(
                objective='binary',           # 二分类任务
                random_state=seed,           # 随机种子
                n_estimators=n_estimators,   # 树的数量
                learning_rate=learning_rate, # 学习率
                num_leaves=num_leaves,       # 叶子节点数
                verbosity=-1,               # 静默模式
                n_jobs=1,                   # 使用单线程避免段错误
                importance_type='gain'       # 特征重要性计算方式
            )
            
            # 训练模型
            logger.debug(f"开始训练第{k+1}个模型，目标变量统计: {np.bincount(y_train_ordinal[:, k])}")
            
            model_k.fit(
                X_train, 
                y_train_ordinal[:, k],
                categorical_feature=categorical_features
            )
            
            models.append(model_k)
            
            # 记录训练信息
            train_score = model_k.score(X_train, y_train_ordinal[:, k])
            logger.info(f"第{k+1}个模型训练完成，训练准确率: {train_score:.4f}")
    
    except Exception as e:
        logger.error(f"模型训练过程中出错: {e}")
        raise RuntimeError(f"模型训练失败: {e}")
    
    # ==================== 训练完成 ====================
    end_time = time.time()
    total_time = end_time - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎉 所有模型训练完成！")
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
    
    该函数使用训练好的多个二分类器进行序数分类预测。
    预测过程包括：
    1. 使用每个二分类器预测概率
    2. 根据阈值将概率转换为二分类结果
    3. 统计超过阈值的分类器数量得到最终类别
    
    Args:
        models (List[LGBMClassifier]): 训练好的模型列表
        X_val (np.ndarray): 验证集特征矩阵，形状为(n_samples, n_features)
        threshold (float, optional): 二分类阈值，默认为0.5
        return_probabilities (bool, optional): 是否返回概率矩阵，默认为False
        verbose (bool, optional): 是否显示预测信息，默认为True
    
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            如果return_probabilities=False: 返回预测的类别数组
            如果return_probabilities=True: 返回(类别数组, 概率矩阵)的元组
    
    Raises:
        ValueError: 当输入参数不合法时抛出异常
        RuntimeError: 当预测过程中出现错误时抛出异常
    
    Example:
        >>> X_val = np.random.rand(100, 50)
        >>> predictions = predict(models, X_val)
        >>> print(f"预测了{len(predictions)}个样本")
    """
    # ==================== 参数验证 ====================
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
    
    # ==================== 预测过程 ====================
    try:
        # 初始化概率矩阵
        probabilities = np.zeros((n_samples, n_models))
        
        # 使用每个模型进行预测
        for k, model_k in enumerate(models):
            if verbose and k == 0:
                print(f"\n使用 {n_models} 个模型进行预测...")
            
            # 预测概率（取正类概率）
            prob_k = model_k.predict_proba(X_val)[:, 1]
            probabilities[:, k] = prob_k
            
            if verbose:
                logger.debug(f"模型{k+1}预测完成，概率范围: [{prob_k.min():.3f}, {prob_k.max():.3f}]")
        
        # 转换为类别预测
        class_predictions = convert_ordinal_to_class(probabilities, threshold)
        
        # 验证预测结果
        if not validate_predictions(class_predictions, expected_range=(0, n_models)):
            logger.warning("预测结果可能存在异常值")
        
        if verbose:
            unique_preds, counts = np.unique(class_predictions, return_counts=True)
            logger.info(f"预测完成，类别分布: {dict(zip(unique_preds, counts))}")
            print(f"预测完成！预测范围: [{class_predictions.min()}, {class_predictions.max()}]")
    
    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        raise RuntimeError(f"预测失败: {e}")
    
    # ==================== 返回结果 ====================
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
    
    该函数提供了全面的模型评估指标，包括准确率、各类别的精确率和召回率等。
    
    Args:
        models (List[LGBMClassifier]): 训练好的模型列表
        X_val (np.ndarray): 验证集特征
        y_val (np.ndarray): 验证集真实标签
        threshold (float, optional): 预测阈值，默认为0.5
    
    Returns:
        dict: 包含各种评估指标的字典
    
    Example:
        >>> metrics = evaluate_models(models, X_val, y_val)
        >>> print(f"准确率: {metrics['accuracy']:.4f}")
    """
    # 进行预测
    y_pred, probabilities = predict(models, X_val, threshold, return_probabilities=True, verbose=False)
    
    # 计算基本指标
    accuracy = np.mean(y_pred == y_val)
    
    # 计算RMSE（将类别转换回评分）
    from .model_utils import label_to_rating
    y_val_ratings = label_to_rating(y_val)
    y_pred_ratings = label_to_rating(y_pred)
    rmse = np.sqrt(np.mean((y_val_ratings - y_pred_ratings) ** 2))
    
    # 计算MAE
    mae = np.mean(np.abs(y_val_ratings - y_pred_ratings))
    
    # 构建评估结果
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