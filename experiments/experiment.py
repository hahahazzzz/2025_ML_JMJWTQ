#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验管理模块

该模块提供了电影推荐系统的实验管理功能，主要包括：
1. 实验配置管理：自动保存和加载实验配置
2. 结果记录：支持各种指标的记录和存储
3. 数据保存：自动保存DataFrame、图表等实验产物
4. 实验比较：支持多个实验之间的性能比较
5. 可视化：自动生成实验结果的可视化图表
6. 版本控制：基于时间戳的实验版本管理

主要功能：
- 自动创建实验目录和文件结构
- 支持JSON格式的配置和结果存储
- 提供丰富的可视化功能
- 支持实验间的横向比较
- 集成日志记录系统
- 支持增量式结果更新

使用方式：
    from experiments.experiment import Experiment
    
    # 创建实验
    exp = Experiment("LightGBM_Test", config)
    
    # 记录指标
    exp.log_metric("rmse", 0.85)
    exp.log_metric("mae", 0.67)
    
    # 保存结果
    exp.save_results()
    exp.save_dataframe(predictions_df, "predictions.csv")

实验目录结构：
    experiments/
    ├── ExperimentName_YYYYMMDD_HHMMSS/
    │   ├── config.json          # 实验配置
    │   ├── results.json         # 实验结果
    │   ├── predictions.csv      # 预测结果
    │   ├── *.png               # 可视化图表
    │   └── logs/               # 实验日志

作者: 电影推荐系统开发团队
创建时间: 2024
最后修改: 2024
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.logger import logger


class Experiment:
    """
    实验管理类
    
    该类提供了完整的机器学习实验管理功能，包括配置管理、结果记录、
    数据保存、可视化和实验比较等功能。每个实验都有唯一的ID和独立的目录。
    
    主要特性：
    - 自动化的实验目录管理
    - 配置和结果的JSON序列化
    - 丰富的可视化功能
    - 支持多实验比较
    - 集成日志系统
    - 类型安全的接口
    
    Attributes:
        name (str): 实验名称
        config (Dict): 实验配置
        timestamp (str): 实验创建时间戳
        experiment_id (str): 唯一的实验标识符
        experiment_dir (str): 实验目录路径
        results (Dict): 实验结果字典
    
    Example:
        >>> config = {"model": "LightGBM", "learning_rate": 0.1}
        >>> exp = Experiment("MovieRec_Test", config)
        >>> exp.log_metric("rmse", 0.85)
        >>> exp.save_results()
    """
    
    def __init__(self, 
                 name: str, 
                 config: Dict[str, Any], 
                 base_dir: str = "experiments"):
        """
        初始化实验管理器
        
        创建一个新的实验实例，自动生成唯一的实验ID，创建实验目录，
        并保存实验配置。每个实验都有独立的目录结构用于存储所有相关文件。
        
        Args:
            name (str): 实验名称，用于标识实验类型或目的
                       建议使用描述性名称，如"LightGBM_Baseline"、"DeepFM_Tuned"等
            config (Dict[str, Any]): 实验配置字典，包含所有实验参数
                                   如模型参数、数据参数、训练参数等
            base_dir (str, optional): 实验结果保存的基础目录，默认为"experiments"
        
        Raises:
            OSError: 当无法创建实验目录时抛出异常
            TypeError: 当配置参数类型不正确时抛出异常
            ValueError: 当实验名称为空时抛出异常
        
        Example:
            >>> # 基本用法
            >>> config = {
            ...     "model_type": "LightGBM",
            ...     "n_estimators": 100,
            ...     "learning_rate": 0.1,
            ...     "max_depth": 6
            ... }
            >>> exp = Experiment("LightGBM_Baseline", config)
            >>> 
            >>> # 自定义基础目录
            >>> exp = Experiment("Test_Run", config, base_dir="my_experiments")
        
        Note:
            - 实验ID格式为: {name}_{YYYYMMDD_HHMMSS}
            - 实验目录会自动创建，包含必要的子目录
            - 配置文件会立即保存为JSON格式
            - 所有操作都会记录到日志中
        """
        # 参数验证
        if not isinstance(name, str) or not name.strip():
            raise ValueError("实验名称不能为空")
        
        if not isinstance(config, dict):
            raise TypeError("配置参数必须是字典类型")
        
        if not isinstance(base_dir, str) or not base_dir.strip():
            raise ValueError("基础目录路径不能为空")
        
        # 初始化基本属性
        self.name = name.strip()
        self.config = config.copy()  # 创建配置的副本以避免外部修改
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{self.name}_{self.timestamp}"
        
        # 创建实验目录结构
        try:
            self.base_dir = base_dir
            self.experiment_dir = os.path.join(base_dir, self.experiment_id)
            Path(self.experiment_dir).mkdir(parents=True, exist_ok=True)
            
            # 创建子目录
            self._create_subdirectories()
            
        except OSError as e:
            raise OSError(f"无法创建实验目录 {self.experiment_dir}: {e}")
        
        # 保存配置
        self.save_config()
        
        # 初始化结果字典
        self.results = {}
        
        # 记录实验创建信息
        logger.info(f"创建实验: {self.experiment_id}")
        logger.info(f"实验目录: {self.experiment_dir}")
        logger.debug(f"实验配置: {self.config}")
    
    def _create_subdirectories(self) -> None:
        """
        创建实验子目录结构
        
        创建标准的实验目录结构，包括图表、数据、模型等子目录。
        
        Raises:
            OSError: 当无法创建子目录时抛出异常
        """
        subdirs = ['plots', 'data', 'models', 'logs']
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.experiment_dir, subdir)
            try:
                Path(subdir_path).mkdir(exist_ok=True)
            except OSError as e:
                logger.warning(f"无法创建子目录 {subdir}: {e}")
    
    def save_config(self) -> None:
        """
        保存实验配置到JSON文件
        
        将当前实验的配置参数序列化为JSON格式并保存到实验目录中。
        配置文件用于记录实验的所有参数设置，便于结果复现和实验对比。
        
        文件保存位置: {experiment_dir}/config.json
        
        Raises:
            IOError: 当无法写入配置文件时抛出异常
            TypeError: 当配置包含不可序列化的对象时抛出异常
        
        Example:
            >>> exp = Experiment("test", {"param1": 1, "param2": "value"})
            >>> exp.save_config()  # 自动调用，也可手动调用
        
        Note:
            - 使用UTF-8编码确保中文字符正确保存
            - JSON格式化输出，便于人工阅读
            - 自动在实验初始化时调用
            - 支持嵌套字典和列表结构
            - 自动处理不可序列化的对象
        """
        config_path = os.path.join(self.experiment_dir, "config.json")
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                # 将配置转换为可序列化的字典
                serializable_config = self._make_serializable(self.config)
                json.dump(serializable_config, f, indent=4, ensure_ascii=False)
            logger.info(f"保存实验配置到: {config_path}")
            
        except (IOError, TypeError) as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        将对象转换为JSON可序列化的格式
        
        递归地将复杂对象转换为JSON可序列化的基本类型。
        处理嵌套字典、列表以及自定义对象。
        
        Args:
            obj (Any): 需要序列化的对象
        
        Returns:
            Any: JSON可序列化的对象
        
        Note:
            - 字典和列表会递归处理
            - 自定义对象会转换为字符串表示
            - 基本类型直接返回
            - 处理numpy数组、pandas对象等常见类型
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy数组
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(vars(obj))
        else:
            try:
                # 尝试JSON序列化
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                # 如果无法序列化，则返回字符串表示
                return str(obj)
    
    def log_metric(self, metric_name: str, value: Union[float, int, str], step: Optional[int] = None) -> None:
        """
        记录实验指标
        
        记录单个实验指标到结果字典中。支持数值型和字符串型指标，
        可以记录训练过程中的多个时间点的指标值。
        
        Args:
            metric_name (str): 指标名称，建议使用描述性名称
                             如"rmse", "mae", "accuracy", "training_time"等
            value (Union[float, int, str]): 指标值
                                          支持数值型指标和字符串型描述
            step (Optional[int]): 训练步骤或轮次，用于记录指标变化过程
        
        Raises:
            ValueError: 当指标名称为空时抛出异常
            TypeError: 当指标值类型不支持时抛出异常
        
        Example:
            >>> exp.log_metric("rmse", 0.85)
            >>> exp.log_metric("mae", 0.67, step=100)
            >>> exp.log_metric("model_type", "LightGBM")
            >>> exp.log_metric("training_time", 120.5)
        
        Note:
            - 相同名称的指标会以列表形式累积存储
            - 指标会立即记录到内存中的results字典
            - 需要调用save_results()才能持久化到文件
            - 支持增量式添加指标
            - step参数用于绘制指标变化趋势图
        """
        # 参数验证
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("指标名称不能为空")
        
        if not isinstance(value, (int, float, str)):
            raise TypeError(f"不支持的指标值类型: {type(value)}")
        
        # 初始化指标列表
        if metric_name not in self.results:
            self.results[metric_name] = []
        
        # 构建结果记录
        result = {"value": value}
        if step is not None:
            result["step"] = step
        
        # 记录指标
        self.results[metric_name].append(result)
        logger.info(f"记录指标: {metric_name} = {value}")
    
    def save_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """
        保存实验结果到JSON文件
        
        将所有记录的实验指标序列化为JSON格式并保存到实验目录中。
        结果文件包含所有通过log_metric()记录的指标以及额外提供的结果。
        
        Args:
            results (Optional[Dict[str, Any]]): 额外的结果字典，会合并到现有结果中
                                              用于一次性添加多个指标或复杂结果对象
        
        文件保存位置: {experiment_dir}/results.json
        
        Raises:
            IOError: 当无法写入结果文件时抛出异常
            TypeError: 当结果包含不可序列化对象时抛出异常
        
        Example:
            >>> exp.log_metric("rmse", 0.85)
            >>> exp.log_metric("mae", 0.67)
            >>> exp.save_results()
            >>> 
            >>> # 保存额外结果
            >>> extra_results = {"feature_importance": [0.1, 0.2, 0.3]}
            >>> exp.save_results(extra_results)
        
        Note:
            - 自动处理中文字符编码
            - JSON格式化输出，便于阅读
            - 包含实验的所有指标数据
            - 可以多次调用以更新结果文件
            - 自动处理不可序列化的复杂对象
        """
        # 如果提供了额外的结果，合并到self.results中
        if results is not None:
            if not isinstance(results, dict):
                raise TypeError("额外结果必须是字典类型")
            self.results.update(results)
            
        results_path = os.path.join(self.experiment_dir, "results.json")
        
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                # 使用_make_serializable处理不可序列化的对象
                serializable_results = self._make_serializable(self.results)
                json.dump(serializable_results, f, indent=4, ensure_ascii=False)
            logger.info(f"保存实验结果到: {results_path}")
            
        except (IOError, TypeError) as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """
        保存DataFrame到CSV文件
        
        将pandas DataFrame保存为CSV格式到实验目录中。
        支持自动处理中文字符和索引设置。
        
        Args:
            df (pd.DataFrame): 要保存的pandas DataFrame对象
            filename (str): 文件名，建议包含.csv扩展名
                          如"predictions.csv", "feature_importance.csv"等
        
        Raises:
            ValueError: 当DataFrame为空或filename为空时抛出异常
            IOError: 当无法写入文件时抛出异常
            TypeError: 当df不是DataFrame类型时抛出异常
        
        Example:
            >>> predictions_df = pd.DataFrame({"user_id": [1, 2], "rating": [4.5, 3.2]})
            >>> exp.save_dataframe(predictions_df, "predictions.csv")
            >>> 
            >>> feature_df = pd.DataFrame({"feature": ["age", "genre"], "importance": [0.3, 0.7]})
            >>> exp.save_dataframe(feature_df, "feature_importance.csv")
        
        Note:
            - 默认不保存行索引
            - 使用UTF-8编码确保中文字符正确保存
            - 文件保存在实验目录的data子目录中
            - 支持大型DataFrame的高效保存
        """
        # 参数验证
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df参数必须是pandas DataFrame类型")
        
        if df.empty:
            logger.warning("DataFrame为空，仍将保存空文件")
        
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("文件名不能为空")
        
        # 确保文件保存在data子目录中
        data_dir = os.path.join(self.experiment_dir, "data")
        file_path = os.path.join(data_dir, filename.strip())
        
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"保存DataFrame到: {file_path}")
            logger.debug(f"DataFrame形状: {df.shape}")
            
        except IOError as e:
            logger.error(f"保存DataFrame失败: {e}")
            raise
    
    def save_figure(self, fig: plt.Figure, filename: str, close_fig: bool = True, **kwargs) -> None:
        """
        保存matplotlib图表
        
        将matplotlib图表保存为高质量图片文件到实验目录中。
        支持多种图片格式和自定义保存参数。
        
        Args:
            fig (plt.Figure): matplotlib图表对象
            filename (str): 文件名，建议包含扩展名
                          支持.png, .jpg, .pdf, .svg等格式
            close_fig (bool): 是否在保存后关闭图表，默认为True
            **kwargs: 传递给fig.savefig()的额外参数
                     如dpi, bbox_inches, facecolor等
        
        Raises:
            ValueError: 当filename为空时抛出异常
            IOError: 当无法保存文件时抛出异常
            TypeError: 当fig不是Figure类型时抛出异常
        
        Example:
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3], [1, 4, 2])
            >>> exp.save_figure(fig, "training_curve.png")
            >>> 
            >>> # 自定义保存参数
            >>> exp.save_figure(fig, "high_res.png", dpi=600, facecolor='white')
        
        Note:
            - 默认使用300 DPI高分辨率
            - 自动调整边界以包含所有元素
            - 图片保存在实验目录的plots子目录中
            - 支持矢量格式(PDF, SVG)和位图格式(PNG, JPG)
            - 默认保存后关闭图表以释放内存
        """
        # 参数验证
        if not hasattr(fig, 'savefig'):
            raise TypeError("fig参数必须是matplotlib Figure对象")
        
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("文件名不能为空")
        
        # 确保文件保存在plots子目录中
        plots_dir = os.path.join(self.experiment_dir, "plots")
        file_path = os.path.join(plots_dir, filename.strip())
        
        # 设置默认保存参数
        save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_kwargs.update(kwargs)
        
        try:
            fig.savefig(file_path, **save_kwargs)
            logger.info(f"保存图表到: {file_path}")
            
            # 可选择性关闭图表
            if close_fig:
                plt.close(fig)
                
        except IOError as e:
            logger.error(f"保存图表失败: {e}")
            raise
    
    def plot_metrics(self, metric_name: str) -> Optional[plt.Figure]:
        """
        绘制指标变化图
        
        为指定的指标绘制变化趋势图，支持训练过程中的指标监控。
        自动处理步骤信息和数值格式，生成专业的可视化图表。
        
        Args:
            metric_name (str): 指标名称，必须是已记录的指标
                             如"rmse", "mae", "loss"等
        
        Returns:
            Optional[plt.Figure]: matplotlib图表对象，如果指标不存在则返回None
        
        Raises:
            ValueError: 当指标名称为空时抛出异常
        
        Example:
            >>> # 记录多个时间点的指标
            >>> for epoch in range(10):
            ...     exp.log_metric("loss", loss_values[epoch], step=epoch)
            >>> 
            >>> # 绘制变化趋势
            >>> fig = exp.plot_metrics("loss")
        
        Note:
            - 自动提取步骤信息，如果没有则使用序号
            - 支持数值型指标的趋势可视化
            - 图表自动保存到plots目录
            - 包含网格线和标记点便于读取数值
            - 自动设置合适的图表尺寸和样式
        """
        # 参数验证
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("指标名称不能为空")
        
        metric_name = metric_name.strip()
        
        # 检查指标是否存在
        if metric_name not in self.results or not self.results[metric_name]:
            logger.warning(f"没有找到指标: {metric_name}")
            return None
        
        try:
            # 提取指标值和步骤
            data = self.results[metric_name]
            
            # 检查数据格式并构建DataFrame
            if isinstance(data[0], dict) and "step" in data[0]:
                df = pd.DataFrame([(item["step"], item["value"]) for item in data],
                                 columns=["step", metric_name])
                x = "step"
            else:
                values = [item["value"] if isinstance(item, dict) else item for item in data]
                df = pd.DataFrame(values, columns=[metric_name])
                df["index"] = range(len(df))
                x = "index"
            
            # 验证数值类型
            if not pd.api.types.is_numeric_dtype(df[metric_name]):
                logger.warning(f"指标 {metric_name} 包含非数值数据，无法绘制趋势图")
                return None
            
            # 绘制图表
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=df, x=x, y=metric_name, marker="o", ax=ax, linewidth=2, markersize=6)
            ax.set_title(f"{metric_name} 变化趋势", fontsize=14, fontweight='bold')
            ax.set_xlabel('步骤' if x == 'step' else '序号', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 保存图表
            self.save_figure(fig, f"{metric_name}_trend.png", close_fig=False)
            
            logger.info(f"绘制指标 {metric_name} 趋势图，包含 {len(df)} 个数据点")
            return fig
            
        except Exception as e:
            logger.error(f"绘制指标 {metric_name} 趋势图失败: {e}")
            return None
    
    def compare_experiments(self, 
                          other_experiments: List['Experiment'], 
                          metric_name: str,
                          save_comparison: bool = True) -> Optional[plt.Figure]:
        """
        比较多个实验的指标
        
        将当前实验与其他实验的指定指标进行比较，生成对比可视化图表。
        支持趋势对比和性能对比两种模式，自动选择最适合的可视化方式。
        
        Args:
            other_experiments (List[Experiment]): 其他实验对象列表
                                                用于比较的实验实例
            metric_name (str): 要比较的指标名称
                             如"rmse", "mae", "accuracy"等
            save_comparison (bool): 是否保存比较图表，默认为True
        
        Returns:
            Optional[plt.Figure]: matplotlib图表对象，如果比较失败则返回None
        
        Raises:
            ValueError: 当参数无效时抛出异常
            TypeError: 当other_experiments类型不正确时抛出异常
        
        Example:
            >>> exp1 = Experiment("Baseline", config1)
            >>> exp2 = Experiment("Improved", config2)
            >>> exp3 = Experiment("Final", config3)
            >>> 
            >>> # 比较三个实验的RMSE
            >>> fig = exp1.compare_experiments([exp2, exp3], "rmse")
        
        Note:
            - 自动检测数据类型选择合适的可视化方式
            - 支持时间序列数据的趋势比较
            - 支持单值数据的柱状图比较
            - 自动生成图例和标签
            - 比较图保存在当前实验目录中
            - 处理缺失数据和异常情况
        """
        # 参数验证
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("指标名称不能为空")
        
        if not isinstance(other_experiments, list):
            raise TypeError("other_experiments必须是实验对象列表")
        
        metric_name = metric_name.strip()
        
        # 检查当前实验是否有该指标
        if metric_name not in self.results:
            logger.warning(f"当前实验没有指标: {metric_name}")
            return None
        
        try:
            # 创建比较数据
            all_data = []
            
            # 添加当前实验数据
            current_data = self.results[metric_name]
            for i, item in enumerate(current_data):
                if isinstance(item, dict):
                    all_data.append({
                        "experiment": self.name,
                        "value": item["value"],
                        "step": item.get("step", i)
                    })
                else:
                    all_data.append({
                        "experiment": self.name,
                        "value": item,
                        "step": i
                    })
            
            # 添加其他实验数据
            for exp in other_experiments:
                if not hasattr(exp, 'results') or not hasattr(exp, 'name'):
                    logger.warning(f"跳过无效的实验对象: {exp}")
                    continue
                    
                if metric_name in exp.results:
                    exp_data = exp.results[metric_name]
                    for i, item in enumerate(exp_data):
                        if isinstance(item, dict):
                            all_data.append({
                                "experiment": exp.name,
                                "value": item["value"],
                                "step": item.get("step", i)
                            })
                        else:
                            all_data.append({
                                "experiment": exp.name,
                                "value": item,
                                "step": i
                            })
                else:
                    logger.warning(f"实验 {exp.name} 没有指标 {metric_name}")
            
            if not all_data:
                logger.warning(f"没有找到可比较的 {metric_name} 数据")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(all_data)
            
            # 过滤数值型数据
            numeric_mask = pd.to_numeric(df['value'], errors='coerce').notna()
            df = df[numeric_mask]
            df['value'] = pd.to_numeric(df['value'])
            
            if df.empty:
                logger.warning(f"没有有效的数值型 {metric_name} 数据")
                return None
            
            # 绘制比较图表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 检查是否有多个步骤（时间序列数据）
            has_multiple_steps = df.groupby('experiment')['step'].nunique().max() > 1
            
            if has_multiple_steps:
                # 绘制趋势比较图
                sns.lineplot(data=df, x="step", y="value", hue="experiment", 
                           marker="o", ax=ax, linewidth=2, markersize=6)
                ax.set_xlabel('步骤', fontsize=12)
                ax.grid(True, alpha=0.3)
            else:
                # 绘制柱状比较图
                # 取每个实验的最后一个值或平均值
                summary_df = df.groupby('experiment')['value'].last().reset_index()
                bars = ax.bar(summary_df['experiment'], summary_df['value'], alpha=0.7)
                ax.set_xlabel('实验', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
                
                # 在柱状图上显示数值
                for bar, value in zip(bars, summary_df['value']):
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + max(summary_df['value'])*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f"{metric_name} 实验比较", fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12)
            
            if has_multiple_steps:
                ax.legend(title='实验', fontsize=10)
            
            plt.tight_layout()
            
            # 保存图表
            if save_comparison:
                filename = f"{metric_name}_comparison.png"
                self.save_figure(fig, filename, close_fig=False)
            
            logger.info(f"比较了 {df['experiment'].nunique()} 个实验的 {metric_name} 指标")
            return fig
            
        except Exception as e:
            logger.error(f"比较实验指标 {metric_name} 失败: {e}")
            return None
    
    @staticmethod
    def load_experiment(experiment_dir: str) -> Optional['Experiment']:
        """
        从目录加载已保存的实验
        
        从指定的实验目录中加载配置和结果，重建实验对象。
        用于分析历史实验或继续之前的实验。
        
        Args:
            experiment_dir (str): 实验目录路径
                                包含config.json和results.json的目录
        
        Returns:
            Optional[Experiment]: 重建的实验对象，如果加载失败则返回None
        
        Raises:
            ValueError: 当目录路径无效时抛出异常
            IOError: 当无法读取配置或结果文件时抛出异常
        
        Example:
            >>> # 加载历史实验
            >>> exp = Experiment.load_experiment("experiments/LightGBM_20240101_120000")
            >>> if exp:
            ...     print(f"加载实验: {exp.name}")
            ...     print(f"实验结果: {exp.results}")
        
        Note:
            - 自动检测并加载配置文件
            - 自动加载已保存的结果
            - 保持原有的实验ID和时间戳
            - 支持继续记录新的指标
            - 处理文件缺失和格式错误
        """
        # 参数验证
        if not isinstance(experiment_dir, str) or not experiment_dir.strip():
            raise ValueError("实验目录路径不能为空")
        
        experiment_dir = experiment_dir.strip()
        
        if not os.path.exists(experiment_dir):
            logger.error(f"实验目录不存在: {experiment_dir}")
            return None
        
        try:
            # 加载配置文件
            config_path = os.path.join(experiment_dir, "config.json")
            if not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                return None
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 从目录名提取实验信息
            dir_name = os.path.basename(experiment_dir)
            if '_' in dir_name:
                # 假设格式为 name_timestamp
                parts = dir_name.rsplit('_', 2)  # 分割最后两个下划线
                if len(parts) >= 3:
                    name = '_'.join(parts[:-2])
                    timestamp = f"{parts[-2]}_{parts[-1]}"
                else:
                    name = parts[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            else:
                name = dir_name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建实验对象（不创建新目录）
            exp = object.__new__(Experiment)
            exp.name = name
            exp.config = config
            exp.timestamp = timestamp
            exp.experiment_id = dir_name
            exp.base_dir = os.path.dirname(experiment_dir)
            exp.experiment_dir = experiment_dir
            exp.results = {}
            
            # 加载结果文件
            results_path = os.path.join(experiment_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    exp.results = json.load(f)
                logger.info(f"加载实验结果: {len(exp.results)} 个指标")
            else:
                logger.warning(f"结果文件不存在: {results_path}")
            
            logger.info(f"成功加载实验: {exp.experiment_id}")
            return exp
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"加载实验失败 {experiment_dir}: {e}")
            return None
        except Exception as e:
            logger.error(f"加载实验时发生未知错误 {experiment_dir}: {e}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取实验摘要信息
        
        生成包含实验基本信息、配置参数和关键指标的摘要字典。
        用于快速了解实验的整体情况和主要结果。
        
        Returns:
            Dict[str, Any]: 实验摘要字典，包含以下信息：
                          - 基本信息：名称、ID、创建时间等
                          - 配置摘要：主要参数设置
                          - 结果摘要：关键指标的最终值
                          - 文件信息：目录路径、文件列表等
        
        Example:
            >>> exp = Experiment("test", {"param1": 1})
            >>> exp.log_metric("rmse", 0.85)
            >>> summary = exp.get_summary()
            >>> print(f"实验名称: {summary['name']}")
            >>> print(f"RMSE: {summary['final_metrics']['rmse']}")
        
        Note:
            - 自动提取每个指标的最终值
            - 包含实验目录的文件统计信息
            - 计算实验运行时长（如果有时间戳）
            - 提供配置参数的简化视图
            - 支持JSON序列化输出
        """
        try:
            # 基本信息
            summary = {
                "name": self.name,
                "experiment_id": self.experiment_id,
                "timestamp": self.timestamp,
                "experiment_dir": self.experiment_dir,
                "created_time": self.timestamp
            }
            
            # 配置摘要
            summary["config_summary"] = {
                "total_params": len(self.config),
                "key_params": {k: v for k, v in list(self.config.items())[:5]}  # 前5个参数
            }
            
            # 结果摘要
            final_metrics = {}
            metric_counts = {}
            
            for metric_name, metric_data in self.results.items():
                if isinstance(metric_data, list) and metric_data:
                    # 取最后一个值
                    last_item = metric_data[-1]
                    if isinstance(last_item, dict):
                        final_metrics[metric_name] = last_item["value"]
                    else:
                        final_metrics[metric_name] = last_item
                    metric_counts[metric_name] = len(metric_data)
                elif isinstance(metric_data, (int, float, str)):
                    final_metrics[metric_name] = metric_data
                    metric_counts[metric_name] = 1
            
            summary["results_summary"] = {
                "total_metrics": len(self.results),
                "final_metrics": final_metrics,
                "metric_counts": metric_counts
            }
            
            # 文件信息
            file_info = {
                "config_exists": os.path.exists(os.path.join(self.experiment_dir, "config.json")),
                "results_exists": os.path.exists(os.path.join(self.experiment_dir, "results.json"))
            }
            
            # 统计子目录文件数量
            for subdir in ['plots', 'data', 'models']:
                subdir_path = os.path.join(self.experiment_dir, subdir)
                if os.path.exists(subdir_path):
                    file_count = len([f for f in os.listdir(subdir_path) 
                                    if os.path.isfile(os.path.join(subdir_path, f))])
                    file_info[f"{subdir}_files"] = file_count
                else:
                    file_info[f"{subdir}_files"] = 0
            
            summary["file_info"] = file_info
            
            return summary
            
        except Exception as e:
            logger.error(f"生成实验摘要失败: {e}")
            return {
                "name": getattr(self, 'name', 'Unknown'),
                "experiment_id": getattr(self, 'experiment_id', 'Unknown'),
                "error": str(e)
            }