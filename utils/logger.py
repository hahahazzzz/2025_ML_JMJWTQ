#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志记录工具模块

该模块提供了电影推荐系统的统一日志记录功能，主要特点：
1. 支持同时输出到文件和控制台
2. 自动创建带时间戳的日志文件
3. 提供多种日志级别（DEBUG, INFO, WARNING, ERROR）
4. 统一的日志格式和配置
5. 线程安全的日志记录

使用方式：
    from utils.logger import logger
    logger.info("这是一条信息日志")
    logger.error("这是一条错误日志")

日志文件命名规则：
    run_YYYYMMDD_HHMMSS.log
    例如：run_20241201_143052.log

作者: 电影推荐系统开发团队
创建时间: 2024
最后修改: 2024
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Union
from pathlib import Path


class Logger:
    """
    电影推荐系统专用日志记录器
    
    该类封装了Python标准库的logging模块，提供了更便捷的日志记录接口。
    支持同时向文件和控制台输出日志，并自动管理日志文件的创建和格式化。
    
    主要功能：
    - 自动创建日志目录和文件
    - 支持多种日志级别
    - 统一的日志格式
    - 线程安全
    - 支持日志轮转（可扩展）
    
    Attributes:
        logger (logging.Logger): Python标准日志记录器实例
        log_dir (str): 日志文件保存目录
        log_file (str): 当前日志文件路径
    
    Example:
        >>> logger = Logger(log_dir="logs", log_level=logging.INFO)
        >>> logger.info("系统启动")
        >>> logger.error("发生错误")
    """
    
    def __init__(self, 
                 log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 logger_name: str = "MovieRecommender",
                 enable_console: bool = True,
                 enable_file: bool = True):
        """
        初始化日志记录器
        
        创建一个配置完整的日志记录器，包括文件处理器和控制台处理器。
        日志文件会自动以当前时间戳命名，避免文件冲突。
        
        Args:
            log_dir (str, optional): 日志文件保存目录，默认为"logs"
            log_level (int, optional): 日志级别，默认为logging.INFO
                                     可选值：logging.DEBUG, logging.INFO, 
                                            logging.WARNING, logging.ERROR, logging.CRITICAL
            logger_name (str, optional): 日志记录器名称，默认为"MovieRecommender"
            enable_console (bool, optional): 是否启用控制台输出，默认为True
            enable_file (bool, optional): 是否启用文件输出，默认为True
        
        Raises:
            OSError: 当无法创建日志目录时抛出异常
            PermissionError: 当没有写入权限时抛出异常
        
        Example:
            >>> # 基本用法
            >>> logger = Logger()
            >>> 
            >>> # 自定义配置
            >>> logger = Logger(
            ...     log_dir="custom_logs",
            ...     log_level=logging.DEBUG,
            ...     logger_name="MyApp"
            ... )
        """
        # 参数验证
        if not isinstance(log_dir, str) or not log_dir.strip():
            raise ValueError("日志目录路径不能为空")
        
        if log_level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
            raise ValueError(f"无效的日志级别: {log_level}")
        
        # 初始化基本属性
        self.log_dir = log_dir
        self.logger_name = logger_name
        self.enable_console = enable_console
        self.enable_file = enable_file
        
        # 创建日志记录器
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        
        # 避免重复添加处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建日志目录
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"无法创建日志目录 {log_dir}: {e}")
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建文件处理器
        if enable_file:
            self._setup_file_handler(log_level)
        
        # 创建控制台处理器
        if enable_console:
            self._setup_console_handler(log_level)
        
        # 记录初始化信息
        self.info(f"日志系统初始化完成 - 目录: {log_dir}, 级别: {logging.getLevelName(log_level)}")
    
    def _setup_file_handler(self, log_level: int) -> None:
        """
        设置文件处理器
        
        创建一个文件处理器，用于将日志写入文件。文件名包含时间戳以避免冲突。
        
        Args:
            log_level (int): 日志级别
        
        Raises:
            PermissionError: 当没有文件写入权限时抛出异常
        """
        try:
            # 生成带时间戳的日志文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = os.path.join(self.log_dir, f"run_{timestamp}.log")
            
            # 创建文件处理器
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(self.formatter)
            
            # 添加到日志记录器
            self.logger.addHandler(file_handler)
            
        except PermissionError as e:
            raise PermissionError(f"没有权限写入日志文件 {self.log_file}: {e}")
    
    def _setup_console_handler(self, log_level: int) -> None:
        """
        设置控制台处理器
        
        创建一个控制台处理器，用于将日志输出到标准输出。
        
        Args:
            log_level (int): 日志级别
        """
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # 为控制台设置简化的格式
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # 添加到日志记录器
        self.logger.addHandler(console_handler)
    
    def info(self, message: Union[str, Exception]) -> None:
        """
        记录信息级别的日志
        
        用于记录一般性的信息，如程序运行状态、重要操作的完成等。
        
        Args:
            message (Union[str, Exception]): 日志消息，可以是字符串或异常对象
        
        Example:
            >>> logger.info("数据加载完成")
            >>> logger.info(f"处理了{count}条记录")
        """
        self.logger.info(str(message))
    
    def warning(self, message: Union[str, Exception]) -> None:
        """
        记录警告级别的日志
        
        用于记录可能的问题或需要注意的情况，但不会影响程序正常运行。
        
        Args:
            message (Union[str, Exception]): 日志消息，可以是字符串或异常对象
        
        Example:
            >>> logger.warning("配置文件不存在，使用默认配置")
            >>> logger.warning("检测到异常值，已自动处理")
        """
        self.logger.warning(str(message))
    
    def error(self, message: Union[str, Exception]) -> None:
        """
        记录错误级别的日志
        
        用于记录错误信息，通常是程序运行中遇到的问题或异常。
        
        Args:
            message (Union[str, Exception]): 日志消息，可以是字符串或异常对象
        
        Example:
            >>> logger.error("文件读取失败")
            >>> try:
            ...     risky_operation()
            ... except Exception as e:
            ...     logger.error(f"操作失败: {e}")
        """
        self.logger.error(str(message))
    
    def debug(self, message: Union[str, Exception]) -> None:
        """
        记录调试级别的日志
        
        用于记录详细的调试信息，通常只在开发和调试阶段使用。
        只有当日志级别设置为DEBUG时，这些消息才会被输出。
        
        Args:
            message (Union[str, Exception]): 日志消息，可以是字符串或异常对象
        
        Example:
            >>> logger.debug("进入函数 process_data")
            >>> logger.debug(f"变量值: x={x}, y={y}")
        """
        self.logger.debug(str(message))
    
    def critical(self, message: Union[str, Exception]) -> None:
        """
        记录严重错误级别的日志
        
        用于记录严重的错误，通常是导致程序无法继续运行的问题。
        
        Args:
            message (Union[str, Exception]): 日志消息，可以是字符串或异常对象
        
        Example:
            >>> logger.critical("系统内存不足，程序即将退出")
            >>> logger.critical("数据库连接失败，无法继续")
        """
        self.logger.critical(str(message))
    
    def log_exception(self, message: str = "发生异常") -> None:
        """
        记录异常信息（包含完整的堆栈跟踪）
        
        该方法会自动捕获当前的异常信息并记录完整的堆栈跟踪，
        通常在except块中使用。
        
        Args:
            message (str, optional): 异常描述信息，默认为"发生异常"
        
        Example:
            >>> try:
            ...     risky_operation()
            ... except Exception:
            ...     logger.log_exception("执行危险操作时发生异常")
        """
        self.logger.exception(message)
    
    def set_level(self, level: int) -> None:
        """
        动态设置日志级别
        
        Args:
            level (int): 新的日志级别
        
        Example:
            >>> logger.set_level(logging.DEBUG)  # 启用调试模式
            >>> logger.set_level(logging.WARNING)  # 只显示警告和错误
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
        self.info(f"日志级别已更改为: {logging.getLevelName(level)}")
    
    def get_log_file_path(self) -> Optional[str]:
        """
        获取当前日志文件的完整路径
        
        Returns:
            Optional[str]: 日志文件路径，如果未启用文件输出则返回None
        
        Example:
            >>> path = logger.get_log_file_path()
            >>> print(f"日志文件位置: {path}")
        """
        return getattr(self, 'log_file', None)


# ==================== 全局日志记录器实例 ====================
# 创建默认的日志记录器实例，供整个项目使用
# 这样可以确保整个项目使用统一的日志配置
try:
    logger = Logger(
        log_dir="logs",
        log_level=logging.INFO,
        logger_name="MovieRecommender"
    )
except Exception as e:
    # 如果创建日志记录器失败，使用基本的控制台输出
    print(f"警告: 无法创建日志记录器，使用基本输出: {e}")
    
    class BasicLogger:
        """基本日志记录器，当主日志记录器创建失败时使用"""
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}")
        def critical(self, msg): print(f"CRITICAL: {msg}")
        def log_exception(self, msg): print(f"EXCEPTION: {msg}")
    
    logger = BasicLogger()


# ==================== 便捷函数 ====================
def get_logger(name: str = None) -> Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name (str, optional): 日志记录器名称，默认使用全局实例
    
    Returns:
        Logger: 日志记录器实例
    
    Example:
        >>> custom_logger = get_logger("CustomModule")
        >>> custom_logger.info("自定义模块日志")
    """
    if name is None:
        return logger
    else:
        return Logger(logger_name=name)


def setup_logging(log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 enable_debug: bool = False) -> Logger:
    """
    快速设置日志系统
    
    Args:
        log_dir (str, optional): 日志目录，默认为"logs"
        log_level (int, optional): 日志级别，默认为logging.INFO
        enable_debug (bool, optional): 是否启用调试模式，默认为False
    
    Returns:
        Logger: 配置好的日志记录器
    
    Example:
        >>> logger = setup_logging(log_dir="my_logs", enable_debug=True)
        >>> logger.info("日志系统已配置")
    """
    if enable_debug:
        log_level = logging.DEBUG
    
    return Logger(
        log_dir=log_dir,
        log_level=log_level,
        logger_name="MovieRecommender"
    )