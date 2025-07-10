#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志记录工具模块

提供统一的日志记录功能，支持文件和控制台输出。

核心功能：
1. 双重输出：同时支持文件和控制台日志输出
2. 自动管理：自动创建日志目录和文件
3. 格式化输出：提供详细的日志格式，包含时间、级别、函数、行号等信息
4. 级别控制：支持DEBUG、INFO、WARNING、ERROR、CRITICAL五个级别
5. 异常处理：提供异常日志记录和错误处理

日志格式：
- 文件日志：包含完整信息（时间戳、模块名、级别、函数名、行号、消息）
- 控制台日志：简化格式（时间、级别、消息）

文件命名：
- 格式：run_YYYYMMDD_HHMMSS.log
- 每次运行创建新的日志文件
- 自动按时间戳区分不同运行会话

使用方式：
1. 直接使用全局logger实例
2. 通过get_logger()获取命名logger
3. 通过setup_logging()自定义配置

技术特点：
- 支持UTF-8编码，正确处理中文日志
- 线程安全的日志记录
- 完整的错误处理和降级机制
- 支持动态调整日志级别
- 提供日志文件路径查询功能
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Union
from pathlib import Path


class Logger:
    """
    统一的日志记录器
    
    提供文件和控制台双重输出，支持不同级别的日志记录。
    自动创建日志目录和文件。
    
    Attributes:
        log_dir (str): 日志文件保存目录
        logger_name (str): 日志记录器名称
        enable_console (bool): 是否启用控制台输出
        enable_file (bool): 是否启用文件输出
        logger (logging.Logger): Python标准库日志记录器实例
        formatter (logging.Formatter): 日志格式化器
        log_file (str): 当前日志文件路径
    
    Note:
        - 支持同时输出到文件和控制台
        - 自动处理日志目录创建和权限问题
        - 提供完整的异常处理和错误恢复
        - 支持运行时动态调整日志级别
    """
    def __init__(self, 
                 log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 logger_name: str = "MovieRecommender",
                 enable_console: bool = True,
                 enable_file: bool = True):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志文件保存目录，相对或绝对路径
            log_level: 日志级别，使用logging模块的标准级别
            logger_name: 日志记录器名称，用于区分不同模块
            enable_console: 是否启用控制台输出
            enable_file: 是否启用文件输出
            
        Raises:
            ValueError: 当log_dir为空或log_level无效时
            OSError: 当无法创建日志目录时
            PermissionError: 当没有写入权限时
        """
        if not isinstance(log_dir, str) or not log_dir.strip():
            raise ValueError("日志目录路径不能为空")
        
        if log_level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
            raise ValueError(f"无效的日志级别: {log_level}")
        
        self.log_dir = log_dir
        self.logger_name = logger_name
        self.enable_console = enable_console
        self.enable_file = enable_file
        
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"无法创建日志目录 {log_dir}: {e}")
        
        self.formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if enable_file:
            self._setup_file_handler(log_level)
        
        if enable_console:
            self._setup_console_handler(log_level)
        
        self.info(f"日志系统初始化完成 - 目录: {log_dir}, 级别: {logging.getLevelName(log_level)}")
    
    def _setup_file_handler(self, log_level: int) -> None:
        """
        设置文件日志处理器
        
        创建带时间戳的日志文件，配置文件输出格式和编码。
        
        Args:
            log_level: 文件日志的级别
            
        Raises:
            PermissionError: 当没有文件写入权限时
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = os.path.join(self.log_dir, f"run_{timestamp}.log")
            
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(self.formatter)
            
            self.logger.addHandler(file_handler)
            
        except PermissionError as e:
            raise PermissionError(f"没有权限写入日志文件 {self.log_file}: {e}")
    
    def _setup_console_handler(self, log_level: int) -> None:
        """
        设置控制台日志处理器
        
        配置控制台输出格式，使用简化的日志格式以提高可读性。
        
        Args:
            log_level: 控制台日志的级别
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def info(self, message: Union[str, Exception]) -> None:
        """记录INFO级别日志"""
        self.logger.info(str(message))
    
    def warning(self, message: Union[str, Exception]) -> None:
        """记录WARNING级别日志"""
        self.logger.warning(str(message))
    
    def error(self, message: Union[str, Exception]) -> None:
        """记录ERROR级别日志"""
        self.logger.error(str(message))
    
    def debug(self, message: Union[str, Exception]) -> None:
        """记录DEBUG级别日志"""
        self.logger.debug(str(message))
    
    def critical(self, message: Union[str, Exception]) -> None:
        """记录CRITICAL级别日志"""
        self.logger.critical(str(message))
    
    def log_exception(self, message: str = "发生异常") -> None:
        """
        记录异常信息，包含完整的堆栈跟踪
        
        Args:
            message: 异常描述信息
        """
        self.logger.exception(message)
    
    def set_level(self, level: int) -> None:
        """
        动态设置日志级别
        
        Args:
            level: 新的日志级别
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
        self.info(f"日志级别已更改为: {logging.getLevelName(level)}")
    
    def get_log_file_path(self) -> Optional[str]:
        """
        获取当前日志文件路径
        
        Returns:
            日志文件的完整路径，如果未启用文件日志则返回None
        """
        return getattr(self, 'log_file', None)


try:
    logger = Logger(
        log_dir="logs",
        log_level=logging.INFO,
        logger_name="MovieRecommender"
    )
except Exception as e:
    print(f"警告: 无法创建日志记录器，使用基本输出: {e}")
    
    class BasicLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}")
        def critical(self, msg): print(f"CRITICAL: {msg}")
        def log_exception(self, msg): print(f"EXCEPTION: {msg}")
    
    logger = BasicLogger()


def get_logger(name: str = None) -> Logger:
    """
    获取日志记录器实例
    
    Args:
        name: 日志记录器名称，None时返回全局logger
        
    Returns:
        Logger实例
    """
    if name is None:
        return logger
    else:
        return Logger(logger_name=name)


def setup_logging(log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 enable_debug: bool = False) -> Logger:
    """
    设置日志系统
    
    提供便捷的日志系统初始化方法。
    
    Args:
        log_dir: 日志目录路径
        log_level: 基础日志级别
        enable_debug: 是否启用DEBUG级别（会覆盖log_level）
        
    Returns:
        配置好的Logger实例
        
    Example:
        >>> logger = setup_logging("my_logs", enable_debug=True)
        >>> logger.info("日志系统已启动")
    """
    if enable_debug:
        log_level = logging.DEBUG
    
    return Logger(
        log_dir=log_dir,
        log_level=log_level,
        logger_name="MovieRecommender"
    )