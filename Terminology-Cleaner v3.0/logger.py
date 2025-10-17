"""
日志管理模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class TerminologyCleanerLogger:
    """术语清洗工具日志管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化日志系统"""
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger('TerminologyCleaner')
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加 handler
        if self.logger.handlers:
            return
        
        # 控制台输出（INFO 及以上级别）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(message)s'  # 控制台只显示消息
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def add_file_handler(self, log_file=None):
        """
        添加文件日志处理器
        
        参数:
            log_file: 日志文件路径，默认为 logs/cleaner_YYYYMMDD_HHMMSS.log
        """
        if log_file is None:
            # 创建 logs 目录
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            # 生成带时间戳的日志文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'cleaner_{timestamp}.log'
        
        # 文件输出（DEBUG 及以上级别）
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        return str(log_file)
    
    def set_level(self, level):
        """
        设置日志级别
        
        参数:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # 同时更新控制台 handler 的级别
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)
    
    def get_logger(self):
        """获取 logger 实例"""
        return self.logger


# 全局 logger 实例
_logger_manager = TerminologyCleanerLogger()
logger = _logger_manager.get_logger()


def setup_logging(log_file=None, level='INFO'):
    """
    配置日志系统
    
    参数:
        log_file: 日志文件路径
        level: 日志级别
    """
    if log_file:
        log_path = _logger_manager.add_file_handler(log_file)
        logger.info(f"日志文件: {log_path}")
    
    _logger_manager.set_level(level)
    return logger


def get_logger():
    """获取全局 logger 实例"""
    return logger


