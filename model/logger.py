import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from torch.utils.tensorboard import SummaryWriter

class Logger: 
    def __init__(self, name="F5TTS", log_dir="./logs", max_log_size=100*1024*1024, backup_count=5):
        """
        初始化日志记录器
        Args:
            name: 日志记录器名称
            log_dir: 日志文件保存目录
            max_log_size: 单个日志文件的最大大小，超过该大小后滚动（单位字节），默认100MB
            backup_count: 保留的旧日志文件数
        """
        # 创建日志目录
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 创建tensorboard目录
        self.tb_dir = os.path.join(log_dir, 'tensorboard')
        if not os.path.exists(self.tb_dir):
            os.makedirs(self.tb_dir)
            
        # 初始化tensorboard writer
        self.writer = SummaryWriter(log_dir=self.tb_dir)
            
        # 生成日志文件名,包含时间戳
        log_file = os.path.join(log_dir, f'{name}.log')
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # 创建滚动日志文件处理器
        file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def debug(self, message):
        """记录调试级别的日志"""
        self.logger.debug(message)
        
    def add_scalar(self, tag, scalar_value, global_step=None):
        """记录标量值到tensorboard"""
        self.writer.add_scalar(tag, scalar_value, global_step)

    def info(self, message):
        """记录信息级别的日志"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告级别的日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误级别的日志"""
        self.logger.error(message)

    def critical(self, message):
        """记录严重错误级别的日志"""
        self.logger.critical(message)

    def exception(self, message):
        """记录异常信息"""
        self.logger.exception(message)
