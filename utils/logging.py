import os
import sys
import logging
import time # 增加导入

# 为 name 和 log_filename 设置默认值
def get_logger(log_dir, name='GEN-CMuST', log_filename=None, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    
    # 如果没传文件名，默认使用时间戳
    if log_filename is None:
        log_filename = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"

    logger = logging.getLogger(name)
    # 避免重复添加 Handler 导致日志重复打印
    if not logger.handlers:
        logger.setLevel(level)

        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(file_formatter)

        console_formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    print('Log directory:', log_dir)
    return logger