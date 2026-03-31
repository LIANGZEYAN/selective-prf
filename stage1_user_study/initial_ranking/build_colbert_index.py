#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ColBERT 索引构建脚本
用于为 MSMARCO Passage v1 数据集构建 ColBERT 索引
"""

import os
import sys
import logging
from datetime import datetime

# ============ 第一步：设置环境变量（必须在最开始）============
os.environ['IR_DATASETS_HOME'] = '../../Trec-llm/dataset'
os.environ['HF_HOME'] = '../../Trec-llm/huggingface_cache'

# ============ 第二步：导入基础库 ============
import random
import numpy as np
import torch
import pandas as pd
import pyterrier as pt
from pyterrier.measures import *
from collections import defaultdict
import ir_datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import faiss

# ============ 第三步：初始化 PyTerrier（必须在导入 pyterrier_colbert 之前）============
if not pt.started():
    pt.init()

# ============ 第四步：现在可以导入 pyterrier_colbert ============
import pyterrier_colbert
from pyterrier_colbert.indexing import ColBERTIndexer

# ============ 配置日志 ============
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"colbert_indexing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=" * 50)
        logger.info("ColBERT 索引构建开始")
        logger.info("=" * 50)
        
        # ============ 环境信息 ============
        logger.info(f"IR_DATASETS_HOME: {os.environ['IR_DATASETS_HOME']}")
        logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
        logger.info("PyTerrier 已初始化")
        
        # ============ 检查 GPU ============
        gpu_count = faiss.get_num_gpus()
        logger.info(f"检测到 {gpu_count} 个 FAISS GPU")
        assert gpu_count > 0, "需要至少一个 GPU"
        
        logger.info(f"PyTorch CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA 设备数量: {torch.cuda.device_count()}")
            logger.info(f"当前 CUDA 设备: {torch.cuda.current_device()}")
            logger.info(f"设备名称: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("警告: PyTorch CUDA 不可用")
        
        # ============ 加载数据集 ============
        logger.info("加载 msmarco-passage 数据集...")
        msmarco_dataset = pt.get_dataset('irds:msmarco-passage/dev/small')
        logger.info("数据集加载完成")
        
        # ============ 设置索引路径 ============
        index_path = "../indices"
        os.makedirs(index_path, exist_ok=True)
        index_full_path = os.path.join(index_path, "msmarco-v1-colbert")
        logger.info(f"索引将保存到: {os.path.abspath(index_full_path)}")
        
        # ============ 创建索引器 ============
        logger.info("初始化 ColBERT 索引器...")
        indexer = ColBERTIndexer(
            checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip",
            index_root=index_path,
            index_name="msmarco-v1-colbert",
            chunksize=32,
            ids=True
        )
        logger.info("索引器初始化完成")
        
        # ============ 构建索引 ============
        logger.info("=" * 50)
        logger.info("开始构建索引...")
        logger.info("预计需要数小时，请耐心等待...")
        logger.info("=" * 50)
        
        start_time = datetime.now()
        logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        indexer.index(msmarco_dataset.get_corpus_iter())
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 50)
        logger.info("索引构建完成！")
        logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"总耗时: {duration}")
        logger.info(f"索引保存在: {os.path.abspath(index_full_path)}")
        logger.info("=" * 50)
        
        # ============ 验证索引 ============
        logger.info("验证索引文件...")
        if os.path.exists(index_full_path):
            files = os.listdir(index_full_path)
            logger.info(f"索引包含 {len(files)} 个文件")
            
            # 显示文件大小
            total_size = 0
            for f in files[:10]:  # 只显示前10个
                file_path = os.path.join(index_full_path, f)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    total_size += size
                    logger.info(f"  - {f}: {size / (1024**2):.2f} MB")
            
            if len(files) > 10:
                logger.info(f"... 还有 {len(files) - 10} 个文件")
            
            # 计算所有文件总大小
            total_size_all = sum(
                os.path.getsize(os.path.join(index_full_path, f))
                for f in files if os.path.isfile(os.path.join(index_full_path, f))
            )
            logger.info(f"索引总大小: {total_size_all / (1024**3):.2f} GB")
        else:
            logger.error(f"索引目录不存在: {index_full_path}")
            return 1
        
        logger.info("=" * 50)
        logger.info("✓ 索引构建成功完成！")
        logger.info("=" * 50)
        return 0
        
    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"✗ 发生错误: {str(e)}")
        logger.error("=" * 50)
        logger.error("详细错误信息:", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)