import json
import torch
import argparse
import datetime
import os
import sys
import logging
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
import transformers

# 导入新的 Loader
from utils.Prostt5Loader import load_prostt5_model

# 常量定义 (默认值)
MAX_LENGTH = 64

def get_args():
    parser = argparse.ArgumentParser(description="ProstT5 Fine-tuning Script")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    # 添加路径参数
    parser.add_argument("--model_path", type=str, default='./weights/ProstT5', help="Path to ProstT5 weights")
    parser.add_argument("--data_path", type=str, default='./data/Augmentationx10.jsonl', help="Path to training data (jsonl)")
    
    return parser.parse_args()

def setup_logging(output_dir):
    """配置日志：同时输出到控制台和文件"""
    log_file_path = os.path.join(output_dir, "training.log")
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    transformers_logger = transformers.utils.logging.get_logger("transformers")
    transformers_logger.addHandler(logging.FileHandler(log_file_path))
    
    return logger

def preprocess_function(examples, tokenizer):
    inputs = examples["structure"]
    targets = examples["aa_sequence"]

    formatted_targets = [" ".join(list(seq)) for seq in targets]

    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        truncation=True
    )

    labels = tokenizer(
        formatted_targets, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        truncation=True
    )

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    args = get_args()
    
    # 1. 确定并创建输出目录
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"./finetuned_prostt5/{timestamp}"
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. 初始化日志系统
    logger = setup_logging(args.output_dir)
    
    logger.info("="*40)
    logger.info(f"训练任务启动")
    logger.info(f"  - Batch Size:    {args.batch_size}")
    logger.info(f"  - Learning Rate: {args.learning_rate}")
    logger.info(f"  - Epochs:        {args.epochs}")
    logger.info(f"  - Output Dir:    {args.output_dir}")
    logger.info(f"  - Model Path:    {args.model_path}")
    logger.info(f"  - Data Path:     {args.data_path}")
    logger.info("="*40)

    # 3. 加载模型 (使用 utils.Prostt5Loader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"正在使用设备: {device} 加载模型...")
    
    # Critical Fix: 显式指定 bfloat16，避免 Loader 自动转为 half (fp16) 导致 NaN
    model, tokenizer = load_prostt5_model(
        device, 
        weights_path=args.model_path, 
        torch_dtype=torch.bfloat16
    )
    
    if model is None or tokenizer is None:
        logger.error("模型加载失败，程序退出。")
        return

    # 4. 准备数据
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(eval_dataset)}")
    
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names
    )
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True, 
        num_proc=4,
        remove_columns=eval_dataset.column_names
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model
    )

    # 5. 配置 Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        predict_with_generate=True,
        
        bf16=True,             
        fp16=False,            
        tf32=True,             
        
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        
        logging_dir=os.path.join(args.output_dir, 'runs'),
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("开始微调训练...")
    trainer.train()

    logger.info(f"训练完成，正在保存模型到 {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("所有工作已完成。")

if __name__ == "__main__":
    main()