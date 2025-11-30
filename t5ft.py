import json
import torch
import argparse
import datetime
import os
import sys
import logging
import matplotlib.pyplot as plt
import pandas as pd  # 新增 pandas 用于处理数据

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
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
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

# === 新增函数：保存 Loss 数据和绘图 ===
def save_training_metrics(trainer, output_dir, logger):
    """
    从 Trainer 的日志历史中提取 Loss 数据，保存 CSV 并绘图
    """
    # 1. 提取日志历史
    log_history = trainer.state.log_history
    
    # 过滤出包含 'loss' 的记录 (训练 loss)
    data = []
    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            data.append({
                'step': entry['step'],
                'loss': entry['loss'],
                'epoch': entry.get('epoch', 0)
            })
    
    if not data:
        logger.warning("未在 log_history 中找到训练 Loss 数据，跳过绘图。")
        return

    df = pd.DataFrame(data)
    
    # 2. 保存 CSV
    csv_path = os.path.join(output_dir, "loss_log.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Loss 日志已保存到: {csv_path}")

    # 3. 绘制曲线 (参考 t2struct-ft.py 风格)
    try:
        steps = df['step'].tolist()
        losses = df['loss'].tolist()
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label='Training Loss', alpha=0.7)
        
        # 如果点够多，画平滑曲线
        if len(losses) > 20: 
            window_size = min(10, len(losses)//5)
            moving_avg = df['loss'].rolling(window=window_size).mean()
            plt.plot(steps, moving_avg, label=f'Moving Avg ({window_size})', color='red')
            
        plt.title(f"Training Loss Curve (ProstT5)")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, "training_loss_curve.png")
        plt.savefig(plot_path, dpi=300)
        plt.close() # 关闭画布释放内存
        logger.info(f"Loss 曲线图已保存到: {plot_path}")
        
    except Exception as e:
        logger.error(f"绘图失败: {e}")
# =========================================

def main():
    args = get_args()
    
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"./finetuned_prostt5/{timestamp}"
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = setup_logging(args.output_dir)
    
    logger.info("="*40)
    logger.info(f"训练任务启动")
    logger.info(f"  - Batch Size:    {args.batch_size}")
    logger.info(f"  - Output Dir:    {args.output_dir}")
    logger.info(f"  - Model Path:    {args.model_path}")
    logger.info("="*40)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"正在使用设备: {device} 加载模型...")
    
    model, tokenizer = load_prostt5_model(
        device, 
        weights_path=args.model_path, 
        torch_dtype=torch.bfloat16
    )
    
    if model is None or tokenizer is None:
        logger.error("模型加载失败，程序退出。")
        return

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

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
        logging_steps=50, # 每50步记录一次 Loss，用于画图
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

    # === 调用保存 Metrics 的逻辑 ===
    logger.info("正在处理训练日志和绘图...")
    save_training_metrics(trainer, args.output_dir, logger)
    # ===============================

    logger.info(f"训练完成，正在保存模型到 {args.output_dir} ...")
    
    if trainer.is_world_process_zero():
        model_to_save = trainer.unwrap_model(trainer.model)
        model_to_save.save_pretrained(
            args.output_dir, 
            safe_serialization=False, # 保持生成 pytorch_model.bin
            max_shard_size="100GB"    # 强制不分片
        )
        tokenizer.save_pretrained(args.output_dir)
        
    logger.info("模型已保存为单文件 pytorch_model.bin，Loader 可直接加载。")
    logger.info("所有工作已完成。")

if __name__ == "__main__":
    main()