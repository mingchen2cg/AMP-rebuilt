import argparse
import logging
import os
import sys
import datetime
import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback  # 引入 Callback
)
# 引入 transformers 的日志工具以便接管
import transformers.utils.logging as hf_logging 
from rich.logging import RichHandler

# 引入本地模块 (确保 utils 和 models 文件夹在当前目录下)
from utils.T2structLoader import load_T2Struc_and_tokenizers
from utils.Prostt5Loader import load_prostt5_model
from models.FusionModel import ProstT5WithGatedFusion

# =================================================================
# 0. 日志与输出重定向工具 (新增)
# =================================================================
class StreamToLogger(object):
    """
    将 sys.stdout 和 sys.stderr 重定向到 logger，
    同时保留在控制台的输出。
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            # 记录到日志文件
            self.logger.log(self.log_level, line.rstrip())
        # 同时输出到原来的控制台 (RichHandler 会处理 formatting，这里直接 print 可能会乱，
        # 但为了保留原始流行为，通常交给 handler 处理。
        # 由于我们用了 RichHandler 作为 root handler，这里不需要再次 print 到 stdout，
        # 否则会出现双重打印。但为了兼容进度条，通常不做完全拦截，
        # 简单方案是只依赖 Logging 的 FileHandler)
        pass 

    def flush(self):
        pass

def setup_logging(output_dir):
    # 1. 创建 Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 2. 格式设置
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 3. File Handler (写入文件)
    log_file = os.path.join(output_dir, "train.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # 4. Rich Handler (控制台输出)
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    
    # 5. 设置 Transformers 的日志
    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    # 将 transformers 的日志也导向我们的 file handler
    hf_logger = hf_logging.get_logger("transformers")
    hf_logger.addHandler(file_handler)

    return logger

# 全局 logger 占位，稍后在 main 中初始化
logger = logging.getLogger("root")

# =================================================================
# 1. 自定义 DataCollator
# =================================================================
class FusionDataCollator:
    def __init__(self, text_tokenizer, prostt5_tokenizer, max_text_len=128, max_struc_len=512):
        self.text_tokenizer = text_tokenizer
        self.prostt5_tokenizer = prostt5_tokenizer
        self.max_text_len = max_text_len
        self.max_struc_len = max_struc_len

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        structures = [item['structure'] for item in batch]
        
        text_encodings = self.text_tokenizer(
            texts, padding="longest", truncation=True, max_length=self.max_text_len, return_tensors="pt"
        )

        struc_encodings = self.prostt5_tokenizer(
            structures, padding="longest", truncation=True, max_length=self.max_struc_len, return_tensors="pt"
        )
        
        aa_sequences = [item['aa_sequence'] for item in batch]
        formatted_targets = [" ".join(list(seq)) for seq in aa_sequences]
        
        labels_encodings = self.prostt5_tokenizer(
            formatted_targets, padding="longest", truncation=True, max_length=self.max_struc_len, return_tensors="pt"
        )
        
        labels = labels_encodings.input_ids
        labels[labels == self.prostt5_tokenizer.pad_token_id] = -100

        return {
            "text_input_ids": text_encodings.input_ids,
            "text_attention_mask": text_encodings.attention_mask,
            "structure_input_ids": struc_encodings.input_ids,
            "structure_attention_mask": struc_encodings.attention_mask,
            "labels": labels
        }

# =================================================================
# 2. 自定义 Callback (新增：每个 Epoch 保存)
# =================================================================
class SaveModelAtEpochEndCallback(TrainerCallback):
    """
    在每个 Epoch 结束时，强制保存模型权重和 Tokenizer 到指定目录
    """
    def __init__(self, output_dir, text_tokenizer, prost_tokenizer):
        self.output_dir = output_dir
        self.text_tokenizer = text_tokenizer
        self.prost_tokenizer = prost_tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        # 当前 Epoch 数 (state.epoch 是 float，转为 int)
        current_epoch = int(state.epoch)
        save_path = os.path.join(self.output_dir, f"epoch_{current_epoch}_model")
        
        os.makedirs(save_path, exist_ok=True)
        
        logger.info(f"=== Epoch {current_epoch} 结束，正在保存模型至 {save_path} ===")
        
        # 获取模型对象 (处理可能的 DataParallel 包装)
        model = kwargs['model']
        if hasattr(model, 'module'):
            model = model.module
            
        # 1. 保存模型权重 (bin 格式)
        torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        
        # 2. 保存 Tokenizers
        self.text_tokenizer.save_pretrained(os.path.join(save_path, "text_tokenizer"))
        self.prost_tokenizer.save_pretrained(os.path.join(save_path, "prostt5_tokenizer"))
        
        logger.info(f"=== 模型 Epoch {current_epoch} 保存完毕 ===")

# =================================================================
# 3. 模型初始化函数
# =================================================================
def initialize_fusion_model(t2struc_path, prostt5_path, device):
    logger.info("正在初始化融合模型...")
    
    logger.info(f"加载 T2Struc: {t2struc_path}")
    t2struc_full, text_tokenizer, _ = load_T2Struc_and_tokenizers(weights_path=t2struc_path)
    text_encoder = t2struc_full.lm
    
    try:
        actual_text_dim = text_encoder.get_input_embeddings().weight.shape[1]
        logger.info(f"检测到 Text Encoder 真实维度: {actual_text_dim}")
    except:
        logger.warning("无法检测真实维度，根据报错日志强制设定为 1024")
        actual_text_dim = 1024

    logger.info(f"加载 ProstT5: {prostt5_path}")
    prostt5_model, prostt5_tokenizer = load_prostt5_model(
        device="cpu", weights_path=prostt5_path, torch_dtype=torch.bfloat16
    )
    
    fusion_model = ProstT5WithGatedFusion(
        t2struc_text_encoder=text_encoder,
        prostt5_model=prostt5_model,
        hidden_dim=prostt5_model.config.hidden_size,
        text_dim=actual_text_dim 
    )
    
    fusion_model.to(device)
    
    return fusion_model, text_tokenizer, prostt5_tokenizer

# =================================================================
# 4. 主训练流程
# =================================================================
def main():
    parser = argparse.ArgumentParser(description="自适应门控融合模型训练脚本")
    parser.add_argument("--data_path", type=str, default='./data/Augmentationx10.jsonl', help="训练数据路径")
    parser.add_argument("--t2struc_path", type=str, required=True, help="T2Struc 权重路径")
    parser.add_argument("--prostt5_path", type=str, default='./weights/ProstT5', help="ProstT5 权重路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # --- 设置输出目录 ---
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"./finetuned_fusion/{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 启动日志系统 (核心修改) ---
    global logger
    logger = setup_logging(args.output_dir)
    
    logger.info("=== 启动训练任务 ===")
    logger.info(f"输出目录: {args.output_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 初始化模型
    model, text_tok, prost_tok = initialize_fusion_model(
        args.t2struc_path, args.prostt5_path, device
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"总参数量: {all_params/1e6:.2f} M")
    logger.info(f"可训练参数: {trainable_params/1e6:.2f} M (占比 {trainable_params/all_params:.1%})")
    
    # 2. 准备数据
    logger.info(f"加载数据集: {args.data_path}")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    
    collator = FusionDataCollator(
        text_tokenizer=text_tok, 
        prostt5_tokenizer=prost_tok,
        max_text_len=128,
        max_struc_len=512
    )
    
    # 3. 配置 Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        
        # 这里的 save_strategy 设为 epoch，Trainer 会自动保存 checkpoint-xxx
        # 但我们主要依赖自定义 Callback 来保存干净的权重
        save_strategy="epoch", 
        evaluation_strategy="epoch",
        
        bf16=True, 
        dataloader_num_workers=4,
        remove_unused_columns=False, 
        label_names=["labels"],
        save_safetensors=False,
        
        # 禁用 Trainer 自带的进度条控制台输出，
        # 因为我们想把所有东西都重定向到 log (可选，若不需要可注释掉)
        disable_tqdm=False 
    )
    
    # 4. 初始化 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        tokenizer=prost_tok,
        # 添加自定义 Callback
        callbacks=[SaveModelAtEpochEndCallback(args.output_dir, text_tok, prost_tok)]
    )
    
    # 5. 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 6. 保存最终模型 (作为一个 fallback)
    logger.info("正在保存最终模型 (Final)...")
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True) 
    
    torch.save(model.state_dict(), os.path.join(final_path, "pytorch_model.bin"))
    text_tok.save_pretrained(os.path.join(final_path, "text_tokenizer"))
    prost_tok.save_pretrained(os.path.join(final_path, "prostt5_tokenizer"))
    
    logger.info(f"全部训练完成！最终模型已保存至 {final_path}")

if __name__ == "__main__":
    main()