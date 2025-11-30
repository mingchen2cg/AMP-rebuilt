import argparse
import datetime
import json
import logging
import os

# --- 第三方库 ---
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # 使用 PyTorch 原生的 AdamW，替代 transformers.AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from rich.logging import RichHandler

# --- 本地模块 ---
# 确保你的 utils 目录下有对应的更新后的 loader
from utils.T2structLoader import load_T2Struc_and_tokenizers

# 全局 logger
logger = logging.getLogger("rich")

# --- 1. 自定义数据集类 ---
class CustomProteinDataset(Dataset):
    """Dataset for loading text-structure pairs."""
    def __init__(self, data_path):
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.jsonl'):
            self.data = pd.read_json(data_path, lines=True)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .jsonl")
        
        if 'text' not in self.data.columns or 'structure' not in self.data.columns:
            raise ValueError("Data file must contain 'text' and 'structure' columns.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            "text": item['text'],
            "structure": item['structure']
        }

# --- 2. Data Collator ---
class DataCollator:
    def __init__(self, text_tokenizer, structure_tokenizer):
        self.text_tokenizer = text_tokenizer
        self.structure_tokenizer = structure_tokenizer

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        structures = [item['structure'] for item in batch]

        text_encodings = self.text_tokenizer(
            texts, return_tensors="pt", max_length=768, truncation=True, padding="longest",
        )
        structure_encodings = self.structure_tokenizer(
            structures, return_tensors="pt", max_length=1024, truncation=True, padding="longest",
        )

        return {
            "text_ids": text_encodings.input_ids,
            "text_masks": text_encodings.attention_mask,
            "structure_token_ids": structure_encodings.input_ids,
            "structure_token_masks": structure_encodings.attention_mask,
        }

def setup_logging(save_dir):
    log_file_path = os.path.join(save_dir, "training.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(file_formatter)

    console_handler = RichHandler(show_path=False, show_time=True, omit_repeated_times=False)

    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]",
        handlers=[console_handler, file_handler], force=True 
    )
    return logging.getLogger("rich")

def train(args, run_output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("--- Experiment Configuration ---")
    logger.info(f"Device: {device}")
    logger.info(f"Output Directory: {run_output_dir}")
    logger.info(f"Weights Path: {args.weights_path}")
    logger.info("--------------------------------")

    logger.info("Loading model and tokenizers...")
    
    # 使用更新后的 loader，传入 weights_path
    model, text_tokenizer, structure_tokenizer = load_T2Struc_and_tokenizers(
        weights_path=args.weights_path
    )
    
    model.to(device)
    model.train()
    logger.info("Model loaded successfully.")

    # 准备数据
    dataset = CustomProteinDataset(args.data_path)
    data_collator = DataCollator(text_tokenizer, structure_tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=data_collator,
        shuffle=True, num_workers=4, pin_memory=True
    )

    # 优化器 (使用 torch.optim.AdamW)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.1
    )

    total_training_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )

    logger.info(f"Starting training loop. Dataset size: {len(dataset)}")
    
    all_losses = []
    global_step = 0
    loss_csv_path = os.path.join(run_output_dir, "loss_log.csv")

    for epoch in range(args.epochs):
        logger.info(f"--- Starting Epoch {epoch + 1}/{args.epochs} ---")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        epoch_losses = []
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(batch)
            loss = outputs['outputs'].loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            all_losses.append(loss_val)
            epoch_losses.append(loss_val)
            global_step += 1

            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.2e}"})

            if global_step % args.logging_steps == 0:
                logger.info(f"Epoch: {epoch + 1} | Step: {global_step}/{total_training_steps} | Lr: {current_lr:.2e} | Loss: {loss_val:.4f}")
                pd.DataFrame({"step": list(range(len(all_losses))), "loss": all_losses}).to_csv(loss_csv_path, index=False)

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}")
        pd.DataFrame({"step": list(range(len(all_losses))), "loss": all_losses}).to_csv(loss_csv_path, index=False)

    logger.info("Training finished. Saving final model...")
    final_model_dir = os.path.join(run_output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(final_model_dir, "pytorch_model.bin"))
    logger.info(f"Model weights saved to: {final_model_dir}")

    # 绘图
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(all_losses)), all_losses, label='Training Loss', alpha=0.7)
        if len(all_losses) > 100:
            window_size = 50
            moving_avg = pd.Series(all_losses).rolling(window=window_size).mean()
            plt.plot(moving_avg, label=f'Moving Avg ({window_size})', color='red')
            
        plt.title(f"Training Loss Curve - {args.model_name}")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_output_dir, "training_loss_curve.png"), dpi=300)
    except Exception as e:
        logger.error(f"Failed to plot loss curve: {e}")
    
    with open(os.path.join(run_output_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune T2struc model.")
    
    # 路径参数
    parser.add_argument("--data_path", type=str, default='./data/Augmentationx10.jsonl', help="Path to the training data file.")
    # 默认权重路径
    parser.add_argument("--weights_path", type=str, 
                        default="/t9k/mnt/AMP-rebuilt/weights/Pinal/T2struc-1.2B",
                        help="Path to directory containing config.yaml and pytorch_model.bin")
    
    parser.add_argument("--output_dir", type=str, default="./finetuned_t2struct", help="Base directory for saving outputs.")
    
    # 训练参数
    parser.add_argument("--model_name", type=str, default="T2struc-1.2B", help="Used for logging purposes only.")
    parser.add_argument("--epochs", type=int, default=4, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log metrics and save CSV every X steps.")

    args = parser.parse_args()

    # 创建输出目录
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(args.output_dir, start_time)
    os.makedirs(run_output_dir, exist_ok=True)

    setup_logging(run_output_dir)
    train(args, run_output_dir)