import torch
import logging
import json
import os
import argparse
from typing import List, Dict, Any

# 相对引用 utils 包内部的模块
from utils.T2structLoader import (
    load_T2Struc_and_tokenizers, 
    T2StrucPrepareGenerationInputs, 
    T2StrucGeneration, 
    T2StrcuDefaultGenerationConfig
)
from utils.Prostt5Loader import load_prostt5_model, GEN_KWARGS_FOLD2AA
from utils.seq_processing import clean_sequence

# 配置日志
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]")
logger = logging.getLogger("rich")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _get_best_3di_candidate(
    prompt_text: str,
    t2struc: torch.nn.Module,
    text_tokenizer: Any,
    structure_tokenizer: Any
) -> str:
    """
    [内部工具函数] 执行一次 T2Struc 推理，并返回概率最高的一条清洗后的 3Di 序列。
    由于开启了采样 (do_sample=True)，多次调用此函数会返回不同结果。
    """
    if not prompt_text.endswith("."):
        prompt_text += "."
        
    generation_inputs = T2StrucPrepareGenerationInputs(t2struc, prompt_text, text_tokenizer, structure_tokenizer)
    generation_config = T2StrcuDefaultGenerationConfig()
    
    # 执行生成
    results = T2StrucGeneration(t2struc, generation_inputs, generation_config, structure_tokenizer)
    
    # 排序并取 Top 1
    zipped_results = zip(results["structure"], results['structrue_logp'])
    best_candidate = sorted(zipped_results, key=lambda x: x[1], reverse=True)[0]
    
    # 清理序列
    return clean_sequence(best_candidate[0])


def _batch_prostt5_translate(
    candidates_3di: List[str],
    prostt5_model: torch.nn.Module,
    prostt5_tokenizer: Any
) -> List[str]:
    """
    [内部工具函数] 批量将 3Di 序列翻译为 AA 序列。
    """
    if not candidates_3di:
        return []
        
    # 构造输入
    batch_input_texts = ["<fold2AA>" + " " + seq for seq in candidates_3di]
    
    ids = prostt5_tokenizer.batch_encode_plus(
        batch_input_texts,
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    ).to(DEVICE)

    with torch.no_grad():
        backtranslations = prostt5_model.generate(
            ids.input_ids,
            attention_mask=ids.attention_mask,
            max_length=60, 
            min_length=5,
            num_return_sequences=1,
            **GEN_KWARGS_FOLD2AA
        )

    decoded = prostt5_tokenizer.batch_decode(backtranslations, skip_special_tokens=True)
    return ["".join(seq.split(" ")) for seq in decoded]


def generate_single_amp(
    prompt_text: str,
    t2struc: torch.nn.Module,
    text_tokenizer: Any,
    structure_tokenizer: Any,
    prostt5_model: torch.nn.Module,
    prostt5_tokenizer: Any,
) -> str:
    """
    [接口] 单次推理：生成 1 条 AMP 序列。
    """
    # 1. 获取 1 条最佳结构
    best_3di = _get_best_3di_candidate(prompt_text, t2struc, text_tokenizer, structure_tokenizer)
    # 2. 翻译为 AA
    results = _batch_prostt5_translate([best_3di], prostt5_model, prostt5_tokenizer)
    return results[0] if results else ""


def generate_k_amps(
    prompt_text: str,
    t2struc: torch.nn.Module,
    text_tokenizer: Any,
    structure_tokenizer: Any,
    prostt5_model: torch.nn.Module,
    prostt5_tokenizer: Any,
    k: int = 1
) -> List[str]:
    """
    [接口] 批量推理：为单个 Prompt 生成 K 条 AMP 序列。
    """
    # 1. 循环 K 次获取结构 (利用采样差异性)
    candidates_3di = []
    for _ in range(k):
        best_3di = _get_best_3di_candidate(prompt_text, t2struc, text_tokenizer, structure_tokenizer)
        candidates_3di.append(best_3di)
    
    # 2. 批量翻译
    return _batch_prostt5_translate(candidates_3di, prostt5_model, prostt5_tokenizer)


def run_batch_file_inference(
    t2struc_path: str, 
    prostt5_path: str, 
    json_path: str, 
    k: int = 1
) -> Dict[str, Any]:
    """读取 JSON 文件并进行批量生成"""
    
    # 路径兼容处理：如果传入的是目录，自动补全 pytorch_model.bin
    real_weights_path = t2struc_path
    if os.path.isdir(t2struc_path):
        real_weights_path = os.path.join(t2struc_path, "pytorch_model.bin")
        
    logger.info(f"正在加载 T2Struc: {real_weights_path}")
    logger.info(f"正在加载 ProstT5: {prostt5_path}")

    # 加载模型
    try:
        t2struc, txt_tok, struc_tok = load_T2Struc_and_tokenizers("T2struc-1.2B", real_weights_path)
        prostt5_model, prostt5_tok = load_prostt5_model(DEVICE, prostt5_path)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return {}

    # 读取数据
    if not os.path.exists(json_path):
        logger.error(f"未找到 JSON 文件: {json_path}")
        return {}
        
    with open(json_path, 'r', encoding='utf-8') as f:
        prompts_list = json.load(f).get("prompts", [])

    results_data = {"config": {"k": k}, "results": []}
    logger.info(f"开始处理 {len(prompts_list)} 个 Prompts，每个生成 {k} 条...")

    for item in prompts_list:
        p_id = item.get("id")
        prompt = item.get("prompt")
        if not prompt: continue
        
        try:
            sequences = generate_k_amps(prompt, t2struc, txt_tok, struc_tok, prostt5_model, prostt5_tok, k=k)
            results_data["results"].append({
                "id": p_id,
                "prompt": prompt,
                "generated_sequences": sequences
            })
            logger.info(f"ID {p_id} 完成。")
        except Exception as e:
            logger.error(f"ID {p_id} 出错: {e}")
            
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP 推理脚本")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="batch", help="运行模式")
    parser.add_argument("--prompt", type=str, default="Design a short antimicrobial peptide", help="单条推理的提示词")
    parser.add_argument("--json", type=str, default="/t9k/mnt/amp/prompt/100.json", help="批量推理的JSON路径")
    parser.add_argument("--output", type=str, default=None, help="批量结果保存路径 (.fasta)，不填则默认生成")
    parser.add_argument("--k", type=int, default=3, help="每个提示词生成的序列数")
    
    # 默认路径配置
    T2STRUC_DIR = "./finetuned_t2struct/2025-11-27_21-54-10/final_model/"
    PROSTT5_DIR = "./finetuned_prostt5/epo2lr1e4/checkpoint-5090"
    
    args = parser.parse_args()

    if args.mode == "single":
        # === 单条模式：直接打印 AA 序列 ===
        real_weights = os.path.join(T2STRUC_DIR, "pytorch_model.bin")
        try:
            t2, tt, st = load_T2Struc_and_tokenizers("T2struc-1.2B", real_weights)
            p5, p5t = load_prostt5_model(DEVICE, PROSTT5_DIR)
            
            logger.info(f"Prompt: {args.prompt}")
            res = generate_single_amp(args.prompt, t2, tt, st, p5, p5t)
            
            # 直接输出纯净的序列
            print(f"\n[Generated AA Sequence]:\n{res}")
        except Exception as e:
            logger.error(f"推理失败: {e}")
        
    else:
        # === 批量模式：保存为 FASTA (ID + Prompt) ===
        output_data = run_batch_file_inference(T2STRUC_DIR, PROSTT5_DIR, args.json, k=args.k)
        
        if output_data and output_data.get("results"):
            # 确定输出文件名
            if args.output:
                fasta_path = args.output
            else:
                base_name = os.path.basename(args.json).split('.')[0]
                fasta_path = f"{base_name}_results.fasta"

            logger.info(f"正在写入 FASTA 文件: {fasta_path}")
            
            count = 0
            with open(fasta_path, 'w', encoding='utf-8') as f:
                for item in output_data["results"]:
                    p_id = item.get("id", "unknown")
                    # 获取 Prompt 并移除换行符，防止破坏 FASTA 格式
                    raw_prompt = item.get("prompt", "")
                    clean_prompt = raw_prompt.replace("\n", " ").replace("\r", "").strip()
                    
                    seqs = item.get("generated_sequences", [])
                    
                    for i, seq in enumerate(seqs):
                        # FASTA Header 格式: >ID_序号 Prompt内容
                        header = f">{p_id}_{i} {clean_prompt}"
                        f.write(f"{header}\n{seq}\n")
                        count += 1
            
            logger.info(f"完成！共写入 {count} 条序列到 {os.path.abspath(fasta_path)}")
        else:
            logger.warning("未生成任何有效结果，文件未保存。")