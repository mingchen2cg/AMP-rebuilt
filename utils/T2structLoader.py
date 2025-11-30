import os
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmTokenizer, GenerationConfig
from models.StructureTokenPredictionModel import StructureTokenPredictionModel

# 全局设备设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 辅助 Lambda：将 Tensor 转换为 List
TO_LIST = lambda seq: [
    seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])
]

def load_T2Struc(cfg, weights_path):
    """
    初始化模型架构并加载权重。
    """
    print("正在初始化 T2Struc 模型架构...")
    # 根据训练配置，使用 bfloat16 初始化模型
    model = StructureTokenPredictionModel(cfg.model).to(torch.bfloat16)

    # 确定权重文件路径
    if os.path.isdir(weights_path):
        weights_file = os.path.join(weights_path, "pytorch_model.bin")
    else:
        weights_file = weights_path

    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"未找到权重文件: {weights_file}")

    print(f"正在加载权重: {weights_file}")
    # 加载权重到 CPU 防止显存瞬间溢出，之后再移动
    state_dict = torch.load(weights_file, map_location='cpu')
    model.load_state_dict(state_dict)

    return model.to(DEVICE)

def load_T2Struc_and_tokenizers(model_name="T2struc-1.2B", weights_path=None):
    """
    加载 T2Struc 模型、配置文件和分词器。
    
    Args:
        model_name (str): 模型名称（用于兼容日志打印）。
        weights_path (str): 包含 'config.yaml' 和 'pytorch_model.bin' 的目录路径或具体文件路径。
    """
    if not weights_path or not os.path.exists(weights_path):
        raise ValueError("请提供有效的 'weights_path'，该路径应包含 config.yaml 和 pytorch_model.bin")

    # 1. 确定配置文件的路径
    # 如果传入的是 pytorch_model.bin 文件路径，则配置文件应在同一目录下
    if os.path.isfile(weights_path):
        base_dir = os.path.dirname(weights_path)
    else:
        base_dir = weights_path
    
    config_path = os.path.join(base_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未在以下路径找到配置文件 config.yaml: {config_path}")

    print(f"正在从 {config_path} 加载配置...")
    cfg = OmegaConf.load(config_path)

    # 2. 加载模型
    model = load_T2Struc(cfg, weights_path)
    model.eval()

    # 3. 加载分词器
    print("正在加载分词器...")
    # 注意：这里假设配置文件中的路径是正确的，或者您可以根据需要在此处覆盖路径
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(cfg.model["lm"])
        structure_tokenizer = EsmTokenizer.from_pretrained(cfg.model["tokenizer"])
    except Exception as e:
        print(f"分词器加载警告: {e}。尝试使用默认配置或检查 config.yaml 中的路径。")
        # 如果 config 中的路径是本地绝对路径且在当前环境不存在，可能需要回退逻辑
        raise e
    
    print("T2Struc 模型和分词器加载成功。")
    return model, text_tokenizer, structure_tokenizer


# --- 生成相关的辅助函数 (保持原样供 inference.py 调用) ---

def T2StrcuDefaultGenerationConfig():
    return GenerationConfig(
        temperature=1,
        top_k=40,
        top_p=1,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.0,
        max_length=60,
        min_length=5,
    )

def T2StrucPrepareGenerationInputs(t2struc, desc, text_tokenizer, structre_tokenizer):
    batch = {}
    desc_encodings = text_tokenizer(
        desc,
        return_tensors="pt",
        max_length=768,
        truncation=True,
        padding="longest",
    )
    batch["text_ids"] = desc_encodings.input_ids.to(DEVICE)
    batch["text_masks"] = desc_encodings.attention_mask.to(DEVICE)
    
    # 获取 Encoder 隐状态
    text_hidden_states, text_attention_mask = t2struc.infer_text(batch)
    
    start_id = structre_tokenizer.cls_token_id
    stop_id = structre_tokenizer.eos_token_id
    pad_id = structre_tokenizer.pad_token_id
    input_ids = (torch.zeros((1)) + start_id).unsqueeze(0).to(torch.long).to(DEVICE)
    
    return {
        "input_ids": input_ids,
        "bos_token_id": start_id,
        "eos_token_id": stop_id,
        "pad_token_id": pad_id, 
        "encoder_hidden_states": text_hidden_states,
        "encoder_attention_mask": text_attention_mask,
    }

def T2StrucGeneration(t2struc, T2StrucGenerationDict, T2StrucGenerationConfig, structre_tokenizer):
    sample_results = t2struc.plm.generate(
        do_sample=True,
        generation_config=T2StrucGenerationConfig,
        num_return_sequences=5,
        return_dict_in_generate=True, 
        **T2StrucGenerationDict
    )
    return {
        "structure": structre_tokenizer.batch_decode(TO_LIST(sample_results.sequences)),
        "structrue_logp": torch.sum(sample_results.log_p, dim=-1).cpu().numpy().tolist(),
    }