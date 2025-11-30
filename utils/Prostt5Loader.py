from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch 
import time

# ProstT5 的生成参数
GEN_KWARGS_FOLD2AA = {
    "do_sample": True,
    "top_p": 0.85,
    "temperature": 1.0,
    "top_k": 3,
    "repetition_penalty": 1.5,
}

def load_prostt5_model(device, weights_path="./weights/ProstT5", torch_dtype=None):
    """
    加载 ProstT5 模型及其分词器。
    
    Args:
        device (str): 设备 ('cuda' or 'cpu')
        weights_path (str): 权重路径
        torch_dtype (torch.dtype, optional): 指定加载精度。
                                           如果不填，则默认根据设备自动处理(GPU下转为half/fp16以优化推理显存)。
                                           训练时建议显式传入 torch.bfloat16。
    """
    try:
        tokenizer = T5Tokenizer.from_pretrained(weights_path, do_lower_case=False)
        
        # 如果指定了精度（例如训练时），直接在加载时指定，并不再做后续的手动类型转换
        if torch_dtype is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(weights_path, torch_dtype=torch_dtype).to(device)
        else:
            # 默认行为（保持原逻辑，主要用于推理优化）
            model = AutoModelForSeq2SeqLM.from_pretrained(weights_path).to(device)
            if device == 'cpu':
                model.float()
            else:
                model.half()
        
        return model, tokenizer
    except Exception as e:
        print(f"加载 ProstT5 模型或分词器失败，请检查 '{weights_path}' 目录中是否包含完整文件。")
        print(f"错误详情: {e}")
        return None, None