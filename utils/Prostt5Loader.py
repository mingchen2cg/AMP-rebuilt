from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch 
import time  # 1. 导入 time 模块


# ProstT5 的生成参数
GEN_KWARGS_FOLD2AA = {
    "do_sample": True,
    "top_p": 0.85,
    "temperature": 1.0,
    "top_k": 3,
    "repetition_penalty": 1.5,
}


def load_prostt5_model(device, weights_path="./weights/ProstT5"):
    """
    加载 ProstT5 模型及其分词器。
    """
    # logger.info("开始加载 ProstT5 模型...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(weights_path, do_lower_case=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(weights_path).to(device)
        model.float() if device == 'cpu' else model.half()
        # logger.info("ProstT5 模型加载成功。")
        return model, tokenizer
    except Exception as e:
        # logger.error(f"加载 ProstT5 模型或分词器失败，请检查 './weights/ProstT5' 目录中是否包含完整文件。")
        # logger.error(f"错误信息: {e}")
        print(f"加载 ProstT5 模型或分词器失败，请检查 '{weights_path}' 目录中是否包含完整文件。")
        return None, None
