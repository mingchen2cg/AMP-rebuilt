import os
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmTokenizer
from transformers import GenerationConfig
from models.StructureTokenPredictionModel import StructureTokenPredictionModel

# 全局设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 辅助 Lambda
TO_LIST = lambda seq: [
    seq[i, ...].detach().cpu().numpy().tolist() for i in range(seq.shape[0])
]

def load_T2Struc(cfg, model_name):
    model =  StructureTokenPredictionModel(cfg.model).to(torch.bfloat16) # Since we train the model with bfloat16

    model_path = os.path.join(MODEL_ROOT, model_name, T2STRUCT)
    if not os.path.exists(model_path):    
        model_dir = os.path.join(MODEL_ROOT, model_name)
        files = [f for f in os.listdir(model_dir) if f.startswith("pytorch_model_part")]
        if len(files) == 12: # since we split the model into 12 parts
            # call os.system cat to merge the parts
            logger.info(f"{model_name} is split into 12 parts, merging...")
            os.system(f"cat {model_dir}/pytorch_model_part* > {model_dir}/pytorch_model.bin")
            logger.info(f"{model_name} merged successfully.")
            os.system(f"rm {model_dir}/pytorch_model_part*")
        else:
            raise FileNotFoundError(f"Model {model_name} not found.")

    model.load_state_dict(torch.load(os.path.join(MODEL_ROOT, model_name, T2STRUCT), map_location='cpu'))

    return model.to(DEVICE)


def load_T2Struc_tokenizers(cfg):
    text_tokenizer =  AutoTokenizer.from_pretrained(cfg.model["lm"])
    structure_tokenizer = EsmTokenizer.from_pretrained(cfg.model["tokenizer"])
    return text_tokenizer, structure_tokenizer


def load_T2Struc_and_tokenizers():
    # assert T2struc_NAME in T2struc_NAMES
    logger.info("T2strcu Model: " + T2struc_NAME)
    cfg = OmegaConf.load(os.path.join(MODEL_ROOT, T2struc_NAME, "config.yaml"))
    model = load_T2Struc(cfg, T2struc_NAME)
    model = model.eval()
    text_tokenizer, structre_tokenizer = load_T2Struc_tokenizers(cfg)
    return model, text_tokenizer, structre_tokenizer

# --- Generation 相关的辅助函数 (保持原样供第一段代码调用) ---

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