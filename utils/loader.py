from Prostt5Loader import load_prostt5_model
from T2structLoader import load_T2Struc_and_tokenizers
import time 
import torch 

def get_current_memory_mb():
    """获取当前显存占用 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"初始显存占用: {get_current_memory_mb():.2f} MB")

    # --- 加载 ProstT5 并计算显存 ---
    mem_before_prost = get_current_memory_mb()
    prostt5_model, prostt5_tokenizer = load_prostt5_model(device)
    mem_after_prost = get_current_memory_mb()
    
    prost_usage = mem_after_prost - mem_before_prost

    if prostt5_model is not None and prostt5_tokenizer is not None:
        print("ProstT5 模型和分词器加载成功。")
        print(f"-> ProstT5 模型显存占用: {prost_usage:.2f} MB")
    
    # --- 加载 T2Struc 并计算显存 ---
    mem_before_t2 = get_current_memory_mb()
    # 注意：如果 load_T2Struc_and_tokenizers 内部没有自动将模型 .to(device)，
    # 这里的显存变化可能为 0。如果该 loader 默认在 CPU，你可能需要手动在此处移动模型到 GPU。
    t2struc_model, t2s_text_tokenizer, t2s_structure_tokenizer = load_T2Struc_and_tokenizers()
    
    # 如果 loader 没有把模型放进 GPU，通常需要手动放进去才能看到显存变化
    # if t2struc_model is not None:
    #     t2struc_model = t2struc_model.to(device) 
    
    mem_after_t2 = get_current_memory_mb()
    t2_usage = mem_after_t2 - mem_before_t2

    if t2struc_model is not None and t2s_text_tokenizer is not None and t2s_structure_tokenizer is not None:
        print("T2Struc 模型和分词器加载成功。")
        print(f"-> T2Struc 模型显存占用: {t2_usage:.2f} MB")

    # --- 总结 ---
    print(f"当前总显存占用: {get_current_memory_mb():.2f} MB")
    print("等待 60 秒钟...")
    time.sleep(60)