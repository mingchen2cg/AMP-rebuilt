"序列处理工具模块"


def clean_sequence(seq: str) -> str:
    """
    清理序列，去掉特殊标记 <cls>, <eos>, <pad>，只保留小写字母并用空格分隔。
    (此函数来自您提供的脚本)
    """
    tokens = seq.split()
    filtered = [tok for tok in tokens if tok.isalpha() and tok.islower() and len(tok) == 1]
    return " ".join(filtered)