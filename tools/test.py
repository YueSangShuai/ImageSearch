import torch
import os

def compare_pt_files(file1_path, file2_path, eps=1e-6):
    """
    对比两个.pt文件是否完全一致（支持单个张量、字典、列表等结构）
    
    Args:
        file1_path: 第一个.pt文件路径
        file2_path: 第二个.pt文件路径
        eps: 浮点数比较的容差（处理浮点精度误差）
    
    Returns:
        bool: 两个文件是否一致
        str: 不一致的详细信息（若不一致）
    """
    # 检查文件是否存在
    if not os.path.exists(file1_path):
        return False, f"文件不存在: {file1_path}"
    if not os.path.exists(file2_path):
        return False, f"文件不存在: {file2_path}"
    
    try:
        # 加载两个文件的数据（强制CPU，避免设备差异）
        data1 = torch.load(file1_path, map_location="cpu")
        data2 = torch.load(file2_path, map_location="cpu")
    except Exception as e:
        return False, f"加载文件失败: {str(e)}"
    
    # 递归对比数据（支持嵌套结构）
    def _compare(a, b, path=""):
        # 类型必须完全一致
        if type(a) != type(b):
            return False, f"类型不一致 {path}：{type(a)} vs {type(b)}"
        
        # 处理张量
        if isinstance(a, torch.Tensor):
            # 形状对比
            if a.shape != b.shape:
                return False, f"形状不一致 {path}：{a.shape} vs {b.shape}"
            # 数据类型对比
            if a.dtype != b.dtype:
                return False, f"数据类型不一致 {path}：{a.dtype} vs {b.dtype}"
            # 数值对比（处理浮点精度）
            if not torch.allclose(a, b, atol=eps, rtol=eps):
                max_diff = torch.abs(a - b).max().item()
                return False, f"数值不一致 {path}（最大差异: {max_diff:.6f}）"
            return True, ""
        
        # 处理字典
        elif isinstance(a, dict):
            # 键集合必须一致
            if set(a.keys()) != set(b.keys()):
                return False, f"字典键不一致 {path}：{set(a.keys())} vs {set(b.keys())}"
            # 逐个键对比值
            for key in a.keys():
                sub_path = f"{path}.{key}" if path else key
                match, msg = _compare(a[key], b[key], sub_path)
                if not match:
                    return False, msg
            return True, ""
        
        # 处理列表/元组
        elif isinstance(a, (list, tuple)):
            # 长度必须一致
            if len(a) != len(b):
                return False, f"长度不一致 {path}：{len(a)} vs {len(b)}"
            # 逐个元素对比
            for i, (item1, item2) in enumerate(zip(a, b)):
                sub_path = f"{path}[{i}]"
                match, msg = _compare(item1, item2, sub_path)
                if not match:
                    return False, msg
            return True, ""
        
        # 处理其他基本类型（int/str/float等）
        else:
            if a != b:
                return False, f"值不一致 {path}：{a} vs {b}"
            return True, ""
    
    # 执行对比
    match, message = _compare(data1, data2)
    return match, message

# 示例用法
if __name__ == "__main__":
    # 替换为实际文件路径
    file1 = "/data/yuesang/LLM/contrastors/extracted_data/vision_inputs/batch_0_sample_0.pt"
    file2 = "/data/yuesang/LLM/contrastors/src/extracted_data/all_image_embeddings.pt"
    
    is_match, info = compare_pt_files(file1, file2)
    if is_match:
        print(f"✅ 两个.pt文件完全一致")
    else:
        print(f"❌ 两个.pt文件不一致：{info}")