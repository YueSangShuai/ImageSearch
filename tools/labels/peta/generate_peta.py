import requests  # 用于发送HTTP请求
import datetime  # 用于记录时间
import logging  # 用于日志记录
import time  # 用于重试延迟
import os
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_result_requests(prompt, max_retries=2):
    """
    使用requests库直接调用API，不依赖OpenAI库
    
    参数:
        prompt: 提示词
        max_retries: 最大重试次数
    返回:
        API返回的文本内容
    """
    # API endpoint
    url = "http://10.12.16.123:8005/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-ollama"
    }
    
    for attempt in range(max_retries):
        try:
            # 记录尝试次数
            if attempt > 0:
                logger.info(f"第 {attempt+1} 次尝试调用API...")
            
            # 开始计时
            start_time = datetime.datetime.now()
            
            # 准备请求负载
            payload = {
                "model": "InternVL3-38B",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"帮我给这段描述{prompt}生成详细的英文描述,你只需要生成描述的相关语句,不用生成其他的比如说什么Sure这些"
                            }
                        ]
                    }
                ],
                "stream": False,
            }
            
            # 发送POST请求
            response = requests.post(url, headers=headers, json=payload)
            
            # 结束计时
            end_time = datetime.datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            # logger.info(f"API调用耗时: {processing_time:.2f}秒")
            
            # 解析响应
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.debug(f"API返回内容: {content[:100]}...")  # 只记录前100字符
                return content
            else:
                error_msg = f"API错误: {response.status_code}"
                logger.error(f"{error_msg} - 响应: {response.text[:200]}...")
                if 500 <= response.status_code < 600 and attempt < max_retries - 1:
                    logger.info("服务器错误，将在3秒后重试...")
                    time.sleep(3)
                    continue
                return None
                
        except Exception as e:
            logger.error(f"调用API时发生异常: {str(e)}", exc_info=True)
            if attempt < max_retries - 1:
                logger.info("将在3秒后重试...")
                time.sleep(3)
                continue
            return None

def read_txt_file_to_list(file_path):
    """
    按行读取文本文件并返回列表（每行作为一个元素）
    
    参数:
        file_path: 文本文件路径
    返回:
        包含所有行的列表
    """
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 去除换行符
                if line:  # 跳过空行
                    lines.append(line)
        logger.info(f"成功读取文件: {file_path}, 共{len(lines)}行")
        return lines
    except FileNotFoundError:
        logger.error(f"错误：文件 {file_path} 未找到！")
        return []
    except Exception as e:
        logger.error(f"读取文件时发生错误: {e}")
        return []

def process_txt_files(txt_path_list):
    """
    处理多个txt文件，逐条立即写入结果到对应的descrip.txt文件
    
    参数:
        txt_path_list: 包含所有输入txt文件路径的列表
    """
    for txt_path in txt_path_list:
        # 检查输入文件是否存在
        if not os.path.exists(txt_path):
            logger.error(f"输入文件不存在: {txt_path}")
            continue
            
        # 读取输入文件内容
        txt_results = read_txt_file_to_list(txt_path)
        if not txt_results:
            logger.warning(f"文件 {txt_path} 没有有效数据可处理")
            continue
        
        # 准备输出文件路径
        save_txt = txt_path+"descrip.txt"
        
        # 如果输出文件已存在，先删除（避免追加到旧文件）
        if os.path.exists(save_txt):
            os.remove(save_txt)
            logger.info(f"已删除旧文件: {save_txt}")
        
        # 处理每一行并立即写入结果
        logger.info(f"开始处理文件: {txt_path}, 共{len(txt_results)}行")
        with tqdm(total=len(txt_results), desc=f"处理 {os.path.basename(txt_path)}") as pbar:
            for one_txt_result in txt_results:
                try:
                    # 分割标签
                    tags = one_txt_result.split()  # 第一个元素是id，后面是标签
                    
                    # 提取id和清理后的标签
                    if not tags:
                        continue
                        
                    id = tags[0]
                    tags = tags[1:]
                    cleaned_tag_string = ' '.join(tags)
                    prompt = cleaned_tag_string
                    
                    # 获取API结果
                    temp = get_result_requests(prompt)
                    
                    if temp is None:
                        result_line = f"{id} API调用失败"
                        logger.warning(f"ID {id} API调用失败")
                    else:
                        result_line = f"{id} {temp}"
                    
                    # 立即写入文件（追加模式）
                    with open(save_txt, 'a', encoding='utf-8') as f:
                        f.write(result_line + '\n')
                    
                    # 更新进度条
                    pbar.update(1)
                    
                except Exception as e:
                    error_msg = f"处理行时发生错误: {str(e)}"
                    logger.error(error_msg)
                    with open(save_txt, 'a', encoding='utf-8') as f:
                        f.write(f"{one_txt_result} 处理失败: {str(e)}\n")
                    pbar.update(1)
                    continue
        
        logger.info(f"完成处理文件: {txt_path}, 结果已保存到: {save_txt}")

if __name__ == "__main__":
    # 输入文件路径列表
    txt_path = [
        "/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/checked_test_lb.txt",
        "/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/checked_train_lb.txt",
        "/data/yuesang/LLM/contrastors/data/pa100k_label/label/yinshe/checked_val_lb.txt"
    ]
    
    # 检查路径是否存在
    valid_paths = [path for path in txt_path if os.path.exists(path)]
    if len(valid_paths) != len(txt_path):
        missing_paths = set(txt_path) - set(valid_paths)
        logger.warning(f"以下路径不存在: {missing_paths}")
    
    # 开始处理
    if valid_paths:
        process_txt_files(valid_paths)
    else:
        logger.error("没有有效的输入文件路径")