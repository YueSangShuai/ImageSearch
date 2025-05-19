import gradio as gr
from imagesearch import ImageSearchAPI
from options import parse_args
import time
import threading
from detect_person import PedestrianDetector
from PIL import Image, ImageDraw
import numpy as np
import os
import uuid
import io

def additional_args(parser):
    parser.add_argument('--port', type=int, default=7868)
    parser.add_argument('--person_model', type=str, help='Path to ONNX model file', default="det_coco_s.onnx")
    parser.add_argument('--person_threshold', type=float, default=0.65, 
                       help='Detection confidence threshold for person (default: 0.5)')
    parser.add_argument('--name', help='name indicator', default='行人搜索概念验证版(V0.1_s3)')


# Initialize the image search API
args = parse_args(additional_args)
image_search = ImageSearchAPI(args.model_path, args.device, args.qdrant_host, args.qdrant_port, args.collection_name)

pedestrianDetector = PedestrianDetector(args.person_model, args.person_threshold)

SEARCH_SHORTCUTS = [
    '做保洁的阿姨',
    'wearing a 红色T恤',
    '玩手机的男子',
    '穿高跟鞋的女子',
    '背双肩包的男孩',
    '一只手捂住嘴的男人'
]

IMAGE_SAMPLES = [
    'data/calling_man.png',
    'data/p44.png',
    'data/little_girl.png',
    'data/multiple_person.jpg',
    'data/p11.png'
]

def update_record_count():
    """向量数据库记录数量"""
    return image_search.count_records()


def search_images(query, top_k=5, threshold=0.5):
    results, total_time = image_search.search(query, top_k, threshold, return_payload=True)
    infos = []
    for result in results:
        payload = result.payload
        path = payload['path']
        category = payload['category']
        bbox = payload['bbox'] if 'bbox' in payload else None
        score = result.score
        if bbox is None:
            infos.append((path, f'{path}: {score: .3f}'))
        else:
            info = f'{path}: {bbox}: {score: .3f}'
            image = Image.open(path).convert("RGB")
            # 创建遮罩，将检测框区域设置为黑色
            mask = Image.new("L", image.size, 128)
            draw = ImageDraw.Draw(mask)
            draw.rectangle(bbox, fill=0)
            # 应用遮罩
            image.paste((255, 255, 255), mask)
            infos.append((image, info))
    return infos, f'图片数量: {update_record_count()}，查询耗时: {total_time:.3f}秒'


# 安全的后台刷新函数
def safe_background_refresh():
    """在后台安全地刷新记录数量"""
    while True:
        time.sleep(60)  # 每60秒刷新一次
        try:
            # 只是获取最新计数，但不尝试更新UI
            count = image_search.count_records()
            print(f"后台刷新: 当前向量数量 = {count}")
        except Exception as e:
            print(f"后台刷新出错: {e}")


def delete_image(images, selected):
    image = images[selected]
    image = image[1].split(": ")[0]
    print("selected:", selected, image)
    del images[selected]
    image_search.delete_image(image)
    return images, min(selected, len(images) - 1)


def on_select(evt: gr.SelectData, images):
    """处理图片选择事件，返回选中图片的索引"""
    return evt.index


def set_and_search(query_text):
    return query_text, gr.update(visible=True)


# 图像搜索函数
def search_by_image(input_image, person_threshold=0.7, top_k=5, threshold=0.5):
    """基于上传图像中的行人进行搜索"""
    if input_image is None:
        return None, "请先上传包含行人的图像"
    
    # 将图像转换为PIL格式
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    
    # 计时：检测阶段开始
    detection_start = time.time()
    
    # 使用行人检测器检测图像中的行人
    detections = pedestrianDetector.detect_image(image_pil=input_image, threshold=person_threshold, output_path=None)
    
    # 计时：检测阶段结束
    detection_end = time.time()
    detection_time = detection_end - detection_start
    
    if not detections:
        return None, "未检测到行人，请上传包含行人的图像"
    
    # 创建临时目录用于保存裁剪的行人图像
    temp_dir = "temp_person_crops"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 生成检测结果图像
    detection_image = input_image.copy()
    draw = ImageDraw.Draw(detection_image)
    
    # 为每个检测到的人分配颜色
    person_colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
    ]
    
    # 筛选有效的检测结果
    valid_detections = []
    for i, (x1, y1, x2, y2, score) in enumerate(detections):
        valid_detections.append((x1, y1, x2, y2, score))
    
    # 所有人物的搜索结果
    all_person_results = []
    total_search_time = 0
    
    # 对每个检测到的行人进行处理
    for idx, (x1, y1, x2, y2, score) in enumerate(valid_detections):
        # 为每个人分配颜色
        color_idx = idx % len(person_colors)
        color = person_colors[color_idx]
        
        # 在图像上绘制边界框和编号
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        person_id = f"#{idx+1}"
        draw.text((x1+5, y1-20), person_id, fill=color, stroke_width=1)
        
        # 裁剪行人区域
        person_crop = input_image.crop((x1, y1, x2, y2))
        
        # 保存裁剪图像
        crop_filename = f"{temp_dir}/person_{idx+1}_{uuid.uuid4()}.jpg"
        person_crop.save(crop_filename)
        
        # 提取特征并搜索
        results, search_time = image_search.search_by_image(person_crop, top_k, threshold)
        total_search_time += search_time
        
        # 将此人的搜索结果添加到列表中
        person_results = []
        for img_path, score in results:
            person_results.append((img_path, score))
        
        # 将此人的结果添加到所有结果中
        all_person_results.append({
            "person_id": idx + 1,
            "color": color,
            "crop": person_crop,
            "results": person_results
        })
    
    # 准备展示结果
    gallery_results = [(detection_image, "检测结果")]
    
    # 为每个人添加搜索结果
    for person_data in all_person_results:
        person_id = person_data["person_id"]
        color = person_data["color"]
        crop = person_data["crop"]
        results = person_data["results"]
        
        # 添加人物裁剪图像
        gallery_results.append((crop, f"行人 #{person_id}"))
        
        # 添加此人的搜索结果
        for img_path, score in results:
            gallery_results.append((img_path, f"#{person_id} | {img_path} : {score:.3f}"))
    
    total_time = detection_time + total_search_time
    return gallery_results, f'注册数: {image_search.count_records()}，检测到人数：{len(valid_detections)}，检测: {detection_time:.3f}秒，搜索: {total_search_time:.3f}秒'


# Create Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# 火星慧知矢量发动机(Vector Engine)"+args.name)

    with gr.Tabs() as tabs:
        with gr.TabItem("文本搜索"):
            with gr.Column():
                query = gr.Textbox(label="输入行人的描述", placeholder="输入查询内容...")
                # Search shortcuts
                shortcut_buttons = []
                with gr.Row():
                    for shortcut in SEARCH_SHORTCUTS:
                        shortcut_buttons.append(gr.Button(shortcut))

                with gr.Row():
                    record_count = gr.Textbox(label="查询信息", value=f'注册图片数量: {image_search.count_records()}', interactive=False)
                    top_k = gr.Slider(minimum=1, maximum=20, step=1, value=10, label="Top K")
                    threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.1, label="Threshold")
                with gr.Row():
                    query_button = gr.Button("查询")
                    clear_button = gr.Button("清除")

            # 存储查询结果和当前选中的索引
            selected_index = gr.Number(label="选中图片索引", visible=False, precision=0)
            gallery = gr.Gallery(label="搜索结果", show_label=True,
                                preview=True, object_fit="contain")

            # 当用户选择图片时
            gallery.select(on_select, inputs=[gallery], outputs=selected_index)

            query_button.click(fn=search_images, inputs=[query, top_k, threshold], outputs=[gallery, record_count])
            query.submit(search_images, inputs=[query, top_k, threshold], outputs=[gallery, record_count])
            clear_button.click(lambda: ("", [], None), outputs=[query, gallery, selected_index])

            # 所有UI组件都定义好后，再添加事件处理
            for i, shortcut in enumerate(SEARCH_SHORTCUTS):
                shortcut_buttons[i].click(
                    lambda s=shortcut: s,
                    outputs=query
                ).then(
                    search_images,
                    inputs=[query, top_k, threshold],
                    outputs=[gallery, record_count]
                )
                
        with gr.TabItem("以图搜图"):
            with gr.Row():
                # 左侧：上传图像控件
                with gr.Column(scale=2):
                    image_input = gr.Image(label="上传包含行人的图片", type="pil", height=256)
                
                # 右侧：样例图片
                with gr.Column(scale=1):
                    gr.Examples(label="样例图片", examples=IMAGE_SAMPLES,
                        inputs=[image_input])

            with gr.Row():
                image_record_count = gr.Textbox(label="查询信息", value=f'注册图片数量: {image_search.count_records()}', interactive=False)
                image_top_k = gr.Slider(minimum=1, maximum=20, step=1, value=10, label="Top K")
                image_threshold = gr.Slider(minimum=0.2, maximum=1.0, step=0.05, value=0.8, label="Threshold")
                person_threshold = gr.Slider(minimum=0.4, maximum=1.0, step=0.05, value=0.6, label="Person Threshold")
            with gr.Row():
                image_search_button = gr.Button("检测并搜索")
                image_clear_button = gr.Button("清除")
                    
            image_gallery = gr.Gallery(label="搜索结果", show_label=True, preview=True, object_fit="contain")
            
            image_search_button.click(
                fn=search_by_image, 
                inputs=[image_input, person_threshold, image_top_k, image_threshold], 
                outputs=[image_gallery, image_record_count]
            )
            image_clear_button.click(
                lambda: (None, [], f'注册图片数量: {image_search.count_records()}'), 
                outputs=[image_input, image_gallery, image_record_count]
            )
            
            # 为每个样例图片按钮添加点击事件
            for i, sample in enumerate(IMAGE_SAMPLES):
                def load_sample_image(sample_path=sample):
                    try:
                        img = Image.open(sample_path)
                        return img
                    except Exception as e:
                        print(f"Error loading sample image {sample_path}: {e}")
                        return None

if __name__ == "__main__":
    # 启动安全的后台刷新线程
    # refresh_thread = threading.Thread(target=safe_background_refresh, daemon=True)
    # refresh_thread.start()
    
    interface.launch(
        allowed_paths=['data/CUHK-PEDES/', 'data/SYNTH-PEDES', 'data/COCO', 'data/imagenet', '/data/zkteco'],
        server_port=args.port)
