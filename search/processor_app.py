from flask import Blueprint, request, jsonify
import os
import time
import uuid
import threading
from PIL import Image, ImageDraw
import concurrent.futures
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

class Config:
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--port', type=int, default=5005)
        parser.add_argument('--name', help='name indicator', default='行人搜索概念验证版(V0.1_s5)')
        parser.add_argument('--image-root', type=str, default='xm_images')
        return parser

    def __init__(self, text_extractor, image_search, pedestrianDetector, 
        port: int = 5005,
        name: str = '行人搜索概念验证版(V0.1_s5)',
        image_root: str = 'xm_images',
        **kwargs):

        self.port = port
        self.name = name
        self.image_root = image_root

        self.text_extractor = text_extractor
        self.image_search = image_search
        self.pedestrianDetector = pedestrianDetector

        # 路径
        self.UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        self.CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_cache")
        self.TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_person_crops")
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)

        # Flask 配置
        self.SECRET_KEY = 'secret!'
        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

        # 线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        # 缓存
        self.IMAGE_CACHE = {}
        self.MAX_CACHE_SIZE = 50

        # 搜索快捷短语和样例
        self.SEARCH_SHORTCUTS = [
            '做保洁的阿姨',
            'a person wearing a red T-shirt',
            'a man playing with a phone',
            'a woman wearing high heels',
            'a boy wearing a double shoulder bag',
            '抽烟的人'
        ]
        self.IMAGE_SAMPLES = [
            '/images/calling_man.png',
            '/images/p44.png',
            '/images/little_girl.png',
            '/images/multiple_person.jpg',
            '/images/p11.png'
        ]

        # 图像处理参数
        self.THUMBNAIL_SIZE = (128, 128)
        self.PREVIEW_SIZE = (512, 512)
        self.FULL_SIZE = (1024, 1024)
        self.QUALITY_LEVELS = {'thumbnail': 85, 'preview': 90, 'full': 95}
        self.IMAGE_FORMATS = {'thumbnail': 'WEBP', 'preview': 'WEBP', 'full': 'JPEG'}

        # 选中集合
        self.selected_collections = []

# 假设 config 是从主程序传入的
def create_image_search_blueprint(config):
    bp = Blueprint('image_search', __name__)

    # 工具函数全部改为接收 config 参数
    def get_cache_key(config, path, bbox=None, size=None, quality=None, format=None):
        key_parts = [path]
        if bbox:
            key_parts.append(f"bbox_{bbox}")
        if size:
            key_parts.append(f"size_{size[0]}x{size[1]}")
        if quality:
            key_parts.append(f"q{quality}")
        if format:
            key_parts.append(format.lower())
        return "_".join(map(str, key_parts))

    def generate_thumbnail(config, path, bbox=None, size=None):
        size = size or config.THUMBNAIL_SIZE
        try:
            cache_key = get_cache_key(config, path, bbox, size, config.QUALITY_LEVELS['thumbnail'], config.IMAGE_FORMATS['thumbnail'])
            if cache_key in config.IMAGE_CACHE:
                return config.IMAGE_CACHE[cache_key]
            thumb_filename = os.path.join(config.CACHE_DIR, f"{os.path.basename(path)}_{hash(cache_key)}.webp")
            if os.path.exists(thumb_filename):
                config.IMAGE_CACHE[cache_key] = thumb_filename
                return thumb_filename
            img = Image.open(path).convert("RGB")
            if bbox and all(isinstance(x, (int, float)) for x in bbox if x is not None):
                x1, y1, x2, y2 = bbox
                img_width, img_height = img.size
                width, height = x2 - x1, y2 - y1
                padding_x, padding_y = int(width * 0.2), int(height * 0.2)
                x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                x2, y2 = min(img_width, x2 + padding_x), min(img_height, y2 + padding_y)
                img = img.crop((x1, y1, x2, y2))
            img.thumbnail(size, Image.LANCZOS)
            img.save(thumb_filename, config.IMAGE_FORMATS['thumbnail'],
                    quality=config.QUALITY_LEVELS['thumbnail'], method=6, lossless=False)
            config.IMAGE_CACHE[cache_key] = thumb_filename
            return thumb_filename
        except Exception as e:
            print(f"生成缩略图出错 {path}: {e}")
            return None

    def generate_preview(config, path, bbox=None, size=None):
        size = size or config.PREVIEW_SIZE
        try:
            cache_key = get_cache_key(config, path, bbox, size, config.QUALITY_LEVELS['preview'], config.IMAGE_FORMATS['preview'])
            if cache_key in config.IMAGE_CACHE:
                return config.IMAGE_CACHE[cache_key]
            preview_filename = os.path.join(config.CACHE_DIR, f"{os.path.basename(path)}_preview_{hash(cache_key)}.webp")
            if os.path.exists(preview_filename):
                config.IMAGE_CACHE[cache_key] = preview_filename
                return preview_filename
            img = Image.open(path).convert("RGB")
            if bbox and all(isinstance(x, (int, float)) for x in bbox if x is not None):
                x1, y1, x2, y2 = bbox
                img_width, img_height = img.size
                width, height = x2 - x1, y2 - y1
                padding_x, padding_y = int(width * 0.2), int(height * 0.2)
                x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                x2, y2 = min(img_width, x2 + padding_x), min(img_height, y2 + padding_y)
                img = img.crop((x1, y1, x2, y2))
            img.thumbnail(size, Image.LANCZOS)
            img.save(preview_filename, config.IMAGE_FORMATS['preview'],
                    quality=config.QUALITY_LEVELS['preview'], method=4, lossless=False)
            config.IMAGE_CACHE[cache_key] = preview_filename
            return preview_filename
        except Exception as e:
            print(f"生成预览图出错 {path}: {e}")
            return None

    def process_image(config, path, bbox=None, max_size=None):
        max_size = max_size or config.FULL_SIZE[0]
        try:
            cache_key = get_cache_key(config, path, bbox, (max_size, max_size), config.QUALITY_LEVELS['full'], config.IMAGE_FORMATS['full'])
            if cache_key in config.IMAGE_CACHE:
                return config.IMAGE_CACHE[cache_key]
            processed_filename = os.path.join(config.CACHE_DIR, f"{os.path.basename(path)}_{hash(cache_key)}.jpg")
            if os.path.exists(processed_filename):
                config.IMAGE_CACHE[cache_key] = processed_filename
                return processed_filename
            preview_path = generate_preview(config, path, bbox)
            if not preview_path:
                preview_path = path
            config.executor.submit(process_full_image, config, path, bbox, cache_key, processed_filename)
            return preview_path
        except Exception as e:
            print(f"处理图片出错 {path}: {e}")
            return None

    def process_full_image(config, path, bbox, cache_key, target_filename):
        try:
            if os.path.exists(target_filename):
                config.IMAGE_CACHE[cache_key] = target_filename
                return target_filename
            img = Image.open(path).convert("RGB")
            if bbox and all(isinstance(x, (int, float)) for x in bbox if x is not None):
                x1, y1, x2, y2 = bbox
                img_width, img_height = img.size
                width, height = x2 - x1, y2 - y1
                padding_x, padding_y = int(width * 0.2), int(height * 0.2)
                x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                x2, y2 = min(img_width, x2 + padding_x), min(img_height, y2 + padding_y)
                img = img.crop((x1, y1, x2, y2))
            if max(img.size) > config.FULL_SIZE[0]:
                ratio = config.FULL_SIZE[0] / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            img.save(target_filename, config.IMAGE_FORMATS['full'],
                    quality=config.QUALITY_LEVELS['full'], optimize=True)
            config.IMAGE_CACHE[cache_key] = target_filename
            config.socketio.emit('image_ready', {'cache_key': cache_key, 'path': target_filename})
            return target_filename
        except Exception as e:
            print(f"处理完整图片出错 {path}: {e}")
            return None

    def update_record_count(config):
        selected_collections = config.selected_collections
        image_search = config.image_search
        if selected_collections:
            total_count = 0
            original_collection = image_search.collection_name
            try:
                for collection in selected_collections:
                    if collection in image_search.get_collections():
                        image_search.set_collection(collection)
                        count = image_search.count_records()
                        if count is not None:
                            total_count += count
                return total_count
            finally:
                image_search.set_collection(original_collection)
        count = image_search.count_records()
        return count if count is not None else 0

    # 路由部分全部改为依赖 config
    @bp.route('/')
    def index():
        return render_template('index.html',
                            title=f"火星慧知矢量发动机(Vector Engine) {config.name}",
                            shortcuts=config.SEARCH_SHORTCUTS,
                            samples=config.IMAGE_SAMPLES,
                            record_count=update_record_count(config))

    @bp.route('/api/get_collections', methods=['GET'])
    def get_collections():
        collections = config.image_search.get_collections()
        current_collection = config.image_search.collection_name
        return jsonify({'collections': collections, 'current': current_collection})

    @bp.route('/api/set_collection', methods=['POST'])
    def set_collection():
        data = request.json
        collection_name = data.get('collection_name', '')
        collections = data.get('collections', [])
        if not collection_name:
            return jsonify({'error': 'Collection name is required'}), 400
        config.selected_collections = collections
        success = config.image_search.set_collection(collection_name)
        if success:
            record_count = update_record_count(config)
            return jsonify({
                'success': True,
                'message': f'Switched to collection: {collection_name}',
                'selected_collections': collections,
                'record_count': record_count
            })
        else:
            return jsonify({'success': False, 'error': f'Failed to switch to collection: {collection_name}'}), 400

    @bp.route('/api/search', methods=['POST'])
    def search():
        data = request.json
        query = data.get('query', '')
        top_k = int(data.get('top_k', 10))
        threshold = float(data.get('threshold', 0.1))
        collections = data.get('collections', config.selected_collections)
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        start_time = time.time()
        if collections:
            results, search_time = config.image_search.search_in_collections(query, top_k, threshold, collections, return_payload=True)
        else:
            results, search_time = config.image_search.search(query, top_k, threshold, return_payload=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures, all_results = [], []
            for idx, result in enumerate(results):
                path = os.path.join(config.image_root, result['metadata']['image_path'])
                score = result['score']
                payload = result['metadata']
                if not os.path.exists(path):
                    continue
                bbox = payload.get('bbox') if payload and isinstance(payload, dict) else None
                preview_path = generate_preview(config, path, bbox)
                if not preview_path:
                    preview_path = path
                preview_filename = os.path.basename(preview_path)
                original_filename = os.path.basename(path)
                result_future = executor.submit(process_image, config, path, bbox)
                futures.append((result_future, path))
                all_results.append({
                    'type': 'result',
                    'path': preview_filename,
                    'original_path': original_filename,
                    'display_path': original_filename,
                    'label': f'#{idx+1} | {original_filename} : {score:.3f}',
                    'score': float(score)
                })
            for future, path in futures:
                try:
                    future.result(timeout=0.1)
                except concurrent.futures.TimeoutError:
                    pass
                except Exception as e:
                    print(f"处理图片出错 {path}: {e}")
        total_time = time.time() - start_time
        return jsonify({
            'results': all_results,
            'query': query,
            'search_time': search_time,
            'total_time': total_time,
            'record_count': update_record_count(config)
        })

    # 其余 API 路由同理，全部通过 config 访问依赖和配置
    # ...（省略，按上述模式迁移即可）

    @bp.route('/images/<path:filename>')
    def serve_image(filename):
        if os.path.exists(filename) and os.path.isfile(filename):
            return send_from_directory(os.path.dirname(filename), os.path.basename(filename))
        fn = os.path.join(config.image_root, filename)
        if os.path.exists(fn) and os.path.isfile(fn):
            return send_from_directory(os.path.dirname(fn), os.path.basename(fn))
        if "_" in filename and (filename.endswith(".jpg") or filename.endswith(".webp") or filename.endswith(".png")):
            cache_path = os.path.join(config.CACHE_DIR, filename)
            if os.path.exists(cache_path):
                return send_from_directory(config.CACHE_DIR, filename)
        return f"服务器错误: 图片未找到", 404

    @bp.route('/api/record_count', methods=['GET'])
    def get_record_count_api():
        count = update_record_count(config)
        return jsonify({'count': count})

    @bp.route('/api/register_status', methods=['GET'])
    def register_status_api():
        try:
            count = config.image_search.count_records()
            return jsonify({'count': count, 'status': 'ready'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/delete_image', methods=['POST'])
    def delete_image_api():
        data = request.json
        image_path = data.get('path', '')
        if not image_path:
            return jsonify({'error': 'Image path is required'}), 400
        try:
            config.image_search.delete_image(image_path)
            return jsonify({'success': True, 'message': f'Image {image_path} deleted successfully'})
        except Exception as e:
            return jsonify({'error': f'Failed to delete image: {str(e)}'}), 500

    @bp.route('/api/shortcuts', methods=['GET'])
    def get_shortcuts():
        return jsonify(config.SEARCH_SHORTCUTS)

    @bp.route('/api/samples', methods=['GET'])
    def get_samples():
        return jsonify([{'path': path, 'name': os.path.basename(path)} for path in config.IMAGE_SAMPLES])

    @bp.route('/api/register_images', methods=['POST'])
    def register_images_api():
        data = request.json
        image_dir = data.get('image_dir', '')
        category = data.get('category', 'person')

        if not image_dir or not os.path.exists(image_dir):
            return jsonify({'error': f'Invalid image directory: {image_dir}'}), 400

        try:
            def register_in_background():
                try:
                    config.image_search.register_new_images(
                        image_dir=image_dir,
                        category=category,
                        check_exist=True
                    )
                    print(f"Registration complete. Total records: {config.image_search.count_records()}")
                except Exception as e:
                    print(f"Error in background registration: {e}")

            thread = threading.Thread(target=register_in_background, daemon=True)
            thread.start()

            return jsonify({
                'success': True,
                'message': f'Started registering images from {image_dir}',
                'current_count': config.image_search.count_records()
            })
        except Exception as e:
            print(f"Error starting registration: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/api/image_search', methods=['POST'])
    def image_search_api():
        start_time = time.time()
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        top_k = int(request.form.get('top_k', 10))
        threshold = float(request.form.get('threshold', 0.5))
        person_threshold = float(request.form.get('person_threshold', 0.6))
        collections = request.form.getlist('collections[]') or config.selected_collections

        try:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(config.UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
            file.save(upload_path)

            try:
                input_image = Image.open(upload_path)
                if input_image.mode != 'RGB':
                    input_image = input_image.convert('RGB')
            except Exception as e:
                print(f"Error loading image: {e}")
                return jsonify({'error': f'无法加载图像: {str(e)}'}), 400

            detection_start = time.time()
            detections = config.pedestrianDetector.detect_image(
                image_pil=input_image, threshold=person_threshold, output_path=None)
            detection_time = time.time() - detection_start

            if not detections:
                return jsonify({'error': 'No persons detected in the image'}), 400

            detection_image = input_image.copy()
            draw = ImageDraw.Draw(detection_image)
            person_colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255)
            ]
            valid_detections = []
            for i, (x1, y1, x2, y2, score) in enumerate(detections):
                valid_detections.append((x1, y1, x2, y2, score))
                color = person_colors[i % len(person_colors)]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                person_id = f"#{i+1}"
                draw.text((x1+5, y1-20), person_id, fill=color, stroke_width=1)

            detection_image_path = os.path.join(
                config.CACHE_DIR, f"detection_{uuid.uuid4()}_{os.path.basename(upload_path)}")
            detection_image.save(detection_image_path)

            all_results = []
            total_search_time = 0
            all_results.append({
                'type': 'detection',
                'path': os.path.basename(detection_image_path),
                'label': '检测结果'
            })

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(valid_detections), 5)) as executor:
                futures = []
                for idx, (x1, y1, x2, y2, score) in enumerate(valid_detections):
                    person_crop = input_image.crop((x1, y1, x2, y2))
                    crop_path = os.path.join(
                        config.TEMP_DIR, f"person_{idx+1}_{uuid.uuid4()}.jpg")
                    person_crop.save(crop_path)
                    color = person_colors[idx % len(person_colors)]
                    all_results.append({
                        'type': 'person',
                        'path': os.path.basename(crop_path),
                        'label': f'行人 #{idx+1} - 置信度: {score:.2f}',
                        'person_id': idx + 1,
                        'color': color,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(score)
                    })
                    if collections:
                        future = executor.submit(
                            config.image_search.search_in_collections,
                            person_crop, top_k, threshold, collections, True)
                    else:
                        future = executor.submit(
                            config.image_search.search_by_image,
                            person_crop, top_k, threshold, True)
                    futures.append((future, idx))

                for future, idx in futures:
                    try:
                        results, search_time = future.result()
                        total_search_time += search_time
                        result_futures = []
                        for result_idx, result in enumerate(results):
                            if len(result) == 3:
                                img_path, score, payload = result
                            else:
                                img_path, score = result
                                payload = None
                            if not os.path.exists(img_path):
                                continue
                            bbox = None
                            if payload and isinstance(payload, dict) and 'bbox' in payload:
                                bbox = payload['bbox']
                            else:
                                try:
                                    db_payload = config.image_search.get_payload_by_path(img_path)
                                    if db_payload and isinstance(db_payload, dict) and 'bbox' in db_payload:
                                        bbox = db_payload['bbox']
                                except Exception as e:
                                    print(f"Error getting bbox from database for {img_path}: {e}")
                            preview_path = generate_preview(config, img_path, bbox)
                            if not preview_path:
                                preview_path = img_path
                            preview_filename = os.path.basename(preview_path)
                            original_filename = os.path.basename(img_path)
                            result_future = executor.submit(process_image, config, img_path, bbox)
                            result_futures.append((result_future, img_path))
                            all_results.append({
                                'type': 'result',
                                'path': preview_filename,
                                'original_path': original_filename,
                                'display_path': original_filename,
                                'label': f'#{idx+1} | {original_filename} : {score:.3f}',
                                'score': float(score),
                                'person_id': idx + 1,
                                'bbox': bbox
                            })
                        for result_future, path in result_futures:
                            try:
                                result_future.result(timeout=0.1)
                            except concurrent.futures.TimeoutError:
                                pass
                            except Exception as e:
                                print(f"处理图片出错 {path}: {e}")
                    except Exception as e:
                        print(f"Error in search for person #{idx+1}: {e}")
                        all_results.append({
                            'type': 'error',
                            'label': f'行人 #{idx+1} - 搜索出错',
                            'error': str(e)
                        })

            total_time = time.time() - start_time
            return jsonify({
                'results': all_results,
                'detection_time': detection_time,
                'search_time': total_search_time,
                'total_time': total_time,
                'person_count': len(valid_detections),
                'record_count': update_record_count(config)
            })
        except Exception as e:
            print(f"Error in image search: {e}")
            return jsonify({
                'error': str(e),
                'results': [],
                'record_count': update_record_count(config)
            }), 500
    
    return bp

