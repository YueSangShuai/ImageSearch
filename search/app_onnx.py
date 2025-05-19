#!/usr/bin/env python3

# app_onnx.py
# 组装最后的服务程序

import os
from flask import Flask
from flask_socketio import SocketIO, emit
import argparse

from det import PedestrianHeadDetector
from api import ImageTextSearch
from model_text import TextFeatureExtractor # 此处使用 model_text 模块中的文本特征提取器，ONNX 模型
from processor_app import create_image_search_blueprint, Config

parser = argparse.ArgumentParser("Flask API for image search")

Config.add_arguments(parser)

TextFeatureExtractor.add_arguments(parser)
ImageTextSearch.add_arguments(parser)
PedestrianHeadDetector.add_arguments(parser)

args = parser.parse_args()

text_extractor = TextFeatureExtractor(**vars(args))
image_search = ImageTextSearch(text_extractor=text_extractor, **vars(args))
pedestrianDetector = PedestrianHeadDetector(**vars(args))

config = Config(text_extractor=text_extractor, image_search=image_search, pedestrianDetector=pedestrianDetector, **vars(args))

# Flask app 和 socketio 依然全局
app = Flask(__name__, static_folder='static', template_folder='templates')
# 注册 blueprint
app.register_blueprint(create_image_search_blueprint(config))

socketio = SocketIO(app)
config.socketio = socketio
@socketio.on('connect')
def handle_connect():
    print("Client connected")

if __name__ == '__main__':
    print(f"Starting Flask server on port {config.port}")
    socketio.run(app, host='0.0.0.0', port=config.port, debug=False)