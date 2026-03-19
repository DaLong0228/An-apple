# 2025-11-13
import uuid  # 用于生成唯一的任务ID和文件名，防止文件重名
import time  # 用于控制视频处理时的暂停延迟（time.sleep）和计算预览图生成时间
import threading  # 用于开启后台线程处理视频，避免阻塞主Web服务；以及提供线程锁(Lock)
from pathlib import Path  # 现代的路径操作库，比传统的 os.path 更好用
from datetime import datetime  # 用于生成历史记录和日志的时间戳

import cv2        # OpenCV库，用于读取、处理和保存图像与视频帧
from flask import Flask, render_template, request, send_from_directory, jsonify # Flask Web框架核心组件
from werkzeug.utils import secure_filename   # 过滤文件名中的危险字符，防止目录遍历攻击

from src.detect import BadmintonDetector  # 导入你自定义的羽毛球检测模型类
from src.utils import load_config     # 导入自定义的配置文件加载函数
# 使用 pathlib 定义各个文件夹的绝对路径
BASE_DIR = Path(__file__).parent  # 当前文件所在的根目录
UPLOAD_DIR = BASE_DIR / "runs" / "uploads"  # 用户上传文件的存放处
IMAGE_OUT_DIR = BASE_DIR / "runs" / "images"  # 图片处理结果存放处
VIDEO_OUT_DIR = BASE_DIR / "runs" / "detect"  # 视频处理结果存放处
PREVIEW_DIR = BASE_DIR / "runs" / "preview"  # 视频处理过程中的实时预览图存放处

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}  # 允许上传的图片格式
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}   # 允许上传的视频格式

app = Flask(__name__)  # 初始化 Flask 应用
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  #限制上传文件最大为 1 GB

jobs = {}  # 内存字典，用于存储当前所有的视频处理任务状态
jobs_lock = threading.Lock()  # 任务字典的线程锁，防止多个请求同时修改 jobs 导致数据混乱
detector_lock = threading.Lock()  # AI模型的线程锁。深度学习推理（尤其是GPU）通常不支持多线程并发，这保证同一时刻只有一个图像/帧在被检测

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)   # 启动时确保预览图文件夹存在

#这部分负责在服务器启动时把深度学习模型加载到内存中
def init_detector():   # 默认的模型权重文件路径
    weights_path = BASE_DIR / "best.pt"  # 如果当前目录没有，去 models 文件夹找
    if not weights_path.exists(): 
