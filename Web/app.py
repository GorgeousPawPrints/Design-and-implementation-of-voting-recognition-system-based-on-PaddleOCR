from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io
import os
import csv
from action.main import action, save_csv, clear_folder
from threading import Lock, Thread
import time

app = Flask(__name__)

# 创建上传目录
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

file_cnt = 0

# 任务状态存储
task_status = {}
task_lock = Lock()

CSV_FILE_PATH = os.path.join(UPLOAD_FOLDER, "information.csv")


@app.route('/upload', methods=['GET'])
def upload_in():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    global file_cnt
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((200, 200))
    file_cnt += 1

    save_path = os.path.join(UPLOAD_FOLDER, f'{file_cnt}_{file.filename}')
    img.save(save_path, format="PNG")

    return jsonify({"message": "Image processed successfully", "size": list(img.size)})


@app.route('/images')
def show_images():
    images = [img for img in os.listdir(UPLOAD_FOLDER)
              if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return render_template("images.html", images=images)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def read_csv_file(CSV_FILE_PATH):
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # 获取表头
            return [dict(zip(headers, row)) for row in reader], None
    except Exception as e:
        return None, str(e)


@app.route('/information')
def show_information_page():
    """显示数据展示页面"""
    return render_template("information.html")


@app.route('/start-processing')
def start_processing():
    task_id = str(len(task_status) + 1)
    task_status[task_id] = {
        'status': 'processing',
        'progress': 0,
        'data': None,
        'error': None
    }

    def background_task():
        try:
            raw_data = action()  # 调用 action 获取 OCR 结果
            save_csv(raw_data)  # 使用 action 提供的 save_csv 方法保存到 CSV

            # 从 CSV 文件中读取数据
            data, error = read_csv_file(CSV_FILE_PATH)
            if error:
                raise Exception(error)

            with task_lock:
                task_status[task_id]['status'] = 'completed'
                task_status[task_id]['data'] = data  # 使用从 CSV 中读取的数据
        except Exception as e:
            with task_lock:
                task_status[task_id]['status'] = 'error'
                task_status[task_id]['error'] = str(e)
        finally:
            clear_folder("./uploads")

    # 启动后台任务
    Thread(target=background_task).start()

    return jsonify({"task_id": task_id})


@app.route('/task-status/<task_id>')
def get_task_status(task_id):
    """获取任务状态"""
    with task_lock:
        status = task_status.get(task_id, {
            'status': 'not_found',
            'progress': 0,
            'error': '任务不存在'
        })
    return jsonify(status)


if __name__ == '__main__':
    app.run(debug=True)