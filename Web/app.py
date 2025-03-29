from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io
import os
import action

app = Flask(__name__)

# 创建一个目录用于保存上传的图片
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

file_cnt = 0

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

    # 保存图片到服务器文件系统
    save_path = os.path.join(UPLOAD_FOLDER, f'{file_cnt}_{file.filename}')
    img.save(save_path, format="PNG")

    return jsonify({"message": "Image processed successfully", "size": list(img.size)})

@app.route('/images')
def show_images():
    # 获取 uploads 文件夹中的所有图片
    images = os.listdir(UPLOAD_FOLDER)
    # 过滤掉非图片文件（可选）
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return render_template("images.html", images=images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # 提供对上传图片的访问
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/information')
def return_information():


if __name__ == '__main__':
    app.run(debug=True)