from urllib import response

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
import os

import librosa
import librosa.display
import tensorflow as tf
import numpy as np
import wave
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Audio





app = Flask(__name__)


def diagnosis(voice_input):
    gru_model = tf.keras.models.load_model('saved_model/respiratory.h5')
    classes = ["COPD", "Bronchiolitis ", "Pneumoina", "URTI", "Healthy"]
    data_x, sampling_rate = librosa.load(voice_input)
    features = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52).T,axis = 0)
    features = features.reshape(1,52)
    test_pred = gru_model.predict(np.expand_dims(features, axis = 1))
    result_prediction = classes[np.argmax(test_pred[0], axis=1)[0]]
    confidence_percentage = test_pred.T[test_pred[0].mean(axis=0).argmax()].mean()
    confidence_percentage = round(confidence_percentage * 100, 2)
    print (result_prediction , confidence_percentage)

    return result_prediction, confidence_percentage


@app.route('/')
def hello_world():
    return 'Page not found!'

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['wav'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# 添加路由
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        try:
            # 解析请求头中的文件名和文件流
            # 通过file标签获取文件
            file = request.files['file']
            if not (file and allowed_file(file.filename)):
                response = {
                    'status': 501,
                    'message': 'please upload file with.wav format'
                }
                return jsonify(response)
                # return jsonify({"error": 1001, "msg": "图片类型：png、PNG、jpg、JPG、bmp"})
            # 当前文件所在路径
            basepath = os.path.dirname(__file__)
            # 一定要先创建该文件夹，不然会提示没有该路径
            upload_path = os.path.join(basepath, 'uploadFiles', secure_filename(file.filename))
            # 保存文件
            file.save(upload_path)

            result_prediction, confidence_percentage = diagnosis(upload_path)
            print(result_prediction, confidence_percentage)
            print("hello , are you ok!")
            print()
            # 返回上传成功界面
            response = {
                'status': 200,
                'message': 'upload success',
                'result_prediction': result_prediction,
                'confidence_percentage': confidence_percentage

            }
            return jsonify(response)
        except Exception as e:
            print(e)
            print("hello , error")
            response = {
                'status': 500,
                'message': 'upload failed'
            }
            # 重新返回上传界面
            return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
    # app.run(port=8080)
