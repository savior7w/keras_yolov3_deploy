#服务端新加入的包
import flask
import io
import json
#import cv2
import base64


import colorsys
#import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from flask import request
from keras.utils import multi_gpu_model


app = flask.Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_path='logs/000/trained_weights_final.h5'
# anchors_path='model_data/yolo_anchors.txt'
# classes_path= 'model_data/voc_classes.txt'
score = 0.1
iou = 0.35
model_image_size= (416, 416)
gpu_num = 1
#获得类别名称为list类型
def get_class():
    classes_path = 'voc_classes.txt'
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

#获得9个anchors的大小  类型为ndarray的二维数组
def get_anchors():
    anchors_path = 'yolo_anchors.txt'
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_imageBase64String(imageFilePath):
    if not os.path.exists(imageFilePath):
        image_base64_string = ''
    else:
        with open(imageFilePath, 'rb') as file:
            image_bytes = file.read()
        image_base64_bytes = base64.b64encode(image_bytes)
        image_base64_string = image_base64_bytes.decode('utf-8')
    return image_base64_string

#加载yolov3模型
def load_yolo_model():
    model_path = 'logs/000/trained_weights_final.h5'
    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5') , 'Keras model or weights must be a .h5 file.'

    global anchors
    global class_names
    global yolo_model

    anchors = get_anchors()
    class_names = get_class()

    num_anchors = len(anchors)
    num_classes = len(class_names)

    K.clear_session()
    yolo_model=yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
    yolo_model.load_weights(model_path)
    print('{} model, anchors, and classes loaded.'.format(model_path))

    global boxes
    global scores
    global classes
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors,
                                       len(class_names), model_image_size,
                                       score_threshold=score, iou_threshold=iou)

    #return boxes, scores, classes

def prepare_image(path):
    image = Image.open(path)
    #检查图片能否被32整除,不能整除则删除多余部分
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    image_data /= 255.
    print('1111111111111111111111111111111111111111111111111111111')
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    print(image_data.shape)
    return image_data


def predicted_result(image):
    #K.clear_session()
    out_boxes, out_scores, out_classes = K.get_session().run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image,
            #model_image_size: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })


    #K.get_session().close()

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    results = []

    for i, c in reversed(list(enumerate(out_classes))):
        print('333333333333333333333333333')
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        # image = image.convert('RGB')
        # draw = ImageDraw.Draw(image)
        # label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))

        #数据调试写死 图像大小
        bottom = min(416, np.floor(bottom + 0.5).astype('int32'))
        right = min(416, np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))


        # detect_result = {'class': predicted_class, 'scores':np.asscalar(out_scores[i]), 'x': np.asscalar(left), 'y': np.asscalar(top), 'h': np.asscalar(right), 'w': np.asscalar(bottom)}
        detect_result = {'label': label, 'x': left.item(),
                         'y': top.item(), 'h': right.item(), 'w': bottom.item()}
        # detect_result = {'label': label}
        results.append(detect_result)
        # data_json = json.dumps(results, sort_keys=True, indent=4, separators=(',', ': '))
    text = str(results)
    return text





@app.route('/predict', methods=['GET', 'POST'])  # 使用methods参数处理不同HTTP方法
def home():
    if request.method == 'POST':
        baseimg = request.form['path']

        img = base64.b64decode(baseimg)
        filepath = "image_db/test.jpg"
        file = open(filepath, 'wb')
        file.write(img)
        file.close()

        image = prepare_image(filepath)
        res = predicted_result(image)

        return res

#flask服务代码
if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_yolo_model()

    app.run(host='0.0.0.0', port= 5000)


