import requests
import os


# 主函数
if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    while True:
        input_content = input('输入图片路径，输入-1退出 ')
        if input_content.strip() == "":
            input_content = r'D:\py\facenet-master\data\face_test_pic\222.jpg'
        if input_content.strip() == "-1":
            break
        elif not os.path.exists(input_content.strip()):
            print('输入图片路径不正确，请重新输入')
        else:
            input_content = input_content.strip()

        postdata = {"path": input_content}
        result = requests.post(url, data=postdata)

        print(result.text)