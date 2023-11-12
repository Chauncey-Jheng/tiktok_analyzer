from flask import Flask
from flask import redirect
from flask import url_for
from flask import render_template
from flask import request
import click
import json
import socket

app = Flask(__name__)       

@app.route('/', methods=["GET","POST"])
def index():
    if request.method == "POST":
        data = request.json
        liveURL = data[0]["liveURL"]
        liveName = data[0]["liveName"]
        sensitiveWord = data[0]["sensitiveWord"]
        print(liveURL)
        print(liveName)
        print(sensitiveWord)
        data = {"liveURL":liveURL, "liveName":liveName, "sensitiveWord":sensitiveWord}
        phone = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        phone.connect(("127.0.0.1", 8086))
        msg = json.dumps(data)
        phone.send(msg.encode("utf-8"))
        phone.close()
        ret_msg = {"status":"400","msg":"监测对象添加成功！直播分析器正在启动中，请耐心等待......"}
        return json.dumps(ret_msg)
    else:    
        return render_template('index.html')

@app.route("/job", methods=['GET'])
def status():
    try:
        phone = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        phone.connect(("127.0.0.1", 8086))
        data = "give_me_live_info"
        msg = data.encode("utf-8")
        phone.send(msg)
        data = phone.recv(1024)
        msg = data.decode("utf-8")
        print("服务端返回的数据：", msg)
        msg = json.loads(msg)
        live_video_path = msg["video_file_path"] 
        ocr_file_path = msg["ocr_file_path"]
        asr_file_path = msg["asr_file_path"]
        phone.close()
    except:
        return b"down"
    # live_video_path = "static/video/sample.mp4"
    asr_result = ""
    ocr_result = ""
    with open(asr_file_path, "rt") as f:
        asr_result = f.read()
    with open(ocr_file_path, "rt") as f:
        ocr_result = f.read()
    ret_msg = {"live_video_path":live_video_path, "asr_result":asr_result, "ocr_result":ocr_result}
    print(ret_msg)
    return json.dumps(ret_msg)

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=3100, debug=True)