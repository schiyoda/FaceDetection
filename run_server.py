from flask import Flask, jsonify, request
import os
import cv2
import urllib.request
import numpy as np

app = Flask(__name__)

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
genderList = ['Male', 'Female']

def load_model():
    global ageNet, genderNet, faceNet
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

def getFaceBox(net, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])

    return bboxes

@app.route("/detectFace", methods=["POST"])
def detectFace():
    if request.method == 'POST':
        bboxes = []
        if(request.files):
            img_file = request.files['img_file']
            file_data = img_file.read()
            nparr = np.fromstring(file_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            bboxes = getFaceBox(faceNet, img)

        elif(request.form):
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0",
            }
            req = urllib.request.Request(url=request.form['url'], headers=headers)
            resp = urllib.request.urlopen(req)
            nparr = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            bboxes = getFaceBox(faceNet, img)

        faces = []
        padding = 20
        for bbox in bboxes:
            face = img[max(0,bbox[1]-padding):min(bbox[3]+padding,img.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, img.shape[1]-1)]
            face_location = {"left":bbox[0], "top":bbox[1], "width":bbox[2]-bbox[0], "height":bbox[3]-bbox[1]}
            
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            gender = {"gender":gender, "confidence":"{:.3f}".format(genderPreds[0].max())}

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            age = {"min":"{}".format(age.split("-")[0]), "max":"{}".format(age.split("-")[1]), "confidence":"{:.3f}".format(agePreds[0].max())}

            faces.append({"face_location":face_location, "gender":gender, "age":age})

    return jsonify({"faces":faces})

port = os.getenv('PORT', '5000')
if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run(host='0.0.0.0', port=int(port))