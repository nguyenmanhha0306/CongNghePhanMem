from flask import  Response
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

from index import server
import dash_html_components as html
import dash_core_components as dcc

face_classifier = cv2.CascadeClassifier(r'model/haarcascade_frontalface_default.xml')
classifier =tf.keras.models.load_model('model/mdfinal.h5')
class_labels = ['gian','So','vui','binh thuong','buon','ngac nhien']
labels = []
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        gray_img= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_img, 1.32, 5)
        for (x, y, w, h) in faces:
    # vẽ 1 đường bao quanh khuôn mặt
         cv2.rectangle(image,(x,y),(x+w,y+h),(50,205,50),2)
         cv2.rectangle(image, (x,y-40), (x+w,y), (50,205,50), -1)
         #Tách phần khuôn mặt vừa tìm được và resize về kích thước 48x48 để chuẩn bị đưa vào bộ mạng Neural Network
         roi_gray = gray_img[y:y+h,x:x+w]
         roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
         if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)  

            # Thực hiện dự đoán cảm xúc
            
            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x+20,y-10)  
            cv2.putText(image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

         else:
             cv2.putText(image,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),3)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
page_2_layout = html.Div([
    html.H1('NHẬN DIỆN CẢM XÚC GƯƠNG MẶT NGƯỜI BẰNG CAMERA',style={'textAlign':'center'}),
    html.Img(src ='/video_feed',style={"paddingLeft":380}),
])