import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import tempfile


font = cv2.FONT_HERSHEY_SIMPLEX 
FRAME_WINDOW = st.image([])
stframe = st.empty()  



faceProto = "model/opencv_face_detector.pbtxt"
faceModel = "model/opencv_face_detector_uint8.pb"

ageProto = "model/age_deploy.prototxt"
ageModel = "model/age_net.caffemodel"

genderProto = "model/gender_deploy.prototxt"
genderModel = "model/gender_net.caffemodel"


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

stframe = st.empty()

def faceBox(net, frame, conf_threshold=0.7):
    frameDnn = frame.copy()
    frameHeight = frameDnn.shape[0]
    frameWidth = frameDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameDnn, 1.0, (300, 300), [104, 117, 123], True, False)

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
            cv2.rectangle(frameDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frameDnn, bboxes




class VideoTransformer(VideoTransformerBase):
    def transform(self,frame):
        img = frame.to_ndarray(format="bgr24")
        
        frame,bboxs=faceBox(faceNet,img)
        
        
        
        padding=20
        for bbox in bboxs:
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

            #predict gender
            genderNet.setInput(blob)
            genderPred=genderNet.forward()
            gender=genderList[genderPred[0].argmax()]


            #predict age
            ageNet.setInput(blob)
            agePred=ageNet.forward()
            age=ageList[agePred[0].argmax()]

            #show outpout
            label="{},{}".format(gender,age)
            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        return frame






def main():
    activities = ["Images" ,"Videos","LiveWebcam","About"]
    choice = st.sidebar.selectbox("Select Activity",activities)

    if choice=="Images":
        st.title("Age and Gender Detection Model")
        image_file=st.file_uploader("Upload Image",type=['jpg','jpeg','png'])
        if image_file is not None:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img_i=cv2.imdecode(file_bytes,1)
            
            
          
            

            st.text("Original Image")
            progress=st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            st.image(image_file)
        else:
            st.error("Image File not Uploaded")

        if st.button("predict"):
            frameFace, bboxes = faceBox(faceNet, img_i)
            for bbox in bboxes:
                face = img_i[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]

                label = "{},{}".format(gender, age)
                cv2.rectangle(frameFace, (bbox[0], bbox[1]-30), (bbox[2]+70, bbox[1]), (0, 255, 0), -1)
                cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)

                frame=cv2.cvtColor(frameFace,cv2.COLOR_BGR2RGB)
            st.image(frame)
    

    elif choice=="Videos":
        st.title("Age and Gender Detection Model")
        video_file = st.file_uploader("Upload Videos",type=['mp4','mpeg','avi'])
        if video_file is not None:
            tfile=tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video=cv2.VideoCapture(tfile.name)

            padding=20

            while True:
                if video.isOpened():
                    ret,frame=video.read()
                    if ret:
                        frame,bboxs=faceBox(faceNet,frame)
                        for bbox in bboxs:
                            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

                            #predict gender
                            genderNet.setInput(blob)
                            genderPred=genderNet.forward()
                            gender=genderList[genderPred[0].argmax()]

                            #predict age
                            ageNet.setInput(blob)
                            agePred=ageNet.forward()
                            age=ageList[agePred[0].argmax()]

                            #show outpout
                            label="{},{}".format(gender,age)
                            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
                            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
                            stframe.image(frame)
                    if not ret:
                        break
                else:
                    st.error("Invalid Format !")
    


    elif choice=="LiveWebcam":
        
        webrtc_streamer(key="WYH",rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),media_stream_constraints={"video": True, "audio": False},video_processor_factory=VideoTransformer)
    
    
    elif choice=="About":
        st.title("Facial Emotion Recognition System")
        st.subheader("Emotion Detection Model Using Streamlit & Python")
        st.markdown("This Application Developed by <a href='https://github.com/datamind321'>DataMind Platform 2.0</a>",unsafe_allow_html=True) 
        
        # st.markdown("If You have any queries , Contact Us On : ") 
        st.header("contact us on : bme19rahul.r@invertisuniversity.ac.in")

        st.divider()
        
        st.markdown("[![Linkedin](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/rahul-rathour-402408231/)")
        
        st.markdown("[![Instagram](https://img.icons8.com/color/1x/instagram-new.png)](https://instagram.com/_technical__mind?igshid=YmMyMTA2M2Y=)") 
        
                
           

    
          







main()