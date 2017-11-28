# -*- coding: utf-8 -*-
import face_recognition
import cv2
from read_data import read_name_list
from read_data import read_file
video_capture = cv2.VideoCapture(0)

#读取摄像头，并识别摄像头中的人脸，进行匹配。

all_encoding, lable_list, counter = read_file('./dataset')
name_list = read_name_list('./dataset')
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if process_this_frame:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        #匹配，并赋值
        for face_encoding in face_encodings:
            i = 0
            j = 0
            for t in all_encoding:
                for k in t:
                    match = face_recognition.compare_faces([k], face_encoding)
                    if match[0]:
                        name = name_list[i]
                        j=1
                i = i+1
            if j == 0:
                name = "unknown"

            face_names.append(name)
                

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),  2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
