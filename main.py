import cv2
import sys
import numpy as np
from keras.applications import resnet50
from keras.models import load_model

# pip install -r requirements.txt
# sudo python main.py haarcascade_frontalface_default.xml

# may need to try running in different terminals to trigger camera permissions

def main():
    cascPath = sys.argv[1]
    modelPath = sys.argv[2]
    faceCascade = cv2.CascadeClassifier(cascPath)
    mask_classifier = load_model(modelPath)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        masked_faces = []
        unmasked_faces = []

        # Detect Faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print(detections.shape)

        if detections.shape[0] > 0:
            for i in range(detections.shape[0]):
                # Get Co-ordinates
                x, y, w, h = detections[i]
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h


                # Predict Output
                face_arr = cv2.resize(frame[y1:y2,x1:x2,::-1], (224, 224), interpolation=cv2.INTER_NEAREST)
                face_arr = np.expand_dims(face_arr, axis=0)
                face_arr = resnet50.preprocess_input(face_arr)
                match = mask_classifier.predict(face_arr)

                if match[0][0]<0.5:
                    masked_faces.append([x1,y1,x2,y2])
                else:
                    unmasked_faces.append([x1,y1,x2,y2])

        # Put Bounding Box on the Faces (Green:Masked,Red:Not-Masked)
        for f in range(len(masked_faces)):
            a,b,c,d = masked_faces[f]
            cv2.rectangle(frame, (a,b), (c,d), (0,255,0), 2)

        for f in range(len(unmasked_faces)):
            a,b,c,d = unmasked_faces[f]
            cv2.rectangle(frame, (a,b), (c,d), (0,0,255), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
