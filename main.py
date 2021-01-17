import cv2
import sys
from keras.models import load_model
import numpy as np
from keras.applications import resnet50
import face_detection

# pip install -r requirements.txt
# sudo python main.py haarcascade_frontalface_default.xml

# may need to try running in different terminals to trigger camera permissions

def main():
    cascPath = sys.argv[1]
    faceCascade = cv2.CascadeClassifier(cascPath)
    modelPath = sys.argv[2]
    mask_classifier = load_model(modelPath)
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

    video_capture = cv2.VideoCapture(0)
    FILE_PATH = "/Users/milly./Desktop/mask.jpg"

    img = cv2.imread(FILE_PATH)

    while True:
        # Capture frame-by-frame
        # ret, frame = video_capture.read()
        frame = img
    
        masked_faces = []
        unmasked_faces = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector.detect(img[:,:,::-1])
        
        print(faces)
        
        if len(faces) > 0 and faces.shape[0] > 0:
            for i in range(faces.shape[0]):
                # Get Co-ordinates
                x1 = int(faces[i][0])
                x2 = int(faces[i][2])
                y1 = int(faces[i][1])
                y2 = int(faces[i][3])


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


        # # Draw a rectangle around the faces
        # for (x, y, w, h, _) in faces:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
