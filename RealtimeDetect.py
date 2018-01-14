import numpy as np
import cv2
import SmileDetect
def detectFace(img):

    face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray= cv2.GaussianBlur(gray, (5,5), 0, 0)
    facespos = face_haar.detectMultiScale(gray, 1.2, 4)
    faces=[]
    pos=[]
    if len(facespos)!=0:
        for x,y,w,h in facespos:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            faces.append(gray[y:y+h,x:x+w].copy())
            pos.append((x,y+h,y,x+w))
        return True,faces,img,pos
    else:
        return False,None,img,None
def detectPhoto(filename):
    print("Loading model data...")
    model = cv2.ml.SVM_load("mymodel.txt")
    cv2.namedWindow("SmileDetect")

    frame = cv2.imread(filename)
    print(frame.shape)
    ret, faces, frame, pos = detectFace(frame)

    if ret:

        data = []
        cv2.putText(frame, "{} faces detected".format(len(faces)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        for face in faces:
            data.append(SmileDetect.getLBP(face, 25, 5))
        data = SmileDetect.normalize(data)
        ret, res = model.predict(data)
        for i in range(len(res)):
            if res[i] == True:
                cv2.putText(frame, "Smiling", pos[i][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Smiling", pos[i][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No face detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("SmileDetect",frame)
    #cv2.imwrite("ha.jpg",frame)
    key=cv2.waitKey()


def main():
    print("Press Q to quit")
    print("Loading model data...")
    model=cv2.ml.SVM_load("mymodel.txt")
    print("Initiating camera...")
    cam=cv2.VideoCapture(0)
    cv2.namedWindow("SmileDetect")
    icon1=cv2.imread("smile.jpg")
    icon0=cv2.imread("nonsmile.jpg")
    icon1=cv2.resize(icon1,(50,50))
    icon0 = cv2.resize(icon0, (50, 50))
    c=[5]*20
    data=[]

    while True:
        ret,frame=cam.read()

        ret, faces, frame, pos = detectFace(frame)
        if ret:
            data=[]
            cv2.putText(frame, "{} faces detected".format(len(faces)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 0), 2)
            for face in faces:
                data.append(SmileDetect.getLBP(face,25,5,None))
            data=SmileDetect.normalize(data)

            if len(data)!=0:
                ret,res=model.predict(data)
            else:
                continue
            for i in range(len(res)):
                if res[i]==True:
                    cv2.putText(frame, "Smiling", pos[i][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frame[pos[i][2]:pos[i][2]+50,pos[i][0]:pos[i][0]+50]=icon1[:,:]
                    c[i]=0

                elif c[i]>=5:
                    cv2.putText(frame, "Not Smiling", pos[i][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frame[pos[i][2]:pos[i][2] + 50, pos[i][0]:pos[i][0] + 50] = icon0[:, :]
                else:
                    c[i]+=1
                    cv2.putText(frame, "Smiling", pos[i][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frame[pos[i][2]:pos[i][2] + 50, pos[i][0]:pos[i][0] + 50] = icon1[:, :]
        else:
            #cv2.rectangle(frame, (pos[0][0], pos[0][2]), (pos[0][3], pos[0][1]), (255, 255, 0), 2)
            cv2.putText(frame, "No face detected", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("SmileDetect", frame)
        key=cv2.waitKey(2)
        if key==ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()

main()