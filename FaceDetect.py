import cv2
def detect(filename):

    img = cv2.imread(filename)
    # 加载分类器
    face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # 把图像转为黑白图像
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray= cv2.GaussianBlur(gray, (5,5), 0, 0) #高斯滤波
    # 检测图像中的所有脸
    faces = face_haar.detectMultiScale(gray, 1.1, 4)
    try:
        x, y, w, h =faces[-1]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    except:
        print("error")
        cv2.imwrite("err\\{}".format(filename[6:]),gray)
        return None
    face=gray[y:y+h,x:x+w]
    return face
    #cv2.imwrite(savepos, face)
    #cv2.namedWindow('img')
    #cv2.imshow('img',gray)
    #cv2.waitKey(0)
    #cv2.imshow('img', face)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
def main():
    for i in range(1, 4001):
        filename="files\\file{:0>4}.jpg".format(str(i))
        img = cv2.imread(filename)
        face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0, 0)
        faces = face_haar.detectMultiScale(gray, 1.1, 4)
        try:
            x,y,w,h=-1,-1,-1,-1
            for tx, ty, tw, th in faces:
                if tw>w and th>h:
                    x,y,w,h=tx,ty,tw,th
            face = gray[y:y + h, x:x + w]
            if x!=-1:
                cv2.imwrite("Faces\\{}.jpg".format(str(i)), face)
            else:
                cv2.imwrite("err\\{}".format(filename[6:]), gray)
        except:
            print("error")
            cv2.imwrite("err\\{}".format(filename[6:]), gray)
        print(100 * i / 4000, '%')


