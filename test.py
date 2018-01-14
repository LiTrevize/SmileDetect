import numpy as np
import cv2
def image():
    img = cv2.imread("file0001.jpg")
    cv2.namedWindow("Image")
    #cv2.imshow("Image", img)
    #cv2.waitKey()
    cv2.imwrite("abc.jpg",img)
    #cv2.split的速度比直接索引要慢,但cv2.split返回的是拷贝,直接索引返回的是引用(改变B就会改变BGRImg)
    B=img[:,:,0]
    G= img[:, :, 1]
    R= img[:, :, 2]
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(B==0)
    #data[data < 0] = 0
    cv2.imshow("Image",grey)
    cv2.waitKey()
def array():
    print(np.arange(1, 10).reshape(3, 3))
    np.linspace(1, 10, 20)
    print(np.zeros((3, 4)))
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    print(matrix)
    print(matrix[:, 1])
    print(np.zeros((10,10,3),np.uint))
def minimum(s="00110100"):
    ss=[]
    for i in range(len(s)):
        ss.append(s[i:]+s[0:i])
    print(ss)
    return min([int(num,2) for num in ss])

def normalize(data):
    maxm=data.max()
    minm=data.min()
    ave=0.5*(maxm+minm)
    det=0.5*(maxm-minm)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j]=(data[i,j]-ave)/det
    return data

a=np.array([[1,2,3],[2,3,4],[5,3,5]],dtype='float32')


def invariant(lbp0):
    lbps=[]
    for i in range(len(lbp0)):
        lbps.append(lbp0[i:]+lbp0[0:i])
    return min([int(num,2) for num in lbps])
def get_variantNum():
    variantNums = []
    for i in range(256):
        if invariant("{:0>8}".format(str(bin(i))[2:]))!=i:
            variantNums.append(i)
    return variantNums
def equalMode(lbp0):
    count=0
    for i in range(len(lbp0)-1):
        if lbp0[i]!=lbp0[i+1]:
            count+=1
    if count>2:
        return "00000101"

def get_equalMode():
    equalNums = []
    for i in range(256):
        if equalMode("{:0>8}".format(str(bin(i))[2:]))=="00000101":
            equalNums.append(i)
    return equalNums
def dist2(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2
def getLBP(img,imgsize=100,blocksize=10):
    if len(img.shape)==3:
        img=img[:,:,0]

    size=min(img.shape[0],img.shape[1])
    img=img[:size,:size]
    img=cv2.resize(img,(imgsize,imgsize))
    lbps=np.zeros((imgsize,imgsize))
    cv2.namedWindow("img")
    cv2.imshow("img",img)
    cv2.waitKey()
    for i in range(1,imgsize-1):
        for j in range(1,imgsize-1):
            if True:#dist2((i,j),(imgsize/2,imgsize/2))<=imgsize**2/4:
                cell=img[i-1:i+2,j-1:j+2].copy()
                val=cell[1,1]
                cell[cell<val]=0
                cell[cell!=0]=1
                lbp=str(cell[0,0])+str(cell[0,1])+str(cell[0,2])+ \
                    str(cell[1,2]) +str(cell[2,2])+str(cell[2,1])+ \
                    str(cell[2, 0]) +str(cell[1,0])
                lbps[i, j] = int(lbp,2)

    print(img)
    cv2.namedWindow("img")
    cv2.imshow("img",lbps)
    cv2.waitKey()
    if blocksize==0:
        sets=[]
        for i in range(1, imgsize - 1):
            for j in range(1, imgsize - 1):
                sets.append(lbps[i,j])
        return np.float32(sets).ravel()

    hist=[0]*256
    hists=[hist.copy() for i in range(imgsize*imgsize//(blocksize**2))]
    for i in range(imgsize//blocksize):
        for j in range(imgsize//blocksize):
            block=lbps[blocksize*i:blocksize*i+blocksize,blocksize*j:blocksize*j+blocksize].copy()
            block=np.int32(block)
            for m in range(blocksize):
                for n in range(blocksize):
                    hists[i*imgsize//blocksize+j][block[m,n]]+=1
    # equivalence-mode
    #for i in range(len(hists)):
    #    for k in variantNums[::-1]:
    #        del hists[i][k]
    return np.row_stack(hists).ravel()