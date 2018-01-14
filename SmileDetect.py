import numpy as np
import cv2
import os
import time

global variantNums
global uniNums
def uniPattern(lbp0):
    count=0
    for i in range(len(lbp0)-1):
        if lbp0[i]!=lbp0[i+1]:
            count+=1
    if count>2:
        return "00000101"
    return lbp0
def get_uniNums():
    uniNums = []
    for i in range(256):
        if uniPattern("{:0>8}".format(str(bin(i))[2:]))=="00000101":
            uniNums.append(i)
    return uniNums

def getInvariant(lbp0):
    lbps=[]
    for i in range(len(lbp0)):
        lbps.append(lbp0[i:]+lbp0[0:i])
    return min([int(num,2) for num in lbps])
def get_variantNum():
    Nums = []
    for i in range(256):
        if getInvariant("{:0>8}".format(str(bin(i))[2:]))!=i:
            Nums.append(i)
    return Nums

variantNums=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 48, 49, 50, 52, 54, 56, 57, 58, 60, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254]
uniNums=[9, 10, 11, 13, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 125, 130, 132, 133, 134, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 194, 196, 197, 198, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 226, 228, 229, 230, 232, 233, 234, 235, 236, 237, 238, 242, 244, 245, 246, 250]
def dist2(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2
def getLBP(img,imgsize=60,blocksize=5,mode=None):
    if len(img.shape)==3:
        img=img[:,:,0]
    size=min(img.shape[0],img.shape[1])
    img=img[:size,:size]
    img=cv2.resize(img,(imgsize,imgsize))
    lbps=np.zeros((imgsize,imgsize))
    for i in range(1,imgsize-1):
        for j in range(1,imgsize-1):
            if dist2((i,j),(imgsize/2,imgsize/2))<=imgsize**2/4:
                cell=img[i-1:i+2,j-1:j+2].copy()
                val=cell[1,1]
                cell[cell<val]=0
                cell[cell!=0]=1
                lbp=str(cell[0,0])+str(cell[0,1])+str(cell[0,2])+ \
                    str(cell[1,2]) +str(cell[2,2])+str(cell[2,1])+ \
                    str(cell[2, 0]) +str(cell[1,0])
                if mode==None:
                    lbps[i, j] = int(lbp,2)
                elif mode=="uniform":
                    lbps[i, j] = int(uniPattern(lbp),2)
                elif mode=="invar":
                    lbps[i,j]=getInvariant(lbp)

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
                    hists[i*(imgsize//blocksize)+j][block[m,n]]+=1
     #equivalence-mode
    if mode=="invar":
        for i in range(len(hists)):
            for k in variantNums[::-1]:
                del hists[i][k]
    elif mode=="uniform":
        for i in range(len(hists)):
            for k in uniNums[::-1]:
                del hists[i][k]
    #half
    for i in range(len(hists)//2):
        del hists[0]
    return np.row_stack(hists).ravel()
def normalize(data,*args):
    data=np.float32(data)
    if len(args)==0:
        maxm=data.max()
        minm=data.min()
    else:
        maxm=args[0]
        minm=args[1]
    for i in range(data.shape[0]):
        try:
            for j in range(data.shape[1]):
                data[i,j]=(data[i,j]-minm)/(maxm-minm)
        except:
            data[i] = (data[i] - minm) / (maxm - minm)
    return data
def getTrainData(imgsize, blocksize, s1, s2, num):
    #get train data
    lbps=[]
    c=0
    print("\ncollecting positive samples for training...")
    for i in range(s1,s1+num):
        if os.path.exists("faces\\{}.jpg".format(i)):
            img=cv2.imread("faces\\{}.jpg".format(i))
            c+=1
            lbps.append(getLBP(img,imgsize,blocksize))
        print("\r{:.2f}%".format(100*(i-s1)/num),end="")
    label1=np.ones((c,1))
    c=0
    print("\ncollecting negative samples for training...")
    for i in range(s2,s2+num):
        if os.path.exists("faces\\{}.jpg".format(i)):
            img=cv2.imread("faces\\{}.jpg".format(i))
            c+=1
            lbps.append(getLBP(img,imgsize,blocksize))
        print("\r{:.2f}%".format(100 * (i-s2) / num),end="")
    label0=np.zeros((c,1))
    trainData=np.float32(lbps)
    trainLabel=np.vstack((label1,label0))
    trainLabel=np.int32(trainLabel)
    np.save("trainData.npy",trainData)
    np.save("trainLabel.npy",trainLabel)
    return trainData,trainLabel
def trainModel(trainData,trainLabel):
    #init models
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_RBF)
    #model.setC(1.0)
    #train
    print("training...")
    model.train(trainData, cv2.ml.ROW_SAMPLE, trainLabel)
    model.save("mymodel.txt")

def getTestData(imgsize, blocksize, s1, s2, num):

    tests=[]
    c = 0
    print("\ncollecting positive samples for testing...")
    for i in range(s1,s1+num):
        if os.path.exists("faces\\{}.jpg".format(i)):
            img = cv2.imread("faces\\{}.jpg".format(i))
            c += 1
            tests.append(getLBP(img,imgsize,blocksize))
        print("\r{:.2f}%".format(100 * (i - s1) / num),end="")
    label1 = np.ones((c, 1))
    c = 0
    print("\ncollecting negative samples for testing...")
    for i in range(s2,s2+num):
        if os.path.exists("faces\\{}.jpg".format(i)):
            img = cv2.imread("faces\\{}.jpg".format(i))
            c += 1
            tests.append(getLBP(img,imgsize,blocksize))
        print("\r{:.2f}%".format(100 * (i - s2) / num),end="")
    label0 = np.zeros((c, 1))
    testData = np.float32(tests)
    testLabel = np.vstack((label1, label0))
    testLabel = np.int32(testLabel)
    np.save("testData.npy",testData)
    np.save("testLabel.npy",testLabel)
    return testData,testLabel
def testModel(testData,testLabel):
    model=cv2.ml.SVM_load("mymodel.txt")
    ret,result=model.predict(testData)
    arg=[0,0,0,0]
    for i in range(result.size):
        if result[i,0]==1 and testLabel[i,0]==1:
            arg[0]+=1
        elif result[i,0]==0 and testLabel[i,0]==0:
            arg[1]+=1
        elif result[i,0]==0 and testLabel[i,0]==1:
            arg[2]+=1
        elif result[i, 0] == 1 and testLabel[i, 0] == 0:
            arg[3] += 1
    print()
    print(arg)
    F1=2*arg[0]/(2*arg[0]+arg[2]+arg[3])
    acc=(arg[0]+arg[1])/(arg[0]+arg[1]+arg[2]+arg[3])
    #print("F1:{}".format(F1))
    print("accuracy:{}".format(acc))

def crossTrain(imgsize, blocksize, trainset, testset,order,mode):
    #get train data
    lbps=[]
    c=0
    print("\ncollecting positive samples for training...")
    for num in trainset:
        dirs="TrainData\\"+str(num)+"\\T"
        files=os.listdir(dirs)
        c+=len(files)
        for i in range(len(files)):
            img = cv2.imread(dirs+"\\"+files[i])
            lbps.append(getLBP(img, imgsize, blocksize,mode))
            print("\rset{}: {:.2f}%".format(str(num),100*i/len(files)),end="")
        print()
    label1=np.ones((c,1))
    c=0
    print("\ncollecting negative samples for training...")
    for num in trainset:
        dirs="TrainData\\"+str(num)+"\\F"
        files=os.listdir(dirs)
        c+=len(files)
        for i in range(len(files)):
            img = cv2.imread(dirs+"\\"+files[i])
            lbps.append(getLBP(img, imgsize, blocksize,mode))
            print("\rset{}: {:.2f}%".format(str(num),100*i/len(files)),end="")
        print()
    label0=np.zeros((c,1))
    trainData=np.float32(lbps)
    trainLabel=np.vstack((label1,label0))
    trainLabel=np.int32(trainLabel)

    # get test data
    lbps = []
    c = 0
    print("\ncollecting positive samples for testing...")
    for num in testset:
        dirs = "TrainData\\" + str(num) + "\\T"
        files = os.listdir(dirs)
        c += len(files)
        for i in range(len(files)):
            img = cv2.imread(dirs + "\\" + files[i])
            lbps.append(getLBP(img, imgsize, blocksize,mode))
            print("\rset{}: {:.2f}%".format(str(num), 100 * i / len(files)), end="")
        print()
    label1 = np.ones((c, 1))
    c = 0
    print("\ncollecting negative samples for testing...")
    for num in testset:
        dirs = "TrainData\\" + str(num) + "\\F"
        files = os.listdir(dirs)
        c += len(files)
        for i in range(len(files)):
            img = cv2.imread(dirs + "\\" + files[i])
            lbps.append(getLBP(img, imgsize, blocksize,mode))
            print("\rset{}: {:.2f}%".format(str(num), 100 * i / len(files)), end="")
        print()
    label0 = np.zeros((c, 1))
    testData = np.float32(lbps)
    testLabel = np.vstack((label1, label0))
    testLabel = np.int32(testLabel)
    #normalize
    print("\nnormalizing data...")
    maxm = max([trainData.max(), testData.max()])
    minm = min([trainData.min(), testData.min()])
    trainData = normalize(trainData, maxm, minm)
    testData = normalize(testData, maxm, minm)
    #train models
    # init models
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_RBF)
    # model.setC(1.0)
    # train
    print("training...")
    model.train(trainData, cv2.ml.ROW_SAMPLE, trainLabel)
    model.save("TrainData\\mymodel{}.txt".format(str(order)))


    #test models
    model = cv2.ml.SVM_load("TrainData\\mymodel{}.txt".format(str(order)))
    ret, result = model.predict(testData)
    arg = [0, 0, 0, 0]
    for i in range(result.size):
        if result[i, 0] == 1 and testLabel[i, 0] == 1:
            arg[0] += 1
        elif result[i, 0] == 0 and testLabel[i, 0] == 0:
            arg[1] += 1
        elif result[i, 0] == 0 and testLabel[i, 0] == 1:
            arg[2] += 1
        elif result[i, 0] == 1 and testLabel[i, 0] == 0:
            arg[3] += 1
    print()
    print(arg)
    F1 = 2 * arg[0] / (2 * arg[0] + arg[2] + arg[3])
    precision=arg[0]/(arg[0]+arg[2])
    acc = (arg[0] + arg[1]) / (arg[0] + arg[1] + arg[2] + arg[3])
    print("F1:{}".format(F1))
    print("accuracy:{}".format(acc))

    f=open("log.txt","a")
    f.write("{}. F1={},precision={},accuracy={},TP,TN,FP,FN={}\n".format(str(order),str(F1),str(precision),str(acc),str(arg)))
    f.write("imgsize={}, blocksize={}, trainset={}, testset={}\n".format(str(imgsize), str(blocksize), str(trainset), str(testset)))

    f.close()


def trainone():
    t0=time.clock()

    getTrainData(25,5, 0, 2165, 1800)
    getTestData(25,5, 1900, 3800, 200)
    trainData=np.load("trainData.npy")
    trainLabel=np.load("trainLabel.npy")
    testData=np.load("testData.npy")
    testLabel=np.load("testLabel.npy")
    print("\nnormalizing data...")
    maxm=max([trainData.max(),testData.max()])
    minm=min([trainData.min(),testData.min()])
    trainData=normalize(trainData,maxm,minm)
    testData=normalize(testData,maxm,minm)

    trainModel(trainData,trainLabel)
    testModel(testData,testLabel)

    t1=time.clock()
    h=(t1-t0)//3600
    mi=(t1-t0-3600*h)//60
    s=(t1-t0-3600*h-60*mi)%60
    print("{}h {}min {:3f}s".format(h,mi,s))

def main():
    set=[0,1,2,3,4,5,6,7,8,9]
    for i in range(10):
        temp=set.copy()
        del temp[i]
        crossTrain(36,6,temp,[i],i,None)
