####### QUICK, BASIC NEURAL NETWORK EXERCISE #######
#######    WRITTEN FROM SCRATCH IN PYTHON    #######

####### IMPORT PACKAGES #######

import numpy as np
import random as rand
import matplotlib.pyplot as plt

####### LOAD DATA SET #######

trImgFile = open("train-images-idx3-ubyte", 'br')
trLabFile = open("train-labels-idx1-ubyte", 'br')
teImgFile = open( "t10k-images-idx3-ubyte", 'br')
teLabFile = open( "t10k-labels-idx1-ubyte", 'br')

# load in all the training images

trImgFile.seek(4) # skip magic number
buf = trImgFile.read(4)
trImgNum  = int.from_bytes(buf,"big") 
buf = trImgFile.read(4)
imgHeight = int.from_bytes(buf,"big")
buf = trImgFile.read(4)
imgWidth  = int.from_bytes(buf,"big")

trainImg = [0]*(trImgNum)

for i in range(0, trImgNum):
    buf = trImgFile.read(imgWidth*imgHeight)
    trainImg[i] = np.asarray(list(buf)).reshape(28,28)

    
# load in all the test images

teImgFile.seek(4) # skip magic number
buf = teImgFile.read(4)
teImgNum = int.from_bytes(buf,"big") 
buf = teImgFile.read(4)
imgHeight = int.from_bytes(buf,"big")
buf = teImgFile.read(4)
imgWidth  = int.from_bytes(buf,"big")

testImg  = [0]*(teImgNum)

for i in range(0, teImgNum):
    buf = teImgFile.read(imgWidth*imgHeight)
    testImg[i] = np.asarray(list(buf)).reshape(28,28)

# load in all the training labels

trLabFile.seek(4) # skip magic number
buf = trLabFile.read(4)
trLabNum = int.from_bytes(buf,"big") 

trainLab = np.zeros(trLabNum, dtype = np.uint8)
buf = trLabFile.read(trLabNum)

for i, byte in enumerate(buf):
    trainLab[i] = int(byte)

# load in all the training labels

teLabFile.seek(4) # skip magic number
buf = teLabFile.read(4)
teLabNum = int.from_bytes(buf,"big") 

testLab = np.zeros(teLabNum, dtype = np.uint8)
buf = teLabFile.read(teLabNum)

for i, byte in enumerate(buf):
    testLab[i] = int(byte)
    
trImgFile.close()
trLabFile.close()
trImgFile.close()
teImgFile.close()

# testImg   (10000)
# testLab   (10000)
# trainImg (60000)
# trainLab (60000)

#### SIGMOID FUNCTION ####
def sig(x):
    return 1/(1 + pow(2.718, -x))

#### DERIVATE OF SIGMOID FUNCTION ####
def dsig(x):
    ans  = pow(2.718, x/2)
    ans += 1/ans
    ans *= ans
    ans  = 1/ans
    return ans

#### PASS INPUT THROUGH NN AND GET OUTPUT ####

def runNet(inp):
    layer[0] = np.expand_dims(inp.flatten(),axis=1)/255
    for i,each in enumerate(weightMat):
        layer[i+1] = each.dot(layer[i])
        layer[i+1] = sig(layer[i+1])
    return layer[-1]

#### GET DESIRED OUTPUT OF NN USING LABEL ####

def getY(label):
    y = np.zeros(layer[-1].shape)
    y[label] = 1;
    return y

#### FIND LOSS FOR PARTICULAR OUTPUT ####

def getCost(label):
    y = getY(label)
    d = (y - layer[-1])
    return (d*d).sum()

#### CLASSIFY AN INPUT USING THE CURRENT WEIGHTS ####

def classify(inp):
    runNet(inp)
    ans = layer[-1].argmax()
    return ans

#### DERIVATIVE OF ACTIVATED LAYER WRT TO UNACTIVATED LAYER ####

def buildSigmoidSensitivityMatrix(n):
    S = layer[n]*(1-layer[n])
    S = S.dot(np.ones((1,S.shape[0])))
    S*=np.eye(S.shape[0])
    
    return S

#### DERIVATIVE OF ONE LAYER RELATIVE TO PREVIOUS LAYER ####

def getLayerTransition(n):
    if(n == 0):
        print("Cannot perform on input layer!")
        return
    
    pos = n >= 0
    
    S = buildSigmoidSensitivityMatrix(n)

    res = weightMat[n - pos].transpose().dot(S)
    return res

#### DERIVATIVE OF UNACTIVATED LAYER WRT PREVIOUS WEIGHT MATRIX ####

def buildBetaOmegaSensitivityMatrix(n): 
    if(n == 0):
        print("Cannot perform on input layer!")
        return
    pos = n >= 0
    
    s = weightMat[n-pos].shape    
    ans = np.zeros((1,s[0],s[1]))
    ans = np.pad(ans,((0,s[0]-1),(0,0),(0,0)), "edge")
    for k,each in enumerate(ans):
        each[k] = layer[n-1].transpose()
    return ans

#### WORKAROUND TO MAKE NUMPY LOOK LIKE MY NOTES :) ####

def convertMatDim(shape, mode=0):
    if(len(shape) > 2):
        if (mode == 0):
            p1 = np.asarray(shape[:-2])
            p2 = np.asarray(shape[-2:])
            p1 = np.flip(p1,0)
        else:
            p1 = np.asarray(shape[:2])
            p2 = np.asarray(shape[2:])
            p2 = np.flip(p2,0)
        
        ans = np.concatenate((p2, p1))
    else:
        ans = shape
    return ans

#### SETS UP AND ANALYZES MATRICES FOR dot3D ####

def dot3Dsetup(mat1, mat2):
    ms1 = convertMatDim(mat1.shape)
    ms2 = convertMatDim(mat2.shape)
    
    # concatenate all but last of first with all but first of last
    ansShape = np.concatenate((ms1[:-1], ms2[1:]))
    ansShape = np.pad(ansShape, (0,4 - len(ansShape)), "constant", constant_values=1)
    ansShape = convertMatDim(ansShape,1).astype(int)
    
    if(ms1[-1] != ms2[0]):
        print("Matrices do not conform!")
        return -1
    if(len(ms1) > 3 or len(ms2) > 3):
        print("No more than 3 dimensions!")
        return -1

    rank1 = len(mat1.shape)
    rank2 = len(mat2.shape)
    
    if(rank2 == 1):
        rank2 += 1
        mat2 = np.expand_dims(mat2,1) # make column vector out of mat2
    
    for i in range(3-rank1):
        mat1 = np.expand_dims(mat1,0) # make each a 3D matrix
    for i in range(3-rank2):
        mat2 = np.expand_dims(mat2,0)
    
    return [ansShape,mat1,mat2,rank1,rank2]

#### CUSTOM 3D MATRIX MULTIPLICATION ####

def dot3D(mat1, mat2):  
    
    ansShape,mat1,mat2,rank1,rank2 = dot3Dsetup(mat1, mat2)
    if(ansShape.any() == -1):
        print("Dot Product Failed :(")
        return
    
    ms1 = mat1.shape
    ms2 = mat2.shape
    ans = np.zeros(ansShape)
    
    for r1 in range(ms1[1]):
        for c1 in range(ms1[2]):
            for l2 in range(ms2[0]):
                for c2 in range(ms2[2]):
                    a = mat1[:,r1,c1]
                    b = mat2[l2,:,c2]
                    ans[l2,c2,r1,c1] = a.dot(b)
            
    while(ans.shape[0] == 1):
        ans = ans[0]
        
    return ans

#### PHEW! FINALLY SOME GRADIENT DESCENT ####

def backpropUsing(inp, label, trainStep):
    y  = getY(label)
    runNet(inp)
    dC = y - layer[-1]
    for n in range(len(layer)-1, 0, -1):
        S  = buildSigmoidSensitivityMatrix(n)
        bw = buildBetaOmegaSensitivityMatrix(n)
        dW = dot3D(bw,S.dot(dC))
        mag = np.linalg.norm(dW)
        if(mag != 0):
            dW /= mag
        else:
            dW /= 0.000001
        dW *= trainStep
        weightMat[n-1] += dW
        dC  = getLayerTransition(n).dot(dC)

####### GENERATE LAYERS #######
structure = [784, 16, 7, 10]
layer   = [np.zeros((i,1)) for i in structure]

####### GENERATE WEIGHT MATRICES AND INITIALIZE WITH RANDOM VALUES (-1, 1) #######
weightMat = [2*np.random.random((structure[i+1],structure[i]))-1 for i in range(len(structure) - 1)]

####### BASIC SGD #######

sampNum = 1000 # perform sgd on first 'sampNum' images in training data

for imgNum in range(sampNum):
    backpropUsing(trainImg[imgNum], trainLab[imgNum], 0.4)

##### TEST NN ON TRAINING DATA FIRST #####

acc = 0
randAcc = 0
for each in range(sampNum):
    rc = np.floor(np.random.rand()*10).astype(int)
    acc += (classify(trainImg[each]) == trainLab[each])
    randAcc += (rc == trainLab[each])

print("Results")
print(acc)
print("out of", sampNum)
print(acc/sampNum)

print("\nRandom Control")
print(randAcc)
print("out of", sampNum)
print(randAcc/sampNum)


##### TEST NN ON TEST DATA #####

acc= 0
for i,lab in enumerate(testLab):
    acc += (classify(testImg[i]) == lab)
print("Correct Classifications out of 10000 Data:\n", acc)


print("Accuracy on Test Data:")
print(acc/len(testLab))
