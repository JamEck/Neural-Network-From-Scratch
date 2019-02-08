import numpy as np
import matplotlib.pyplot as plt

import MNIST_Loader as ml

####### LOAD DATA SET #######

trainImg = ml.loadMNISTimages("train-images-idx3-ubyte").reshape((-1,784))/255
testImg  = ml.loadMNISTimages("t10k-images-idx3-ubyte" ).reshape((-1,784))/255
trainLab = ml.loadMNISTlabels("train-labels-idx1-ubyte")
testLab  = ml.loadMNISTlabels("t10k-labels-idx1-ubyte")

# testImg  (10000, 784)
# testLab  (10000)
# trainImg (60000, 784)
# trainLab (60000)

class NN:
    def __init__(self,sizes):
        self.sizes = sizes.copy()
        self.depth = len(list(sizes))-1 # number of layers, excluding the output layer
        
        # Create Layers and Parameters
        self.layer     = [[]]*self.depth
        self.beta      = [[]]*self.depth
        self.weightMat = [[]]*self.depth
        self.bias      = [[]]*self.depth
        for i in range(self.depth):
            self.layer[i] = np.ones(sizes[i])
            self.beta[i]  = np.ones(sizes[i])
            self.weightMat[i] = 2*np.random.random((sizes[i], sizes[i+1]))-1
            self.bias[i]      = 2*np.random.random(sizes[i+1])-1
        
        # list of parameter gradients for backpropagation
        self.wgrad = [np.zeros(self.weightMat[i].shape) for i in range(self.depth)]
        self.bgrad = [np.zeros(self.bias[i].shape) for i in range(self.depth)]
        
        # Create Layers and Parameters
        self.output = np.ones(sizes[-1])
        self.betaOutput = np.ones(sizes[-1])
        
        self.activation = NN.sigmoid
        self.activationDeriv = NN.dsigmoid
        
        self.costFn = self.MSE
        self.dcostFn = self.dMSE
        self.cost = 0

    def prepInp(self, inp):
        try:
            inp = np.array(inp,dtype=np.float64)
        except:
            print("Could not convert input into numpy array.")
            return
        
        if(inp.ndim == 0):
            inp = np.expand_dims(inp,0)
              
        if(inp.shape[0]!=self.sizes[0]):
            print("Dimensions don't conform to NN input layer.")
            print("Input Entered:", inp.shape, "\nInput Layer: ({}, 1)".format(self.sizes[0]))
            return
        
        return inp
    
    def sigmoid(inp):
        return 1/(1+2.718**(-inp))
    def dsigmoid(inp):
        temp = NN.sigmoid(inp)
        return temp*(1-temp)
    
    def relu(inp):
        return inp*(inp>0)
    def drelu(inp):
        return   1*(inp>0)
    
    def tanh(inp):
        return 2/(1+2.718**(-2*inp))-1

    
    def forward(self, inp):        
        try:
            np.copyto(self.layer[0],self.prepInp(inp))
            np.copyto(self.beta[0],self.prepInp(inp))
        except:
            print("Couldn't copy to input layer.")
            return
                
        for i in range(self.depth-1):
            self.beta[i+1] = self.layer[i].dot(self.weightMat[i]) + self.bias[i]
            self.layer[i+1] = self.activation(self.beta[i+1])
            
        self.betaOutput = self.layer[-1].dot(self.weightMat[-1]) + self.bias[-1]
        self.output = self.activation(self.betaOutput)
        return self.output
    
    def getDesired(self, label, val = 1):
        output = np.zeros(self.output.shape)
        if(label >= self.output.size or label < 0):
            print("Label out of range.")
        else:
            output[label] = val
        return output
        
    def MSE(self, act, des):
        return ((des - act)**2).sum()/act.size
    def dMSE(self, act, des):
        return 2*(des - act)/act.size
    
    def getCost(self, label):
        self.cost = self.costFn(self.output, self.getDesired(label))
        return self.cost

    def Build_Activation_Sensitivity_Matrix(self,inp):
        '''Derivative of an activated layer wrt to its unsquashed layer'''
        k = inp.shape[0]
        ans = np.zeros((k,k))
        for i in range(k):
            ans[i,i] = self.activationDeriv(inp[i])
        return ans
        
    def Build_Beta_Omega_Sensitivity_Matrix(self, layerNum):
        '''Derivative of the unsquashed layer wrt to the Weight Matrix'''
        if(layerNum >= self.depth):
            print("Index out of range.")
            return -1
        
        [j,k] = self.weightMat[layerNum].shape 
        ans = np.zeros((k,j,k))
        for n in range(k):
            np.copyto(ans[n,:,n],self.layer[layerNum])
        return ans
    
    def backprop(self, label):
        accum = self.dcostFn(self.output, self.getDesired(label))
        accum = accum.dot(self.Build_Activation_Sensitivity_Matrix(self.betaOutput))
        
        for i in range(self.depth):
            ind = -(i+1)
            self.wgrad[ind] += accum.dot(self.Build_Beta_Omega_Sensitivity_Matrix(ind).swapaxes(0,1))
            self.bgrad[ind] += accum
            accum = accum.dot(self.weightMat[ind].transpose())
            accum = accum.dot(self.Build_Activation_Sensitivity_Matrix(self.beta[ind]))
            
    def stepSGD(self, r = 0.1):
        for i,dw in enumerate(self.wgrad):
            # normalize gradient vector
            dwmag = np.linalg.norm(dw)
            if(dwmag!=0):
                dw/=dwmag
            self.weightMat[i] += dw*r
        for i,db in enumerate(self.bgrad):
            dbmag = np.linalg.norm(db)
            if(dbmag!=0):
                db/=dbmag
            self.bias[i] += db*r
        
        for each in self.wgrad:
            each.fill(0)
        for each in self.bgrad:
            each.fill(0)

    def train(self, inp, label, batchSize, r=0.1):
        batchSize = int(batchSize)
        if(batchSize<=0):
            print("Invalid Batch Size. Setting to default of 10")
            batchSize = 10
            
        if(inp.ndim == 1):
            inp = np.expand_dims(inp, 0)
        if(label.ndim == 0):
            label=np.expand_dims(label,0)
           
        if(inp.shape[0]!=label.shape[0]):
            print("Unequal inputs and labels.")
            return
        
        for i in range (inp.shape[0]):
            self.forward(inp[i])
            self.backprop(label[i])
            if(i%batchSize==0):
                self.stepSGD(r)
            if(i%(inp.shape[0]/10)==0):
                print("%d%%" % (i/inp.shape[0]*100), end=" ")

           
        self.stepSGD(r)
        print("100%\nFINISHED")
        
    def classify(self, inp):
        self.forward(inp)
        guess = self.output.argmax()
        confidence = self.output.max()
#         print("%d with confidence of %.3f"% (guess, confidence))
        return [guess, confidence]

    def accuracyTrain(self, imgNums):
        perdigit = np.zeros(10)
        digitCount = np.zeros(10)
        for imgNum in imgNums:
            guess = self.classify(trainImg[imgNum])[0]
            lab   = trainLab[imgNum]
#             print("%d)"%imgNum, guess, lab)
            if(guess == lab):
                perdigit[lab] += 1
            digitCount[lab]+=1
            
        return [perdigit/digitCount, perdigit.sum()/imgNums.size*100]
    
    def accuracyTest(self, imgNums):
        perdigit = np.zeros(10)
        digitCount = np.zeros(10)
        for imgNum in imgNums:
            guess = self.classify(testImg[imgNum])[0]
            lab   = testLab[imgNum]
#             print("%d)"%imgNum, guess, lab)
            if(guess == lab):
                perdigit[lab]+=1
            digitCount[lab]+=1
            
        return [perdigit/digitCount, perdigit.sum()/imgNums.size*100]


sizes = [784,16,10,10]

n = NN(sizes)
n.activation      = NN.sigmoid
n.activationDeriv = NN.dsigmoid


N = 10000
epochNum = 3
for epoch in range(epochNum):
    r=0.1
    print("Epoch %d: Learning rate set to %0.2f" % (epoch+1, r))
    n.train(trainImg[:N], trainLab[:N], 20)


print(n.accuracyTrain(np.arange(10000)))
print(n.accuracyTest(np.arange(10000)))


imgNum = 1060
print(n.classify(trainImg[imgNum]))
print(trainLab[imgNum])

n.output.sum()

