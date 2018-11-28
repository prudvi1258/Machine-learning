#!python3
import numpy as np

training_input=([1.81,0.80,0.44],
                [1.77,0.70,0.43],
                [1.60,0.60,0.38],
                [1.54,0.54,0.37],
                [1.66,0.65,0.40],
                [1.90,0.90,0.47],
                [1.75,0.64,0.39],
                [1.77,0.70,0.40],
                [1.59,0.55,0.37],
                [1.71,0.75,0.42],
                [1.81,0.85,0.43])
training_output=([0],
                 [0],
                 [1],
                 [1],
                 [0],
                 [0],
                 [1],
                 [1],
                 [1],
                 [0],
                 [0])                
X = np.array(training_input, dtype=float)
y = np.array(training_output, dtype=float)


class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 3
    self.outputSize = 1
    self.hiddenSize = 3

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x3) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
    self.biash=np.random.randn(1,3)
    self.biaso=np.random.randn(1,1)

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1)+self.biash # dot product of X (input) and first set of 3x2 weights
    #print(self.z.shape)
    self.z2 = self.ReLU(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2)+self.biaso # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    #o[o>0.5]=1
    #o[o<=0.5]=0
    return o 

  def sigmoid(self, s):
    # activation function 
    return 1/(1+np.exp(-s))

  def ReLU(self,s):
    #return s
    return np.maximum(0,s)

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def dReLU(self,s):  
    l=s.copy()  
    l[l>0]=1
    l[l<=0]=0
    return l    

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.dReLU(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += 0.2*X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += 0.2*self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
    self.biash+=np.sum(self.z2_delta, axis=0,keepdims=True)
    self.biaso+=np.sum(self.o_delta, axis=0,keepdims=True)

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

def threshold(s):
  l=s.copy()
  l[l>0.5]=1
  l[l<=0.5]=0
  return l

NN = Neural_Network()
for i in range(5000): # trains the NN 5,000 times
  NN.train(X, y)
  print("Input: \n" + str(X))
  print(NN.W1)
  print(NN.W2)
  print ("Actual Output: \n" + str(y) )
  print ("Predicted Output: \n" + str(NN.forward(X))) 
  print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print ("\n")

final_out=threshold(NN.forward(X))
print ("Final Output: \n" + str(final_out))

#classifying new data
X_new=[[1.63,0.60,0.37],
       [1.75,0.72,0.41]]
print("Input: \n" + str(X_new))
o_new=NN.forward(X_new)
o_new=threshold(o_new)
print ("Predicted Output: \n" + str(o_new))
