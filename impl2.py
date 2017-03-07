import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#preprocessing 
#cdn
train = pd.read_csv('https://files.fm/down.php?i=eadz8pg7&n=train.csv');
test = pd.read_csv('https://files.fm/down.php?i=xfardqbf&n=test.csv');
# train = pd.read_csv('train.csv');
# test = pd.read_csv('test.csv');
drop_cols_train = ['PassengerId','Name','Survived','Cabin','Embarked','Sex','Ticket']
drop_cols_test = ['PassengerId','Name','Cabin','Embarked','Sex','Ticket']

pre_train = train.drop(drop_cols_train,axis=1)
train_x = pd.concat([ 
                        pre_train,
                        pd.get_dummies(train['Embarked']),
                        pd.get_dummies(train['Sex'])
                    ],axis=1)
train_y = pd.DataFrame({'Survived':train['Survived']});

testPid = test['PassengerId'];
pre_test= test.drop(drop_cols_test,axis=1)
test_x = pd.concat([ 
                        pre_test,
                        pd.get_dummies(test['Embarked']),
                        pd.get_dummies(test['Sex'])
                    ],axis=1)

for col in train_x.columns:
    if np.any(pd.isnull(train_x[col])): 
        train_x[col].fillna(np.median(train_x[np.logical_not(pd.isnull(train_x[col]))][col]),inplace=True)

for col in test_x.columns:
    if np.any(pd.isnull(test_x[col])): 
        test_x[col].fillna(np.median(test_x[np.logical_not(pd.isnull(test_x[col]))][col]),inplace=True)

        
  

# Neural_Network #
#-----------------#
class Neural_Network:
    #defining the hyper parameters
    def __init__(self):
        self.inputLayerSize = 10
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        self.W1 = np.random.normal(0, 0.25, (self.inputLayerSize,self.hiddenLayerSize))
        self.W2 = np.random.normal(0, 0.25, (self.hiddenLayerSize,self.outputLayerSize))
    
    #sigmoid function    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def sigmoidPrime(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    
    def forward(self,X):
        #z(3) = XW(1)
        self.z2 = np.dot(X,self.W1)
        #a2 = f(z(2))
        self.a2 = self.sigmoid(self.z2)
        #z(3) = a(2)w(2)
        self.z3 = np.dot(self.a2,self.W2)
        #yhat = f(z(3))
        yhat = self.sigmoid(self.z3)
        return yhat
    
    def costFunction(self,x,y):
        self.yhat = self.forward(x)
        #actual cost
        j = 0.5*np.sum((y-self.yhat)**2) / y.shape[0]
        return j
    
    def logloss(self,x,y):
        self.yhat = self.forward(x)
        j = (y*np.log(self.yhat) + (1-y)*np.log(1-self.yhat)).mean()
        return j

    def costFunctionPrime(self,x,y):
        self.yhat = self.forward(x)
        # back propagation error of layer 2
        delta3 = np.multiply(-(y-self.yhat),self.sigmoidPrime(self.z3))
        djW2 = np.dot(self.a2.T,delta3)
        # back propagation error of layer 1
        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        djW1 = np.dot(x.T,delta2)
        return djW1,djW2


iteration = 100000
NN = Neural_Network()

rate = 0.001
for i in range(iteration):
    dJdW1,dJdW2 = NN.costFunctionPrime(train_x,train_y)
    NN.W1 = NN.W1 - rate*dJdW1
    NN.W2 = NN.W2 - rate*dJdW2
    cost = NN.costFunction(train_x,train_y)
    if(i%1000==0):
        print (cost)


ans = NN.forward(test_x);
round = np.array(np.round(ans),dtype=np.int);

output = pd.DataFrame({'PassengerId':testPid,'Survived':round.ravel()})
#output.to_csv('out1.csv',index=False)
print (round)

