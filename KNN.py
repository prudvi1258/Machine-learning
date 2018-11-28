import pandas as pd
import numpy as np

df = pd.read_csv('iris.csv')
data = df.iloc[:,:-1].values
target = [1]*50 + [2]*50 + [3]*50   #target vector 
classval = ['None','Iris-setosa', 'Iris-versicolor' ,'Iris-virginica']
def most_common(lst):
    return max(set(lst), key=lst.count)

test = np.array([7.2, 3.6, 5.1, 2.5]).reshape(1,-1)
dist = []


for index,pt in enumerate(data):
    clsval = 1*(index <50 ) + 2*(49 < index < 100 ) + 3*(index > 99)
    dist.append( [np.sqrt( np.sum( (test-pt)**2 )), clsval ])

dist = sorted(dist, key= lambda x: x[0])
clslist = [x[1] for x in dist]

print('For my model:')
for K in range(1,6):
    out = most_common(clslist[:K])
    print('For K = {}, class output = {} ({})'.format(K, out,classval[out]))



### Using Scikit-learn
from sklearn.neighbors import KNeighborsClassifier as KNN
print('\nFor the model trained using Sklearn:')
for K in range(1,6):
    model = KNN(n_neighbors = K ) 
    model.fit(data, target)
    out = model.predict(test)[0]
    print('For K = {}, class output = {} ({})'.format(K, out, classval[out]))
    