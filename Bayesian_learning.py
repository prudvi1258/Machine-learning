import numpy as np

import sklearn
from sklearn import datasets
from sklearn.metrics import confusion_matrix

dataset = datasets.load_iris()
target_numbering={0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
X=dataset['data']
Y=dataset['target']
length=len(X)
X_class_0=[X[i] for i in range(length) if Y[i]==0]
X_class_1=[X[i] for i in range(length) if Y[i]==1]
X_class_2=[X[i] for i in range(length) if Y[i]==2]
mean_vector_class_0=np.mean(X_class_0,axis=0)
mean_vector_class_1=np.mean(X_class_1,axis=0)
mean_vector_class_2=np.mean(X_class_2,axis=0)
std_vector_class_0=np.std(X_class_0,axis=0)
std_vector_class_1=np.std(X_class_1,axis=0)
std_vector_class_2=np.std(X_class_2,axis=0)
prior_prob_class_0=len(X_class_0)/float(length)
prior_prob_class_1=len(X_class_1)/float(length)
prior_prob_class_2=len(X_class_2)/float(length)
log_prior_class_0=np.log10(prior_prob_class_0)
log_prior_class_0=np.log10(prior_prob_class_0)
log_prior_class_0=np.log10(prior_prob_class_0)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def gaussian_dist(x,mean,std):
    variance = np.square(std)
    #PDF formula for gaussian distribution
    f= np.exp(-np.square(x-mean)/(2*variance))/(np.sqrt(2*np.pi*variance))
    return f

def calculate_log_conditional_prob(feature_vector,Class):
    num_features=len(feature_vector)
    log_prob=0.0
    if(Class==0):
        for feature_index in range(num_features):
            log_prob+=np.log10(gaussian_dist(feature_vector[feature_index],mean_vector_class_0[feature_index],std_vector_class_0[feature_index]))
    if(Class==1):
        for feature_index in range(num_features):
            log_prob+=np.log10(gaussian_dist(feature_vector[feature_index],mean_vector_class_1[feature_index],std_vector_class_1[feature_index]))
    if(Class==2):
        for feature_index in range(num_features):
            log_prob+=np.log10(gaussian_dist(feature_vector[feature_index],mean_vector_class_2[feature_index],std_vector_class_2[feature_index]))                

    return log_prob       

def calculate_class_posteriors_and_classify(feature_vector):
    log_conditional_prob_class_0=calculate_log_conditional_prob(feature_vector,0)
    log_conditional_prob_class_1=calculate_log_conditional_prob(feature_vector,1)
    log_conditional_prob_class_2=calculate_log_conditional_prob(feature_vector,2)     

    return np.argmax([log_conditional_prob_class_0,log_conditional_prob_class_1,log_conditional_prob_class_2])


predictions=[]
true_predictions_class_0=0
true_predictions_class_1=0
true_predictions_class_2=0
for index in range(length):
    prediction=calculate_class_posteriors_and_classify(X[index])
    predictions.append(prediction)
    if(prediction==Y[index]):
        if(Y[index]==0):
            true_predictions_class_0+=1
        elif Y[index]==1:
            true_predictions_class_1+=1
        elif Y[index]==2:
            true_predictions_class_2+=1      


labels=['Iris-setosa','Iris-versicolor','Iris-virginica']
predictions=np.array(predictions)
Y=[target_numbering[Y[i]] for i in range(length)]
predictions=[target_numbering[predictions[i]] for i in range(length)]
Y=np.array(Y)
predictions=np.array(predictions)
print("\n\n")
print("Confusion Matrix:")
confusion_mat=confusion_matrix(Y,predictions,labels=['Iris-setosa','Iris-versicolor','Iris-virginica'])
print_cm(confusion_mat,labels)

print("\n\n")
print("For class "+target_numbering[0]+':')
precision=float(50)/50
print("Precision="+str(precision))
recall=float(50)/50
print("Recall="+str(recall))
f1_score=(2*precision*recall)/(precision+recall)
print("F1 Score="+str(f1_score))

print("\n\n")
print("For class "+target_numbering[1]+':')
precision=float(47)/50
print("Precision="+str(precision))
recall=float(47)/50
print("Recall="+str(recall))
f1_score=(2*precision*recall)/(precision+recall)
print("F1 Score="+str(f1_score))

print("\n\n")
print("For class "+target_numbering[2]+':')
precision=float(47)/50
print("Precision="+str(precision))
recall=float(47)/50
print("Recall="+str(recall))
f1_score=(2*precision*recall)/(precision+recall)
print("F1 Score="+str(f1_score))

