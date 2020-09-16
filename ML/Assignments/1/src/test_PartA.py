import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#A1: Euclidean distance of two vectors
def l2_norm(x,y):
    x,y = np.array(x),np.array(y)
    return np.linalg.norm(x-y,2)


#A2: Manhattan distance of two vectors
def l1_norm(x,y):
    x,y = np.array(x),np.array(y)
    return np.linalg.norm(x-y,1)


#A3: Accuracy and Generalization Error
def accuracy(x,y):
    x,y = np.array(x),np.array(y)
    pred = (x == y).astype(np.int)
    return pred.mean()

def gen_error(x,y):
    return 1 - accuracy(x,y)

# the confusion matrix is the basic of precision, recall, and F1 scores calculation.

def compute_confusion_matrix(actual, predicted):
    
    arary_actual = np.array(actual)
    array_pred = np.array(predicted)
    
    pd_actual = pd.Series(arary_actual, name='Actual')
    pd_predicted = pd.Series(array_pred, name='Predicted')

    pd_actual = pd.Categorical(pd_actual, categories=[0, 1])
    pd_predicted = pd.Categorical(pd_predicted, categories=[0, 1])

    CM =  pd.crosstab(pd_actual, pd_predicted, dropna=False)
    
    return CM


# the following calculation of precision, recall, and F1 scores are based on the function of the generated confusion matrix

def compute_precision(actual, predicted):
       
    CM =  compute_confusion_matrix(actual, predicted).to_numpy()  # CM is converted into a 2 X 2 array.
    
    TN = CM[0,0]; FP = CM[0,1]; FN = CM[1,0]; TP =  CM[1,1];
    
    precision = TP / (TP + FP)
    
    return precision

def compute_recall(actual, predicted):
    
    CM =  compute_confusion_matrix(actual, predicted).to_numpy()  # CM is converted into a 2 X 2 array.
    
    TN = CM[0,0]; FP = CM[0,1]; FN = CM[1,0]; TP =  CM[1,1];
    
    recall = TP / (TP + FN)
    
    return recall

def compute_F1_score(actual, predicted):
    
    precision = compute_precision(actual, predicted)
    recall = compute_recall(actual, predicted)
    
    F1_score = 2 * precision * recall / (precision + recall)
    
    return F1_score

# The usage of this function is to return the elements used for plotting a ROC, as same as the function roc_curve in sklearn.
# The output elemetns contains fprs, tprs and thresholds. 
# input arguments are y_label, y_prob, and target_label.
# target_label should be either 0 or 1 in our scenario. The defalt value is 1 in this function. 

def generate_ROC_elements(y_label, y_prob, target_label = 1):
    
    # gets the target label.
    if target_label == 0: non_target_label = 1
    if target_label == 1: non_target_label = 0
    
    # converts the input arguments into arrays.
    ar_y_label = np.array(y_label)
    ar_y_prob = np.array(y_prob)
    
    # creates a list to sort the results of predicted y. 
    y_pred = list(y_prob)
    
    # generates list to store the tpr, fpr and threshold.
    tpr_list = [0, 1]     
    fpr_list = [0, 1]
    thres_lish = [1, 0]
    
    # using the for loop to predicte y based on the input y_prob. 
    for i, prob in enumerate(ar_y_prob):
        threshold = prob
        for index, y_prob in enumerate(ar_y_prob):
            if y_prob >= threshold:
                y_pred[index] = target_label
            else:
                y_pred[index] = non_target_label
        
        # uses the function to compute the confusion matrix, and gets the TN, FP, FN, TP. 
        CM = compute_confusion_matrix(y_label, y_pred).to_numpy()           
        TN = CM[0,0]; FP = CM[0,1]; FN = CM[1,0]; TP =  CM[1,1]
        
        # Calculates tpr and fpr. 
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)
    
        # adds the tpr, fpr and threshold into the corresponding lists. 
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        thres_lish.append(threshold)

    # when the for loop is end, generating a dataframe with the lists of threshold, fpr and tpr. 
    data = {'threshold':pd.Series(thres_lish), 'fpr':pd.Series(fpr_list), 'tpr':pd.Series(tpr_list)}
    df_roc = pd.DataFrame(data)
    
    # descending sorting the dataframe according to the threshold column
    df_roc.sort_values(by='threshold', ascending=False, inplace=True)
    
    return np.array(df_roc["fpr"]), np.array(df_roc["tpr"]), np.array(df_roc["threshold"])


# After using the function "generate_precision_recall_curve_elements" above to get the fprs, tprs,
# the following function uses the generated fprs and tprs to plot the ROC curve. 

def plotting_roc_curve(fpr, tpr, label = None): 
    plt.figure(figsize = (10, 10))
    
    # linewidth and fontsize
    lw = 2
    fontsize = 20
    
    # plot roc curve
    plt.plot(fpr, tpr, color='darkorange', lw = lw, label = label) 
    
    # plot y = x
    plt.plot([0, 1], [0, 1], color='navy', lw = lw, linestyle = '--')  
    
    # set the length of x axis and y axis. 
    plt.axis([0, 1, 0, 1.05])
    
    # add title, xlabel, ylabel, and legend. 
    plt.title(f'Receiver operating characteristic Curve ({label})', fontsize = fontsize)
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    plt.legend(loc="lower right", fontsize = fontsize)
    
    plt.show()

# The usage of this function is to return the elements used for plotting a precision-recall curve.
# The output elemetns contains precisions, recalls and thresholds. 
# Input arguments are y_label, y_prob, and target_label.
# Target_label should be either 0 or 1 in our scenario. The defalt value is 1 in this function. 
# It is similar to the function used to generate tprs and fprs above. 
def generate_precision_recall_curve_elements(y_label, y_prob, target_label = 1):
    
    # gets the target label.
    if target_label == 0: non_target_label = 1
    if target_label == 1: non_target_label = 0
    
    # converts the input arguments into arrays.
    ar_y_label = np.array(y_label)
    ar_y_prob = np.array(y_prob)
    
    # creates a list to sort the results of predicted y. 
    y_pred = list(y_prob)
    
    # generates list to store the tpr, fpr and threshold.
    precision_list = []     
    recall_list = []
    thres_lish = []
    
    # using the for loop to predicte y based on the input y_prob. 
    
    for i, prob in enumerate(ar_y_prob):
        threshold = prob
        for index, y_prob in enumerate(ar_y_prob):
            if y_prob >= threshold:
                y_pred[index] = target_label
            else:
                y_pred[index] = non_target_label
        
        # uses the function to compute the confusion matrix, and gets the TN, FP, FN, TP. 
        CM = compute_confusion_matrix(y_label, y_pred).to_numpy()           
        TN = CM[0,0]; FP = CM[0,1]; FN = CM[1,0]; TP =  CM[1,1]
        
        # Calculates tpr and fpr. 
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    
        # adds the tpr, fpr and threshold into the corresponding lists. 
        precision_list.append(precision)
        recall_list.append(recall)
        thres_lish.append(threshold)

    # when the for loop is end, generating a dataframe with the lists of threshold, fpr and tpr. 
    data = {'threshold':pd.Series(thres_lish), 'precision':pd.Series(precision_list), 'recall':pd.Series(recall_list)}
    df_roc = pd.DataFrame(data)
    
    # descending sorting the dataframe according to the threshold column
    df_roc.sort_values(by='threshold', ascending = True, inplace = True)
        
    return np.array(df_roc["precision"]), np.array(df_roc["recall"]), np.array(df_roc["threshold"])


# After using the function "generate_precision_recall_curve_elements" above to get the precisions, recalls and thresholds,
# the following function uses the generated precisions, recalls and thresholds to plot the precision-recall curve. 
    
def plotting_precision_recall_curves(precisions, recalls, thresholds):
    plt.figure(figsize = (10, 6))
    lw = 2
    fontsize = 20
    plt.plot(thresholds, precisions, "b--",  lw = lw, label = "Precision")
    plt.plot(thresholds, recalls, "g-",  lw = lw, label = "Recall")
    
    # set the length of x axis and y axis. 
    plt.axis([-0.05, 1.05, 0, 1.05])
    
    # add title, xlabel, ylabel, and legend. 
    plt.title(f'Precision-Recall curve', fontsize = fontsize)
    plt.xlabel('threshold', fontsize = fontsize)
    plt.legend(loc="lower right", fontsize = fontsize)
    
    plt.show()

def calculate_auc(fpr_x_axis, tpr_y_axis):
    
    # Trapezoidal numerical integration 
    auc = np.trapz(tpr_y_axis, fpr_x_axis)
    
    return auc    
    


#A9: KNN_Classifier model
class knnClassifier():
    def __init__(self):
        pass

    
    def fit(self,X,Y,n_neighbours = 5,weights='uniform',**kwargs):
        self.X = X
        self.Y = Y
        self.k = n_neighbours
        self.weights = weights
        self.distance = kwargs.get('distance','Euclidean')
        self.prob = []
    
    def predict(self,X):
        pred = []
        self.prob = []
        for i in range(X.shape[0]):
            np.seterr(divide='ignore')  ## just ignoring warning if value is too small as it results devide by 0
            if(self.weights == 'uniform' and self.distance == 'Manhattan'):
                indx_opt = np.argpartition(np.linalg.norm(X[i]-self.X,1,axis=1) , range(self.k))  
            elif(self.weights == 'uniform' and self.distance != 'Manhattan'):
                indx_opt = np.argpartition(np.linalg.norm(X[i]-self.X,2,axis=1) , range(self.k)) 
            elif(self.distance == 'Manhattan'):
                dist = np.linalg.norm(X[i]-self.X,2,axis=1)
                indx_opt = np.argpartition(dist , range(self.k))
            else:
                dist = np.linalg.norm(X[i]-self.X,2,axis=1)
                indx_opt = np.argpartition(dist , range(self.k))             
            indx_opt = indx_opt[:self.k]        
            labels = list( map(lambda i : self.Y[i], indx_opt))
            if(self.weights == 'distance'):
                dist = list( map(lambda i : dist[i], indx_opt))
                
                lab = labels.copy()
                lab = np.array(lab)
                dist = np.array(dist)
                dist = 1 / (dist + 0.001)
                #dist = np.nan_to_num(dist)
                #dist = np.where(dist==np.inf,1.7e+50,dist)
                weght_1 = (lab * dist).sum()
                where_0 = np.where(lab == 0)
                where_1 = np.where(lab == 1)
                lab[where_0] = 1
                lab[where_1] = 0
                weight_0 = (lab * dist).sum()
                if(weght_1 > weight_0):
                    pred.append(1.)
                else:
                    pred.append(0.)
            else:
                labels = np.array(labels)
                guess =  (labels == 1).astype(np.int).mean()
                self.prob.append(guess)
                if(guess > 0.5):
                    pred.append(1.)  ## float or int
                else:
                    pred.append(0.)           
        pred = np.array(pred)
        #return np.reshape(pred,(X.shape[0],1))
        return pred
    
    def getProb(self,X):
        self.predict(X)
        return self.prob

#############################################################################

##18

def sFold(folds,data,labels,model,error_fuction,**model_args):
#def sFold(folds,data,labels,model,error_fuction,**model_args):
    if(labels.shape == (labels.shape[0],)):
        labels = np.expand_dims(labels,axis=1)
    dataset = np.concatenate([data,labels],axis=1)
    s_part = s_partition(dataset,folds)
    pred_y = []
    true_y = []
    for idx,val in enumerate(s_part):
        test_y = val[:,-1]
        #test_y = np.expand_dims(test_y, axis=1)
        test = val[:,:-1]
        train = np.concatenate(np.delete(s_part,idx,0))
        label = train[:,-1]
        train = train[:,:-1]        
        model.fit(train,label,**model_args)       
        pred = model.predict(test)
        pred_y.append(pred)
        true_y.append(test_y)
    pred_y = np.concatenate(pred_y)
    true_y = np.concatenate(true_y)

    avg_error = error_fuction(pred_y,true_y).mean()   
    result = {'Expected labels':true_y, 'Predicted labels': pred_y,'Average error':avg_error }
    return result


#helper
def s_partition(x,s):
    return np.array_split(x,3)
            

