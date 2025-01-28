import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay

class performance:
    def __init__(self):
        pass

    def test(self,model,device,test_data, per_fnc = "confuse matrix"):
        data = torch.utils.data.DataLoader(test_data, batch_size=1)
        model.eval()
        total_labels = []
        total_ouputs = []
        with torch.no_grad():
            for inputs, labels in data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                total_labels.append(labels[0])
                total_ouputs.append(outputs[0])

        if per_fnc == "confuse matrix":
            self.__confusion_matrix(total_ouputs,total_labels)

        elif per_fnc == "series":
            self.__series(total_ouputs,total_labels)

    def __confusion_matrix(self,y_test,y_ref):
            y_true = []
            y_pred = []
            for idx in range(len(y_test)):
                if (y_test[idx][0]>y_test[idx][1]):
                    y_pred.append([0])
                if (y_test[idx][1]>y_test[idx][0]):
                    y_pred.append([1])
                if (y_ref[idx][0] == 1) &(y_ref[idx][1] == 0):
                    y_true.append([0])
                if (y_ref[idx][0] == 0) &(y_ref[idx][1] == 1):
                    y_true.append([1])
                    # error.append(loss.detach().numpy())
            cm = confusion_matrix(y_true, y_pred)
            cm_d = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [True, False])
            # Compute the accuracy
            accuracy = accuracy_score(y_true, y_pred)
            sensitivity = 0
            specificity = 0
            # Compute the specificity and sensitivity
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp/(tp+fn)
            # Compute the ROC curve and area under the curve (AUC)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            print('==============================Depticon==============================')
            print('Accuracy:', accuracy)
            print('Specificity:', specificity)
            print('Sensitivity:', sensitivity)
            print('ROC AUC:', roc_auc)
            print('==============================Depticon==============================')
            cm_d.plot()
            plt.show()

    def __series(self,y_test, y_ref):
        loss = []
        loss_fn = torch.nn.L1Loss()
        for idx in range(len(y_test)):
            tem = loss_fn(y_test[idx],y_ref[idx])
            loss.append(tem.cpu().detach().numpy())

        loss = np.array(loss)
        print('Max loss: ',np.max(loss))
        print('Mean loss: ',np.mean(loss))
        print('Min loss: ',np.min(loss))
        print('Number of test: ',len(y_test))
        plt.hist(loss, bins=100, edgecolor='black')
        plt.show()