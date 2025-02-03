import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import os
from torch.utils.data import Subset
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Simple training
class simple_training:
    def __init__(self,training_data,validate_data = None,criterion = nn.CrossEntropyLoss()):
        self.training_data = training_data
        self.validate_data = validate_data
        self.criterion = criterion

    def training(self,model,device,optimizer,num_epochs=1,batch = 1,visualize = 1):
        train_loss = 0
        val_loss = 0
        for epoch in range(num_epochs):
            train_dataloader = torch.utils.data.DataLoader(self.training_data, batch_size=batch,shuffle=True)
            if hasattr(model,'trainer'):
                train_loss = model.trainer(device, self.criterion, train_dataloader, optimizer)
            else:
                train_loss = self.__train_epoch(model, device, self.criterion, train_dataloader, optimizer)
            
            if self.validate_data != None:
                val_dataloader = torch.utils.data.DataLoader(self.validate_data,batch_size=1,shuffle=True)
                if hasattr(model,'validator'):
                    val_loss = model.validator(device, self.criterion,val_dataloader)
                else:
                    val_loss = self.__val_epoch(model, device, self.criterion, val_dataloader)
            if (visualize == 1):
                # if isinstance():
                print('Iteration: {}/{}. loss_train: {}. loss_val: {}'.format(epoch+1, num_epochs,train_loss, val_loss))

    def __train_epoch(self, model, device, loss_fn, data, optimizer):
        train_loss = 0.0
        model.to(device)
        model.train()
        
        for inputs, labels in data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach().numpy()
        train_loss = train_loss / len(data)
        return train_loss 

    def __val_epoch(self,model, device, loss_fn, data):
        val_loss = 0.0
        model.eval()
        total_labels = []
        total_ouputs = []
        with torch.no_grad():
            for inputs, labels in data:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs,labels)
                val_loss += loss.cpu().detach().numpy()
                total_labels.append(labels[0])
                total_ouputs.append(outputs[0])
            val_loss = val_loss / len(data)
        return val_loss

#Validation training
class k_fold_training:
    __fold = None
    def __init__(self,train_data,k,criterion = nn.CrossEntropyLoss()):
        self.train_data = train_data
        self.k = k
        self.criterion = criterion
        if self.k > 1:
            self.fold_gen(True,42)

    def fold_gen(self,shuff,r_state):
        if self.k > 1:
            splits = KFold(n_splits=self.k,shuffle=shuff,random_state=r_state)
            self.__fold = splits.split(self.train_data)

    def training(self,model,device,optimizer,num_epochs=1,batch = 1,visualize = 1):
        if self.k > 1:
            actor = simple_training(self.train_data, criterion=self.criterion)
            for fold, (train_idx,val_idx) in enumerate(self.__fold):
                fmodel = model.to(device)
                for layer in fmodel.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

                if (visualize == 1):
                    print("Fold :",fold+1)
                    print("Train and validate dataset are overlap: ",any(x in val_idx for x in train_idx))
                actor.training_data = Subset(self.train_data, train_idx)
                actor.validate_data = Subset(self.train_data, val_idx)
                actor.training(model,device,optimizer,num_epochs,batch,visualize)

        else:
            model.to(device)
            print(next(model.parameters()).is_cuda)
            actor = simple_training(self.train_data, criterion=self.criterion)
            actor.training_data = self.train_data
            actor.training(model,device,optimizer,num_epochs,batch,visualize)