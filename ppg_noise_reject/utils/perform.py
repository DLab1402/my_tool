import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from template_gen import temp_find

def model_calculate(data,model,criteria,device = "cpu",rate = 0.5):
    model.eval()
    inputs = []
    outputs = []
    latents = []
    losses = []
    disease = []
    preds = []
    labels = []

    with torch.no_grad():
        for temp,label in data:
            temp = temp.to(device)
            output = model(temp)
            if hasattr(model, "hidden"):
                latent = model.hidden
                latents.extend(latent.cpu().unbind())

            loss = criteria(output,temp)
            inputs.extend(temp.cpu().unbind())
            outputs.extend(output.cpu().unbind())
            
            losses.append(loss.cpu())
            if len(label) > 1:
                disease.extend(label[:,1].cpu().tolist())

            labels.extend(label[:,0].cpu().tolist())

            loss = loss.cpu().numpy()

            if loss > rate:
                preds.append(0)
            else:
                preds.append(1)
    
    inputs = torch.stack(inputs)
    inputs = inputs.squeeze(1)
    inputs = inputs.detach().cpu().numpy()
    outputs = torch.stack(outputs)
    outputs = outputs.squeeze(1)
    outputs = outputs.detach().cpu().numpy()
    latents = torch.stack(latents)
    latents = latents.detach().cpu().numpy()
    losses = torch.stack(losses)
    losses = losses.detach().cpu().numpy()
    disease = np.array(disease)
    preds = np.array(preds)
    lables = np.array(labels)

    return inputs,outputs,latents,losses,labels,preds,disease

def CM(labels,preds):
    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()
    # cm = confusion_matrix(labels.numpy(), preds.numpy())
    print("TP:",TP)
    print("TN:",TN)
    print("FP:",FP)
    print("FN:",FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

def latent_sketch(latents,losses,disease):
    latents = (latents - latents.min()) / (latents.max() - latents.min())
    losses = (losses-losses.max()) / (losses.max() - losses.min())

    print("Latent shape:", latents.shape)

    idx0 = disease == 0
    idx1 = disease == 1

    pca = PCA(n_components=3)
    latent_nd = pca.fit_transform(latents)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance (2D):", np.sum(pca.explained_variance_ratio_))
    sketch = np.transpose(latents)

    #Latent comparison
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].imshow(sketch[:,idx0],aspect='auto',cmap='gray_r',interpolation='nearest')
    axes[1].imshow(sketch[:,idx1],aspect='auto',cmap='gray_r',interpolation='nearest')
    plt.tight_layout()
    plt.show()

    #Distribution comparison 2D
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.scatter(latent_nd[:,0], latent_nd[:,1],c=losses,cmap='gray_r',s=20)
    ax1.grid(alpha=0.3)

    ax2.scatter(latent_nd[idx0,0], latent_nd[idx0,1],s=20,label="Healthy")
    ax2.scatter(latent_nd[idx1,0], latent_nd[idx1,1],s=20,label="Arrhythmias")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.scatter(latent_nd[idx0,0], latent_nd[idx0,1], latent_nd[idx0,2],s=20,label="Healthy")
    ax3.scatter(latent_nd[idx1,0], latent_nd[idx1,1], latent_nd[idx1,2],s=20,label="Arrhythmias")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def single_sketch(inputs,outputs,labels,preds,losses,disease = None,show_num = 20):
    K = show_num
    random_idx = np.random.choice(range(len(outputs)),K,replace=False)
    rows = K//4
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()
    for i, idx in enumerate(random_idx):
        input_signal = inputs[idx]
        output_signal = outputs[idx]
        pre = preds[idx]
        lab = labels[idx]
        if disease is not None:
            dis = str(disease[idx])
            axes[i].set_title(f"Loss:{losses[idx]:.4f}|{dis}|{lab}:{pre}")
        else:
            axes[i].set_title(f"Loss:{losses[idx]:.4f}|{lab}:{pre}")
        axes[i].plot(input_signal)
        axes[i].plot(output_signal)

    plt.tight_layout()
    plt.show()

def dis_perform(data,model,criteria,rate = 0.5,display = ["Latent","Sketch"],device = "cpu",show_num = 20):
    model.eval()
    inputs,outputs,latents,losses,labels,preds,disease = model_calculate(data,model,criteria = criteria,device = device,rate = rate)
    if "Latent" in display:
        latent_sketch(latents,losses,disease)
    if "Sketch" in display:
        single_sketch(inputs,outputs,labels,preds,losses,disease = disease,show_num=show_num)

def inference_perform(data,model,criteria,rate = 0.5,display = ["CM","PCA","Sketch"],device = "cpu",show_num = 20):
    model.eval()
    inputs,outputs,latents,losses,labels,preds,_ = model_calculate(data,model,criteria = criteria,device = device,rate = rate)
    if "CM" in display:
        CM(labels,preds)
    if "PCA" in display:
        latent_sketch(latents,losses,disease = preds)
    if "Sketch" in display:
        single_sketch(inputs,outputs,labels,preds,losses,show_num=show_num)

def inference(data,model,criteria= nn.MSELoss, rate = 0.5,device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sig = data
    adjust = np.zeros(len(sig))
    a = temp_find(sig)
    a.num = 128
    _,feet,temp = a.temping()
    model.eval()
    with torch.no_grad():
        for i,item in enumerate(temp):
            if len(item) < 128:
                # Pad with zeros if the signal is shorter
                item_tensor = torch.tensor(item, dtype=torch.float32) # Specify dtype for item
                padding = torch.zeros(128 - len(item), dtype=torch.float32) # Use torch.float32 for padding
                item1 = torch.cat((item_tensor, padding), 0)
            elif len(item) > 128:
                # Crop if the signal is longer
                item1 = torch.tensor(item[:128], dtype=torch.float32) # Specify dtype for item
            else:
                item1 = torch.tensor(item, dtype=torch.float32) # Specify dtype for item

            min_val = item1.min()
            max_val = item1.max()
            item1 = (item1 - min_val) / (max_val - min_val + 1e-8)
            item1 = item1.unsqueeze(0)
            tensor = item1.unsqueeze(0).to(device)
            loss = criteria(model(tensor),tensor)
            err = loss.item()
            if err > rate:
                adjust[feet[i]:feet[i+1]] = 1
    return adjust