import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from template_gen import temp_find

def dis_perform(data,model,criteria,rate = 0.5,display = ["Latent","Sketch"],device = "cpu",show_num = 20):
  model.eval()
  num_show = 20
  all_inputs = []
  all_outputs = []
  all_latents = []
  all_losses = []
  all_disease = []

  with torch.no_grad():
    for temp,label in data:
      temp = temp.to(device)
      output = model(temp)
      latent = model.hidden
      loss = criteria(output,temp)
      all_inputs.extend(temp.cpu().unbind())
      all_outputs.extend(output.cpu().unbind())
      all_latents.extend(latent.cpu().unbind())
      all_losses.append(loss.cpu())
      all_disease.extend(label[:,1].cpu().tolist())

  all_inputs = torch.stack(all_inputs)
  all_inputs = all_inputs.squeeze(1)
  all_outputs = torch.stack(all_outputs)
  all_outputs = all_outputs.squeeze(1)
  all_latents = torch.stack(all_latents)
  all_losses = torch.stack(all_losses)
  all_disease = torch.tensor(all_disease)

  if "Latent" in display:
    all_latents_flat = torch.flatten(all_latents, start_dim=1)
    latent_np = all_latents_flat.detach().cpu().numpy()
    latent_scaled = (latent_np - latent_np.min()) / (latent_np.max() - latent_np.min() + 1e-8)
    losses_np = all_losses.detach().cpu().numpy()
    loss_norm = (losses_np-losses_np.max()) / (losses_np.max() - losses_np.min() + 1e-8)

    print("Latent shape:", latent_np.shape)

    idx0 = all_disease == 0
    idx1 = all_disease == 1
    # idx2 = all_disease == 2
    # idx3 = all_disease == 3

    pca = PCA(n_components=3)
    latent_nd = pca.fit_transform(latent_np)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance (2D):", np.sum(pca.explained_variance_ratio_))
    sketch = np.transpose(latent_scaled)

    #Latent comparison
    fig, axes = plt.subplots(2,1,figsize=(12,12))
    axes[0].imshow(sketch[:,idx0],aspect='auto',cmap='gray_r',interpolation='nearest')
    axes[1].imshow(sketch[:,idx1],aspect='auto',cmap='gray_r',interpolation='nearest')
    # axes[2].imshow(sketch[:,idx2],aspect='auto',cmap='gray_r',interpolation='nearest')
    # axes[3].imshow(sketch[:,idx3],aspect='auto',cmap='gray_r',interpolation='nearest')
    plt.tight_layout()
    plt.show()

    #Distribution comparison 2D
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.scatter(latent_nd[:,0], latent_nd[:,1],c=loss_norm,cmap='gray_r',s=20)
    # ax1.scatter(losses_np[idx1],bins=30,label="Arrhythmias")
    # ax1.hist(losses_np[idx2],bins=30,label="2")
    # ax1.hist(losses_np[idx3],bins=30,label="3")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.scatter(latent_nd[idx0,0], latent_nd[idx0,1],s=20,label="Healthy")
    ax2.scatter(latent_nd[idx1,0], latent_nd[idx1,1],s=20,label="Arrhythmias")
    # ax2.scatter(latent_nd[idx2,0], latent_nd[idx2,1],s=20,label="2")
    # ax2.scatter(latent_nd[idx3,0], latent_nd[idx3,1],s=20,label="3")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.scatter(latent_nd[idx0,0], latent_nd[idx0,1], latent_nd[idx0,2],s=20,label="Healthy")
    ax3.scatter(latent_nd[idx1,0], latent_nd[idx1,1], latent_nd[idx1,2],s=20,label="Arrhythmias")
    # ax3.scatter(latent_nd[idx2,0], latent_nd[idx2,1], latent_nd[idx2,2],s=20,label="2")
    # ax3.scatter(latent_nd[idx3,0], latent_nd[idx3,1], latent_nd[idx3,2],s=20,label="3")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

  if "Sketch" in display:
    all_outputs_np = all_outputs.detach().cpu().numpy()
    all_inputs_np = all_inputs.detach().cpu().numpy()
    all_disease_np = all_disease
    all_losses_np = all_losses.detach().cpu().numpy()
    latent_np = all_latents.detach().cpu().numpy()
    latent_scaled = (latent_np - latent_np.min()) / (latent_np.max() - latent_np.min() + 1e-8)

    print("Total samples:", len(all_inputs_np))

    K = show_num
    random_idx = np.random.choice(range(len(all_outputs_np)),K,replace=False)
    rows = K//2+1
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()
    for i, idx in enumerate(random_idx):
        input_signal = all_inputs_np[idx]
        output_signal = all_outputs_np[idx]
        dis = str(all_disease_np[idx])
        axes[2*i].plot(input_signal)
        axes[2*i].plot(output_signal)
        axes[2*i].set_title(f"Loss:{all_losses_np[idx]:.4f}|{dis}")
        axes[2*i].set_xlabel("Sample Index")
        # axes[2*i+1].imshow(latent_scaled[random_idx[i]],aspect='auto',cmap='gray_r',interpolation='nearest')

    plt.tight_layout()
    plt.show()

def inference_perform(data,model,criteria,rate = 0.5,display = ["CM","PCA","Sketch"],device = "cpu",show_num = 20):
  model.eval()
  num_show = 20
  all_inputs = []
  all_outputs = []
  all_latents = []
  all_losses = []
  all_labels = []
  all_disease = []
  all_preds = []
  all_corr = []

  with torch.no_grad():
    for temp,label in data:
      temp = temp.to(device)
      output = model(temp)
      latent = model.hidden
      loss = criteria(output,temp)
      all_inputs.extend(temp.cpu().unbind())
      all_outputs.extend(output.cpu().unbind())
      all_latents.extend(latent.cpu().unbind())
      all_losses.append(loss.cpu())
      all_labels.extend(label[:,0].cpu().tolist())

      loss = loss.cpu().numpy()

      if loss > rate:
        all_preds.append(0)
      else:
        all_preds.append(1)

  all_inputs = torch.stack(all_inputs)
  all_inputs = all_inputs.squeeze(1)
  all_outputs = torch.stack(all_outputs)
  all_outputs = all_outputs.squeeze(1)
  all_latents = torch.stack(all_latents)
  all_losses = torch.stack(all_losses)
  all_labels = torch.tensor(all_labels)
  all_preds = torch.tensor(all_preds)

  print("Total validation samples:", len(all_inputs))
  print("Good",len(all_labels[all_labels == 1]))
  print("Bad",len(all_labels[all_labels == 0]))

  if "CM" in display:
    TP = ((all_preds == 1) & (all_labels == 1)).sum().item()
    TN = ((all_preds == 0) & (all_labels == 0)).sum().item()
    FP = ((all_preds == 1) & (all_labels == 0)).sum().item()
    FN = ((all_preds == 0) & (all_labels == 1)).sum().item()
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
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

  if "Latent" in display:
    all_latents_flat = torch.flatten(all_latents, start_dim=1)
    latent_np = all_latents_flat.detach().cpu().numpy()
    latent_scaled = (latent_np - latent_np.min()) / (latent_np.max() - latent_np.min() + 1e-8)
    losses_np = all_losses.detach().cpu().numpy()
    loss_norm = (losses_np-losses_np.max()) / (losses_np.max() - losses_np.min() + 1e-8)

    print("Latent shape:", latent_np.shape)

    good_idx = all_labels == 1
    bad_idx = all_labels == 0

    pg_idx = all_preds == 1
    pb_idx = all_preds == 0

    pca = PCA(n_components=3)
    latent_nd = pca.fit_transform(latent_np)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance (2D):", np.sum(pca.explained_variance_ratio_))
    sketch = np.transpose(latent_scaled)
    fig, axes = plt.subplots(1,2,figsize=(12,2))
    axes[0].imshow(sketch[:,good_idx],aspect='auto',cmap='gray_r',interpolation='nearest')
    axes[1].imshow(sketch[:,bad_idx],aspect='auto',cmap='gray_r',interpolation='nearest')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1,2,figsize=(12,2))
    axes[0].imshow(sketch[:,pg_idx],aspect='auto',cmap='gray_r',interpolation='nearest')
    axes[1].imshow(sketch[:,pb_idx],aspect='auto',cmap='gray_r',interpolation='nearest')
    plt.tight_layout()
    plt.show()

    #Distribution comparison 2D
    fig = plt.figure(figsize=(12, 3))

    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')

    ax1.hist(losses_np[good_idx],bins=30,label="0")
    ax1.hist(losses_np[bad_idx],bins=30,label="1")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.hist(losses_np[pg_idx],bins=30,label="0")
    ax2.hist(losses_np[pb_idx],bins=30,label="1")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3.scatter(latent_nd[good_idx,0], latent_nd[good_idx,1],s=20,label="0")
    ax3.scatter(latent_nd[bad_idx,0], latent_nd[bad_idx,1],s=20,label="1")
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4.scatter(latent_nd[good_idx,0], latent_nd[good_idx,1], latent_nd[good_idx,2],s=20,label="0")
    ax4.scatter(latent_nd[bad_idx,0], latent_nd[bad_idx,1], latent_nd[bad_idx,2],s=20,label="1")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

  if "Sketch" in display:
    all_outputs_np = all_outputs.detach().cpu().numpy()
    all_inputs_np = all_inputs.detach().cpu().numpy()
    all_labels_np = all_labels.detach().cpu().numpy()
    all_preds_np = all_preds.detach().cpu().numpy()
    all_losses_np = all_losses.detach().cpu().numpy()
    latent_np = all_latents.detach().cpu().numpy()
    latent_scaled = (latent_np - latent_np.min()) / (latent_np.max() - latent_np.min() + 1e-8)

    print("Total samples:", len(all_inputs_np))

    K = show_num
    random_idx = np.random.choice(range(len(all_outputs_np)),K,replace=False)
    rows = K//2+1
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()
    for i, idx in enumerate(random_idx):
        input_signal = all_inputs_np[idx]
        output_signal = all_outputs_np[idx]

        if all_labels_np[idx] == 1:
          label = "G"
        else:
          label = "B"

        if all_preds_np[idx] == 1:
          pred = "G"
        else:
          pred = "B"

        axes[2*i].plot(input_signal)
        axes[2*i].plot(output_signal)
        axes[2*i].set_title(f"Loss:{all_losses_np[idx]:.4f}|{label}:{pred}")
        axes[2*i].set_xlabel("Sample Index")
        # axes[2*i+1].imshow(latent_scaled[random_idx[i]],aspect='auto',cmap='gray_r',interpolation='nearest')

    plt.tight_layout()
    plt.show()

def inference(data,model,criteria= nn.MSELoss, rate = 0.5,device = None):
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  sig = data
  adjust = np.zeros(len(sig))
  a = temp_find(sig)
  a.num = 128
  peak,feet,temp = a.temping()
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
        adjust[peak[i]:peak[i+1]] = 1
  return adjust