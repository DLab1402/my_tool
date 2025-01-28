import torch
import torch.nn as nn

class orgininal_gan(nn.Module):
    def __init__(self, gen_dims, dis_dims, gen_activation_fn=nn.ReLU, dis_activation_fn=nn.LeakyReLU):
        super(orgininal_gan, self).__init__()
        
        gen_layers = []
        
        for i in range(len(gen_dims)-1):
            gen_layers.append(nn.Linear(gen_dims[i], gen_dims[i+1]))
            gen_layers.append(gen_activation_fn())
        
        gen_layers.append(nn.Tanh())  # Final activation for output layer
        self.generator = nn.Sequential(*gen_layers)

        dis_layers = []
        
        for i in range(len(dis_dims)-1):
            dis_layers.append(nn.Linear(dis_dims[i], dis_dims[i+1]))
            dis_layers.append(dis_activation_fn(0.2))  # LeakyReLU with slope of 0.2
        
        dis_layers.append(nn.Linear(dis_layers[len(dis_layers)-1],1))
        dis_layers.append(nn.Sigmoid())  # Final activation for binary classification
        self.discriminator = nn.Sequential(*dis_layers)


    def forward(self, gen_input = None, dis_input = None):
        if gen_input != None and dis_input != None:
            return self.generator(gen_input), self.discriminator(dis_input)
        if gen_input == None and dis_input != None:
            return self.generator(gen_input)
        if gen_input != None and dis_input == None:
            return self.discriminator(dis_input)

    def trainer(self,device, criterion, train_dataloader, optimizer):
        if isinstance(optimizer,list):
            gen_optimizer = optimizer[0]
            dis_optimizer = optimizer[1]
        gen_train_loss = 0.0
        dis_train_loss = 0.0

        self.generator.train()
        self.discriminator.train()
        
        for inputs, labels in train_dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            #======generator training=========
            gen_optimizer.zero_grad()
            fake_data = self.generator.forward(gen_input = inputs)
            gen_loss = criterion(fake_data,labels)
            gen_loss.backward()
            gen_optimizer.step()
            gen_train_loss += gen_loss.cpu().detach().numpy()

            #======discriminator training=====
            dis_optimizer.zero_grad()
            
        
        gen_train_loss = gen_train_loss / len(train_dataloader)
        dis_train_loss = dis_train_loss / len(train_dataloader)
        return gen_train_loss 

    def validator(self,device, criterion, val_dataloader):
        pass

# Example usage
z_dim = 100
gen_hidden_dims = [128, 256, 512]
disc_hidden_dims = [512, 256, 128]

model = orgininal_gan(gen_dims = gen_hidden_dims, dis_dims = disc_hidden_dims)

print(model.generator)
print(model.discriminator)