import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    This is the template for code a model
    para_i is the paramerter i needed to define the model, it can be any type of data structure
    vis is the attribute to store the intermediate feature maps for visualization
    structure_calculate is the function to calculate the structure of the model and print out the details if visualize=True
    visualizer is the function to print out the shapes of the intermediate feature maps stored in vis
    All above should be modified according to the model you want to implement and must have in the code
    The notes in the code is the relative code that follwows the model you want to implement
"""

class model_name(nn.Module):
    #visual feature map
    vis = []

    para = {
        "para_1":         [], #Discription of para 1
        "para_2":         [], #Discription of para 2
    }

    def __init__(self, config):
        super(model_name, self).__init__()

        for key in self.para:
            self.para[key] = config[key]

        self.para_1 = self.para["para_1"]
        self.para_2 = self.para["para_2"]
        
        "Initialize your layers here"

        self.structure_calculate()

        "Define your layers here"
    
    def structure_calculate(self,visualize=False):
        "Calculate the structure of your model here"
        if visualize == True:
            "print out the structure details here"
            pass

    def forward(self, x):
        self.vis.clear()
        out = x
        self.vis.append(out)
        "Define the forward pass here"
        return out

    def visualizer(self):
        for item in self.vis:
            print(item.shape)

if __name__ == "__main__":
    config = {
        "para 1":         800,
        "para 2":           1
    }

    input_tensor = torch.rand((4, 2, 3))
    model = model_name(config)
    output = model(input_tensor)
    model.structure_calculate(True)
    print(f"\nInput shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    model.visualizer()