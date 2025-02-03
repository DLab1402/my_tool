import torch

class CNNA(torch.nn.Module):
	vis = []
	encoder_layers = torch.nn.ModuleList()
	decoder_layers = torch.nn.ModuleList()

	para = {"BottleNeck":       None, # an array (1,l) or a number
		    "Structure":        None, # an array (1,N)
			"Encode acfnc":     None, # a string 
			"Decode acfnc":     None, # a string
			"Kernel size":      None, # an array (1,2*(N-1))
			"Stride":           None, # an array (1,2*(N-1))
			"Padding":          None, # an array (1,2*(N-1))
			"Pooling":          None, # ["<name>",[k,s]]
    }

	def __init__(self, para):
		super().__init__()
		self.para = para
		self.layer_make()
		self.en_pooling = self.pooling_chose()
		self.de_pooling = self.pooling_chose()
		self.en_acfnc = self.acfnc_chose(self.para["Encode acfnc"])
		self.de_acfnc = self.acfnc_chose(self.para["Decode acfnc"])
		
	def layer_make(self):
		L = len(self.para["Structure"])
		for i in range(L-1):
			self.encoder_layers.append(torch.nn.Conv2d(self.para["Structure"][i], 
														self.para["Structure"][i+1], 
														kernel_size = self.para["Kernel size"][i],
														stride = self.para["Stride"][i],
														padding = self.para["Padding"][i]))
			
			self.decoder_layers.append(torch.nn.ConvTranspose2d(self.para["Structure"][L-i-1], 
														self.para["Structure"][L-i-2], 
														kernel_size = self.para["Kernel size"][L-1+i],
														stride = self.para["Stride"][L-1+i],
														padding = self.para["Padding"][L-1+i]))	

		if isinstance(self.para["BottleNeck"], int):
			self.bottle_neck = torch.nn.Conv2d(self.para["Structure"][L-1], 
														self.para["Structure"][L-1], 
														kernel_size = (1,1),
														stride = 1,
														padding = 0)
		elif isinstance(self.para["BottleNeck"], list):
			self.bottle_neck = torch.nn.ModuleList()
			for i in range(len(self.para["BottleNeck"])-1):
				self.bottle_neck.append(torch.nn.Linear(self.para["BottleNeck"][i],self.para["BottleNeck"][i+1]))
		else:
			return self.para["BottleNeck"]

	def pooling_chose(self):
		if self.para["Pooling"][0] == "Average":
			return torch.nn.AvgPool2d(kernel_size= self.para["Pooling"][1][0],stride = self.para["Pooling"][1][1],padding = self.para["Pooling"][1][2])
		else:
			return torch.nn.MaxPool2d(kernel_size= self.para["Pooling"][1][0],stride = self.para["Pooling"][1][1],padding = self.para["Pooling"][1][2])
	
	def acfnc_chose(self,name):
		if name == "ReLU":
			return torch.nn.ReLU()
		elif name == "LeakyReLU":
			return torch.nn.LeakyReLU(0.2)
		elif name == "Tanh":
			return torch.nn.Tanh()
		else:
			return torch.nn.Sigmoid()

	def forward(self, x):
		self.vis.clear()
		out = x
		self.vis.append(out)

		for i in range(len(self.encoder_layers)):
			out = self.encoder_layers[i](out)
			out = self.en_acfnc(self.en_pooling(out))
			self.vis.append(out)

		if isinstance(self.para["BottleNeck"], int):
			out = self.bottle_neck(out)
			self.vis.append(out)
		elif isinstance(self.para["BottleNeck"], list):
			for i in range(len(self.para["BottleNeck"])-1):
				out = self.bottle_neck[i](out)
				self.vis.append(out)
		else:
			out = self.bottle_neck(out)
			self.vis.append(out)

		for i in range(len(self.decoder_layers)):
			out = self.decoder_layers[i](out)
			out = self.de_acfnc(self.de_pooling(out))
			self.vis.append(out)

		return out
	
	def visualizer(self):
		for item in self.vis:
			print(item.shape)

#Test
# if __name__ == "__main__":
# 	para = {"BottleNeck":       1, # an array (1,l) or a number
# 		    "Structure":        [3,8,16,32,32], # an array (1,N)
# 			"Encode acfnc":    	"Tanh", # a string 
# 			"Decode acfnc":     "Tanh", # a string
# 			"Kernel size":      [(4,4),(4,4),(2,2),(2,2),(2,2),(2,2),(4,4),(4,4)], # an array (1,N-1)
# 			"Stride":           [(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2)], # an array (1,N-1)
# 			"Padding":          [(1,1),(1,1),(0,0),(0,0),(0,0),(0,0),(1,1),(1,1)], # an array (1,N-1)
# 			"Pooling":          ["Average",[1,1]]
#     }

# 	input_tensor = torch.rand((8,3,384,240))
# 	model = CNNA(para = para)
# 	y = model.forward(input_tensor)
# 	model.visualizer()
# 	# for p in model.parameters():
# 	# 	print(p.numel())