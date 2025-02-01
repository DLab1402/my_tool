import torch

class Conv1d_batchnorm(torch.nn.Module):
	
	def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = 1, activation = 'relu'):
		super().__init__()
		self.activation = activation
		self.conv1 = torch.nn.Conv1d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = 'same')
		self.batchnorm = torch.nn.BatchNorm1d(num_out_filters)
	
	def forward(self,x):
		x = self.conv1(x)
		x = self.batchnorm(x)
		
		if self.activation == 'relu':
			return torch.nn.functional.relu(x)
		else:
			return x



class MultiResUnet(torch.nn.Module):
	vis = []
	para = {"Number class":         [],
			"Serie length":         [], #The length of the 1D data
            "Filter nums":          [], #The number of filters for each multiresblocks of one side of the model (1xn) 
            "Expand":               [], #The number of stride (1x[n-1])
			"Respath kernel":       [], #The kernel size of each respath (1x1)
            "Block kernel":         [], #The kernel size of each multiresblocks (1x3)
			"Filter rate":          [], #The ratio of filer number of each multiresblock (1x3)
			"Pooling kernel":       [], #The kernel size of each pooling layer (1xn)
			"Transpose kernel":     []	#
    }

	def __init__(self, para, alpha=1.67):
		super().__init__()
		
		self.alpha = alpha
		self.para = para
		
		en_str,bn,de_str,resp,ups = self.structure_calculate(False)

	
	def structure_calculate(self,visualize=False):
		return

	def forward(self, x : torch.Tensor)->torch.Tensor:
		return
	
	def visualizer(self):
		for item in self.vis:
			print(item.shape)

#Test
if __name__ == "__main__":
	para = {#
    }

	input_tensor = torch.rand((4, 1, 1024))
	model = MultiResUnet(para = para, alpha = 1.67)
	model.structure_calculate(True)
	y = model.forward(input_tensor)
	for p in model.parameters():
		print(p.numel())