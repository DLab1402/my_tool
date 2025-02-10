import torch

class Conv1d_batchnorm(torch.nn.Module):
	'''
	1D Convolutional layers

	Arguments:
		num_in_filters {int} -- number of input filters
		num_out_filters {int} -- number of output filters
		kernel_size {tuple} -- size of the convolving kernel
		stride {tuple} -- stride of the convolution (default: {(1, 1)})
		activation {str} -- activation function (default: {'relu'})
	'''

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


class Multiresblock(torch.nn.Module):
	'''
	MultiRes Block
	
	Arguments:
		num_in_channels {int} -- Number of channels coming into mutlires block
		num_filters {int} -- Number of filters in a corrsponding UNet stage
		alpha {float} -- alpha hyperparameter (default: 1.67)
	'''

	def __init__(self, num_in_channels, num_filters, filter_rate = [1.67, 0.333, 0.5], kernel_size = [3,5,7]):
	
		super().__init__()
		self.alpha = filter_rate[0]
		self.W = num_filters * filter_rate[0]
		
		filt_cnt_1 = int(self.W*filter_rate[0])
		filt_cnt_2 = int(self.W*filter_rate[1])
		filt_cnt_3 = int(self.W*filter_rate[2])
		num_out_filters = filt_cnt_1 + filt_cnt_2 + filt_cnt_3
		
		self.shortcut = Conv1d_batchnorm(num_in_channels ,num_out_filters , kernel_size = 1, activation='None')

		self.conv_1 = Conv1d_batchnorm(num_in_channels, filt_cnt_1, kernel_size = kernel_size[0], activation='relu')

		self.conv_2 = Conv1d_batchnorm(filt_cnt_1, filt_cnt_2, kernel_size = kernel_size[1], activation='relu')
		
		self.conv_3 = Conv1d_batchnorm(filt_cnt_2, filt_cnt_3, kernel_size = kernel_size[2], activation='relu')

		self.batch_norm1 = torch.nn.BatchNorm1d(num_out_filters)
		self.batch_norm2 = torch.nn.BatchNorm1d(num_out_filters)

	def forward(self,x):

		shrtct = self.shortcut(x)
		
		a = self.conv_1(x)
		b = self.conv_2(a)
		c = self.conv_3(b)

		x = torch.cat([a,b,c],axis=1)
		x = self.batch_norm1(x)

		x = x + shrtct
		x = self.batch_norm2(x)
		x = torch.nn.functional.relu(x)
	
		return x


class Respath(torch.nn.Module):
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
	'''

	def __init__(self, num_in_filters, num_out_filters, respath_length, kernel = 3):
	
		super().__init__()

		self.respath_length = respath_length
		self.shortcuts = torch.nn.ModuleList([])
		self.convs = torch.nn.ModuleList([])
		self.bns = torch.nn.ModuleList([])

		for i in range(self.respath_length):
			if(i==0):
				self.shortcuts.append(Conv1d_batchnorm(num_in_filters, num_out_filters, kernel_size = 1, activation='None'))
				self.convs.append(Conv1d_batchnorm(num_in_filters, num_out_filters, kernel_size = kernel,activation='relu'))

				
			else:
				self.shortcuts.append(Conv1d_batchnorm(num_out_filters, num_out_filters, kernel_size = 1, activation='None'))
				self.convs.append(Conv1d_batchnorm(num_out_filters, num_out_filters, kernel_size = kernel, activation='relu'))

			self.bns.append(torch.nn.BatchNorm1d(num_out_filters))
		
	
	def forward(self,x):

		for i in range(self.respath_length):

			shortcut = self.shortcuts[i](x)

			x = self.convs[i](x)
			x = self.bns[i](x)
			x = torch.nn.functional.relu(x)

			x = x + shortcut
			x = self.bns[i](x)
			x = torch.nn.functional.relu(x)

		return x


class MultiResUnet(torch.nn.Module):
	'''
	MultiResUNet
	
	Arguments:
		input_channels {int} -- number of channels in image
		num_classes {int} -- number of segmentation classes
		alpha {float} -- alpha hyperparameter (default: 1.67)
	
	Returns:
		[keras model] -- MultiResUNet model
	'''
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

		self.encoder_layer = torch.nn.ModuleList([])
		self.decoder_layer = torch.nn.ModuleList([])
		self.respath_layer = torch.nn.ModuleList([])
		self.ups_layer = torch.nn.ModuleList([])
		self.pooling = torch.nn.ModuleList([])

		L = len(en_str)

		for i in range(len(en_str)):
			self.encoder_layer.append(Multiresblock(en_str[i][0],self.para["Filter nums"][i]))
			self.pooling.append(torch.nn.MaxPool1d(2))
			self.respath_layer.append(Respath(resp[i][0],resp[i][1],respath_length=4))

		self.bottle_neck = Multiresblock(bn[0],self.para["Filter nums"][-1])

		for i in range(len(de_str)):
			self.decoder_layer.append(Multiresblock(de_str[i][0],self.para["Filter nums"][L-i-1]))
			self.ups_layer.append(torch.nn.ConvTranspose1d(ups[i][0],ups[i][1],kernel_size=2,stride=2))

		self.conv_final = Conv1d_batchnorm(de_str[-1][1], self.para["Number class"], kernel_size = 1, activation='None')
	
	def structure_calculate(self,visualize=False):
		out_filter = lambda n:int(n*self.alpha*self.para["Filter rate"][0])+int(n*self.alpha*self.para["Filter rate"][1])+int(n*self.alpha*self.para["Filter rate"][2])
		
		out = [out_filter(item) for item in self.para["Filter nums"]]

		E_MRBs = []
		D_MRBs = []
		ResPaths = []
		ups = []
		
		L = len(self.para["Filter nums"])

		for i in range(L-1):
			if i == 0:
				E_MRBs.append([1,out[i]])
			else:
				E_MRBs.append([out[i-1],out[i]])
			
			ResPaths.append([out[i],out[i]])
		
		BNs = [out[L-2],out[L-1]]

		for i in range(L-1):
			if i == 0:
				ups.append([out[L-1],self.para["Filter nums"][L-i-2]])
			else:
				ups.append([D_MRBs[i-1][1],self.para["Filter nums"][L-i-2]])		

			D_MRBs.append([ups[-1][1]+ResPaths[L-i-2][1],out[L-i-2]])
		
		if visualize == True:
			print(out)
			print(E_MRBs)
			print(BNs)
			print(D_MRBs)
			print(ResPaths)
			print(ups)
		
		return [E_MRBs,BNs,D_MRBs,ResPaths,ups]

	def forward(self, x : torch.Tensor)->torch.Tensor:
		e = []
		r = []
		d = []

		self.vis.clear()

		L = len(self.encoder_layer)

		for i in range(len(self.encoder_layer)):
			if i == 0:
				r.append(self.encoder_layer[i](x))
			else:
				r.append(self.encoder_layer[i](e[-1]))
			
			e.append(self.pooling[i](r[-1]))
			self.vis.append(e[-1])
		
		b = self.bottle_neck(e[-1])
		self.vis.append(b)

		for i in range(len(self.decoder_layer)):
			if i == 0:
				up = torch.cat([self.ups_layer[i](b),r[L-i-1]],axis=1)
			else:
				up = torch.cat([self.ups_layer[i](d[-1]),r[L-i-1]],axis=1)
			
			d.append(self.decoder_layer[i](up))
			self.vis.append(d[-1])

		out = self.conv_final(d[-1])
		self.vis.append(out)
		return out
	
	def visualizer(self):
		for item in self.vis:
			print(item.shape)

#Test
if __name__ == "__main__":
	para = {"Number class":         3,
			"Serie length":         1024, #The length of the 1D data
            "Filter nums":          [32,64,100,128], #The number of filters for each multiresblocks of one side of the model (1xn) 
            "Expand":               [2,2,2,1], #The number of stride (1x[n-1])
			"Respath kernel":       3, #The kernel size of each respath (1x1)
            "Block kernel":         [3,5,7], #The kernel size of each multiresblocks (1x3)
			"Filter rate":          [1.67,0.333,0.5], #The ratio of filer number of each multiresblock (1x3)
			"Pooling kernel":       [2,2,2,2], #The kernel size of each pooling layer (1xn)
			"Transpose kernel":     [2,2,2,2]	#
    }

	input_tensor = torch.rand((4, 1, 1024))
	model = MultiResUnet(para = para, alpha = 1.67)
	model.structure_calculate(True)
	y = model.forward(input_tensor)
	# for p in model.parameters():
	# 	print(p.numel())