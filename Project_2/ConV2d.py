class Layer():
    def __init__(self):
        self.input_ = None
        self.output_ = None
    def forward_prop(self,input_):
        raise NotImplementedError
    
    def backward_prop(self,error,learning_rate):
        raise NotImplementedError
        
class ConV2D(Layer):
    def __init__(self,filters,kernel_size,stride,pad):
        self.input_ = None
        self.kernel_size = kernel_size
        self.example = None
        self.n_h_img = None
        self.n_w_img = None
        self.n_c_img = None
        self.weights_ = None
        self.bias_ = None
        self.stride = stride
        self.pad = pad
        
    def zero_pad(self,X,pad):
        return np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),mode="constant",constant_values = 0)
    
    def single_conv(self,W,b,X):
        return np.sum(X*W) + float(b)
    
    
    
    def forward_prop(self,input_):
        np.random.seed(101)
        self.input_ = input_
        self.example = self.input_.shape[0]
        self.n_h_img = self.input_.shape[1]
        self.n_w_img = self.input_.shape[2]
        self.n_c_img = self.input_.shape[3]
        self.weights_ = np.random.randn(filters,kernel_size[0],kernel_size[1],self.n_c_img)/np.sqrt(kernel_size[0])
        self.bias_ = np.random.randn(filters,1,1,1)
        
        
        n_c,f,f,n_c_prev = self.weights_.shape
        
        n_h = (self.n_h_img-f+2*self.pad)//stride + 1
        n_w = (self.n_w_img-f+2*self.pad)//stride + 1
        
        ouput = np.zeros((self.example,n_h,n_w,n_c))
        
        input_pad = self.zero_pad(self.input_,self.pad)
        
        for i in range(self.example):
            input_prev = input_pad[i]
            
            for h in range(n_h):
                vert_start = h*self.stride
                vert_stop = vert_start + f
                
                for w in range(n_w):
                    hor_start = w*self.stride
                    hor_stop = hor_start + f
                    
                    for c in range(n_c):
                        input_slice = input_prev[vert_start:vert_stop,hor_start:hor_stop,:]
                        W = self.weights_[c] 
                        b = self.bias_[c]
                        ouput[i,h,w,c] = self.single_conv(W,b,input_slice)
        self.output_ = ouput
        return self.output_
    
    def backward_prop(self,error,learning_rate,optimizer=None,t=None,regu=None):
        output_error = np.zeros_like(self.input_)
        
        dW = np.zeros_like(self.weights_)
        db = np.zeros_like(self.bias_)
        
        f,f = self.kernel_size
        
        (m, n_h, n_w, n_c) = error.shape
        input_pad = self.zero_pad(self.input_,self.pad)
        output_error_pad = self.zero_pad(output_error,self.pad)
        
        for i in range(self.example):
            
            for h in range(n_h):
                vert_str = h*stride
                vert_stp = vert_str + f
                
                for w in range(n_w):
                    hor_start = w*stride
                    hor_stop = hor_start+f
                    
                    for c in range(n_c):
                        output_error_pad[i,vert_str:vert_stp,hor_start:hor_stop,:] += self.weights_[c]*error[i,h,w,c]
                        dW[c] += input_pad[i,vert_str:vert_stp,hor_start:hor_stop,:]*error[i,h,w,c]
                        db[c] += error[i,h,w,c]
            
            output_error[i] = output_error_pad[i,self.pad:-self.pad, self.pad:-self.pad, :]
        
        self.weights_ -= learning_rate*dW
        self.bias_ -= learning_rate*db
        
        return output_error
    
class Pool(Layer):
    def __init__(self,kernel_size,stride,type_):   
        self.input_ = None
        self.kernel_size = kernel_size
        self.example = None
        self.n_h_img = None
        self.n_w_img = None
        self.n_c_img = None
        self.stride = stride
        self.type_ = type_
    
    def forward_prop(self,input_):
        
        self.input_ = input_
        self.example = self.input_.shape[0]
        self.n_h_img = self.input_.shape[1]
        self.n_w_img = self.input_.shape[2]
        self.n_c_img = self.input_.shape[3]
        
        f,f = self.kernel_size
        
        n_h = (self.n_h_img-f)//self.stride + 1
        n_w = (self.n_w_img-f)//self.stride + 1
        n_c = self.n_c_img
        
        output = np.zeros((self.example,n_h,n_w,n_c))
        
        for i in range(self.example):
            input_prev = self.input_[i]
            for h in range(n_h):
                vert_start = h*stride
                vert_stop = vert_start + f
                for w in range(n_w):
                    hor_start = w*stride
                    hor_stop = hor_start + f
                    for c in range(n_c):
                        input_slice = input_prev[vert_start:vert_stop,hor_start:hor_stop,c]
                        if self.type_ == "Max":
                            output[i,h,w,c] = np.max(input_slice)
                        elif self.type_ == "Mean":
                            output[i,h,w,c] = np.mean(input_slice)
        
        
        self.output = output
        return self.output
    
    def mask(self,X):
        X = X==np.max(X)
        return X
    
    def avg_value(self,error,shape):
        n_h,n_w = shape
        return  (np.ones_like(X)/(n_h*n_w))*error
    
    def backward_prop(self,error,learning_rate,optimizer=None,t=None,regu=None):
        
        output_error = np.zeros_like(self.input_)
        
        f,f = self.kernel_size
        (m, n_h, n_w, n_c) = error.shape
        
        for i in range(self.example):
            for h in range(n_h):
                
                vert_start = h*self.stride
                vert_stop = vert_start+f
                
                for w in range(n_w):
                    
                    hort_start = w*self.stride
                    hort_stop = hort_start+f
                    
                    for c in range(n_c):
                        
                        input_slice = self.input_[i,vert_start:vert_stop,hort_start:hort_stop,c]
                        
                        if self.type_ == "Max":
                            output_error[i,vert_start:vert_stop,hort_start:hort_stop,c] += self.mask(input_slice)*error[i,h,w,c]
                        if self.type_ == "Mean":
                            shape = f,f
                            output_error[i,vert_start:vert_stop,hort_start:hort_stop,c] = self.avg_value(error[i,h,w,c],shape)
        
        return output_error

