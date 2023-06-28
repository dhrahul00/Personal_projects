import numpy as np

class Layer():
    def __init__(self):
        self.input_ = None 
        self.output_ = None 
        
    def forward_prop(self,input_):
        raise NotImplementedError
    
    def backward_prop(self,error,learning_rate,optimizer,t,encoder,b,lamda):
        raise NotImplementedError
        
class Optimizer:
    def NAG(self,weights,prev_weights,gamma):
        weight_look_ahead = weights - gamma*prev_weights
        return weight_look_ahead

    def ADAGRAD(self,dw,prev_weights):
        return prev_weights + dw**2

    def RMSprop(self,dw,prev_weights,beta):
        return beta*prev_weights+(1-beta)*dw**2

    def ADAM(self,dw,m,v,beta1,beta2):
        m = beta1*m + (1-beta1)*dw
        v = beta2*v + (1-beta2)*dw**2
        return m,v

class FC_layer(Layer):
    def __init__(self,input_size,output_size):
        np.random.seed(101)
        self.weights_ = np.random.randn(output_size,input_size)
        self.bias_ = np.zeros((1,output_size))
        self.prev_weights_ = np.zeros_like(self.weights_)
        self.prev_bias_ = self.bias_.copy()
        self.m_w = self.prev_weights_.copy()
        self.v_w = self.prev_weights_.copy()
        self.m_b = self.prev_bias_.copy()
        self.v_b = self.prev_bias_.copy()
        
    def forward_prop(self,input_):
        self.input_ = input_
        self.output_ = self.input_@self.weights_.T + self.bias_
        return self.output_
    
    def backward_prop(self,error,learning_rate,optimizer,t,regu): 
        
        lamda,b = regu
        dw = error.T@self.input_ + 2*lamda*self.weights_
        db = np.sum(error,axis=0,keepdims=True)
        
        if optimizer == "NAG":
            gamma = 0.9
            weight_look_ahead = Optimizer().NAG(weights=self.weights_,prev_weights=self.prev_weights_,gamma=gamma)
            input_error =  error@weight_look_ahead
            
            weights_update = gamma*self.prev_weights_ + dw * learning_rate
            bias_update = gamma*self.prev_bias_ + db * learning_rate
            
            self.weights_ -= weights_update
            self.bias_ -= bias_update
            
            self.prev_weights_ = weights_update
            self.prev_bias_ = bias_update
            
        if optimizer == "ADAGRAD":
            tol = 1e-06
            input_error = error@self.weights_
            self.prev_weights_ = Optimizer().ADAGRAD(dw=dw,prev_weights=self.prev_weights_)
            weights_update = (learning_rate*dw)/(np.sqrt(self.prev_weights_+tol))
            
            self.prev_bias_ = Optimizer().ADAGRAD(dw=db,prev_weights=self.prev_bias_)
            bias_update = (learning_rate*db)/(np.sqrt(self.prev_bias_+tol))
            
            self.weights_ -= weights_update
            self.bias_ -= bias_update
         
        if optimizer == "RMSprop":
            tol = 1e-06
            beta = 0.95
            input_error = error@self.weights_
            self.prev_weights_ = Optimizer().RMSprop(dw=dw,prev_weights=self.prev_weights_,beta=beta)
            self.prev_bias_ = Optimizer().RMSprop(dw=db,prev_weights=self.prev_bias_,beta=beta)
            
            weights_update = (learning_rate*dw)/np.sqrt(self.prev_weights_+tol)
            bias_update = (learning_rate*db)/np.sqrt(self.prev_bias_+tol)
            
            self.bias_ -= bias_update
            self.weights_ -= weights_update
        
        if optimizer == "ADAM":
            beta1 = 0.9 
            beta2 = 0.999
            tol = 1e-06
            input_error = error@self.weights_
            self.m_w,self.v_w = Optimizer().ADAM(dw=dw,m=self.m_w,v=self.v_w,beta1=beta1,beta2=beta2)
            self.m_b,self.v_b = Optimizer().ADAM(dw=db,m=self.m_b,v=self.v_b,beta1=beta1,beta2=beta2)
            
            m_hat_w = self.m_w/(1-beta1**t)
            v_hat_w = self.v_w/(1-beta2**t)
            
            m_hat_b = self.m_b/(1-beta1**t)
            v_hat_b = self.v_b/(1-beta2**t)
            
            weights_update = (learning_rate*m_hat_w)/(np.sqrt(v_hat_w+tol))
            bias_update = (learning_rate*m_hat_b)/(np.sqrt(v_hat_b+tol))
            
            self.weights_ -= weights_update
            self.bias_ -= bias_update
            
        if optimizer == None:
            input_error =  error@self.weights_ 
            self.weights_ -= learning_rate*dw
            self.bias_ -= learning_rate*db
        
        return input_error

class AC_layer(Layer):
    def __init__(self,activation,activation_dr):
        self.act = activation
        self.act_dr = activation_dr
        
    def forward_prop(self,input_):
        self.input_ = input_
        self.output_ = self.act(self.input_)
        return self.output_
    
    def backward_prop(self,error,learning_rate,optimizer,t,regu):
        lamda,b = regu
        if b:
            roh = 0.05
            roh_hat = np.mean(self.output_,axis = 0)
            #print(roh_hat,self.output_.shape)
            return error*self.act_dr(self.input_) + b*(-(roh/roh_hat)+((1-roh)/(1-roh_hat)))
        else:
            return error*self.act_dr(self.input_)
        
class Activation:
    
    def sigmoid(self,z):
        pos = z >= 0
        neg = z < 0
        low = np.ones_like(z)
        low[pos] += np.exp(-z[pos])
        low[neg] += np.exp(z[neg])
        up = np.ones_like(z)
        up[neg] = np.exp(z[neg])
        return up/low

    def sigmoid_dr(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def softmax(self,z):
        z -= np.max(z,axis=1,keepdims = True)
        return (np.exp(z).T/(np.sum(np.exp(z),axis=1))).T

    def softmax_dr(self,z):
        return self.softmax(z)*(1-self.softmax(z))

    def tanh(self,x):
        return np.tanh(x)

    def tanh_prime(self,x):
        return 1-np.tanh(x)**2

# Neural Network:
class Network():
    def __init__(self):
        self.layers = []
        self.error_list = []
        
    def add(self,layer):
        self.layers.append(layer)
        
    def error(self,X,y,batch_size,loss_type):
        if loss_type == "MSE":
            return 2*(X-y)/batch_size
        if loss_type == "Cross_entropy":
            return (X-y)/batch_size
        
    def path(self,neuron_info):
        for i in range(1,len(neuron_info)):
            self.add(FC_layer(neuron_info[i-1],neuron_info[i]))
            if i == len(neuron_info)-1:
                self.add(AC_layer(Activation().sigmoid,Activation().sigmoid_dr))
            else:
                self.add(AC_layer(Activation().softmax,Activation().softmax_dr))
    
    def fit(self,X,y,epoch,learning_rate,batch_size,optimizer,loss_type,regularization):
        
        regu = None
        if regularization == None:
            lamda,b = 0,0
            regu = lamda,b    
        elif "SPARSE"  in regularization and "L2" in regularization:
            b = float(input("For SPARSE regularization beta = "))
            lamda = float(input("For L2 regularization lambda = "))
            regu = lamda,b
        elif "SPARSE" in regularization and "L2" not in regularization:
            b = float(input("For SPARSE regularization beta = "))
            lamda = 0
            regu = lamda,b
        elif "L2" in regularization and "SPARSE" not in regularization:
            lamda = float(input("For L2 regularization lambda = "))
            b = 0
            regu = lamda,b
        else:
            regu = 0,0
        
        
        m = X.shape[0]
        t = 1
        if m < batch_size:
            batch_no = 1
        else:
            batch_no = m//batch_size
        
        
        if regularization != None and "Noise" in regularization:
            noise_level = float(input("ADD noise level: "))
            X += np.random.uniform(low=0.0,high=noise_level,size=(X.shape))
            
        
        for i in range(epoch):
            t+= 1
            for batch in range(batch_no):
                str_idx = batch*batch_size
                stp_idx = (batch+1)*batch_size
                
                output = X[str_idx:stp_idx]
                for layer in  self.layers:
                    output = layer.forward_prop(output)
                    
                #error = 2*(output-y[str_idx:stp_idx])/batch_size
                error = self.error(output,y[str_idx:stp_idx],batch_size,loss_type)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error,learning_rate,optimizer,t,regu)
                
            original_error = np.linalg.norm((self.predict(X) - y))
            self.error_list.append(original_error)
            if i % 50 == 0:
                print(f" {i}th Epoch error : {self.error_list[i]}")
            
    def predict(self,X):
        output = X
        for layer in self.layers:
            output = layer.forward_prop(output)
        return output


    