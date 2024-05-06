This code compares RMSE values for fourier neural operator, CNN and LSTM on different timeseries datasets like sunspot, lazer, lorenz etc. 
RMSE values are compared by running 30 experiments of 1000 epochs on all 3 models respectively to obtain correct results.

Implementation of FNO for Time Series -> 

Implemented a 1D spectral convolutional fourier layer which takes a 1D signal as input, processes it in frequency domain and returns a transformed 1D signal as output. 
This transformation is achieved through spectral convolution using fourier coefficients and learnable weights. 

Different methods / functions used in code - 
METHOD-1 (INITIALIZATION) = Weights of this layer are complex numbers scaled by a factor of (1 / in_channels * out_channels) to prevent numerical instability during training. 
METHOD-2 (COMPLEX MULTIPLICATION) = It uses einstein summation ( torch.einsum ) to perform multiplication efficiently. 
METHOD-3 (FORWARD PASS) = For each batch of input tensor x, fourier transform is computed. Then complex multiplication is performed using STEP 2 and then, inverse fourier transform is applied to get output tensor. 

Adam optimiser with learning rate 0.001 is used. 

Implementation of CNN and LSTM -> 
Training is performed on batch size 64.
For LSTM Model, we have added 1 unit of LSTM layer with a fully connected dense layer of output_steps units.  
FOr CNN Model, we have added 1D convolutional layer with 64 filters and kernal size = 3. Further, ReLU activation function is used and a max pooling layer of pool size = 2 is added. 
Then, we flatten the output to 1D array and add a fully connected dense layer using ReLU activation. 

Results -> 
For most of the datasets, RMSE (CNN) < RMSE (LSTM) << RMSE (Fourier). 
Results can be seen in the powerpoint rmse_fourier. 
