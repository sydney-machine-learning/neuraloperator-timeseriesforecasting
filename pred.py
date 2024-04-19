import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM

train_data = pd.read_csv('data/rossler_train.csv', index_col=0)
test_data = pd.read_csv('data/rossler_test.csv', index_col=0)
train_data = train_data.values
test_data = test_data.values

input_steps, output_steps, n_features, num_epochs = 5, 10, 1, 1000

X_train, Y_train = (train_data[:, 0:input_steps].astype(np.float32),
                    train_data[:, input_steps: input_steps + output_steps].astype(np.float32))
X_test, Y_test = (test_data[:, 0:input_steps].astype(np.float32),
                  test_data[:, input_steps: input_steps + output_steps].astype(np.float32))

X_train = X_train.reshape(X_train.shape[0], -1, n_features)
X_test = X_test.reshape(X_test.shape[0], -1, n_features)

print(X_train.shape)
print(X_test.shape)

Y_train = Y_train.reshape(Y_train.shape[0], -1, 1)
Y_test = Y_test.reshape(Y_test.shape[0], -1, 1)

print(Y_train.shape)
print(Y_test.shape)


def root_mean_sq_error(train, predict):
    error = np.subtract(train, predict)
    sq_error = np.sum(np.square(error)) / train.shape[0]
    return np.sqrt(sq_error)


# defines a 1D spectral convolutional fourier layer
class ModelFourier(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(ModelFourier, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def compl_mul1d(self, input_batch, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input_batch, weights)

    def forward(self, x):
        batch_size = x.shape[0]

        # Compute Model_Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Model_Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


def model_lstm():
    model = Sequential()
    model.add(LSTM(1, input_shape=(input_steps, n_features)))
    model.add(Dense(output_steps))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=64,
              verbose=1)
    return model


def model_cnn():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(output_steps, activation='relu'))
    model.add(Dense(output_steps))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1000, batch_size=64,
              verbose=1)
    return model


def run_model_fourier():
    model = ModelFourier(in_channels=1, out_channels=10, modes=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(torch.from_numpy(X_train))
        print(outputs.shape)
        loss = criterion(outputs, torch.from_numpy(Y_train))
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    # predicting from the model
    train_predict = model(torch.from_numpy(X_train))
    test_predict = model(torch.from_numpy(X_test))

    # Calculate MSE performance metrics
    return [root_mean_sq_error(Y_train, train_predict.detach().numpy()),
            root_mean_sq_error(Y_test, test_predict.detach().numpy())]


def run_model_cnn_lstm(flag=0):
    model = model_cnn()
    if flag:
        model = model_lstm()

    print(X_train.shape)
    train_predict = model.predict(X_train)
    train_predict = train_predict.reshape(train_predict.shape[0], train_predict.shape[1], 1)
    print(train_predict.shape)

    test_predict = model.predict(X_test)
    test_predict = test_predict.reshape(test_predict.shape[0], test_predict.shape[1], 1)

    # Calculate MSE performance metrics
    print(Y_train.shape)

    return [root_mean_sq_error(Y_train, train_predict),
            root_mean_sq_error(Y_test, test_predict)]


def main():
    num_exp = 30
    rmse_mean = [0, 0]
    for run in range(num_exp):
        arr = run_model_cnn_lstm(0)
        rmse_mean[0] += arr[0]
        rmse_mean[1] += arr[1]

    rmse_mean[0] /= num_exp
    rmse_mean[1] /= num_exp
    print(rmse_mean)


if __name__ == "__main__":
    main()
