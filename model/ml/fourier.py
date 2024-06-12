import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, dropout_p: float=0.2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.flatten = nn.Flatten()
        self.linear = None
        self.dropout = nn.Dropout(p=dropout_p)

    def compl_mul2d(self, input, weights):
        # print("Input shape:", input.shape)
        # print("Weights shape:", weights.shape)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        # print("x_ft shape:", x_ft.shape)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],
                                                                    self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2],
                                                                     self.weights2)
        # print("out_ft shape:", out_ft.shape)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        # Check output shape
        x = self.dropout(self.flatten(x))

        if self.linear is None:
            self.linear = nn.Linear(x.shape[1], 1)

        x = self.linear(x)

        return x


# Example usage
batch_size = 256
in_channels = 1
out_channels = 8
modes1 = 2
modes2 = 2
dropout_p = 0.1

model = SpectralConv2d(in_channels, out_channels, modes1, modes2, dropout_p)

# Example input tensor with shape (batch_size, in_channels, height, width)
x = torch.randn(batch_size, in_channels, 7, 10)

# Forward pass
output = model(x)

# Check output shape
print("Output shape:", output.shape)  # Should print (batch_size, 1)