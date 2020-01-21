import math

'''
Calculations for resulting output image sizes after convolution.
These equations are taken from PyTorch docs under convolution layers
category in torch.nn section.
'''


def conv2d_out(x_in, kernel_size, stride=1, padding=0, dilation=1):
    x_out = (x_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return math.floor(x_out)

def max_pool2d_out(x_in, kernel_size, stride=None, padding=0, dilation=1):
    if not isinstance(stride, int):
        stride = kernel_size
    x_out = (x_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return math.floor(x_out)


if __name__ == "__main__":
    size = conv2d_out(x_in=100, kernel_size=11, stride=1, padding=0)  # 100 -> 90
    size = max_pool2d_out(x_in=size, kernel_size=3, padding=0)  # 90 -> 30
    size = conv2d_out(x_in=size, kernel_size=3, stride=1, padding=0)  # 30 -> 28
    size = max_pool2d_out(x_in=size, kernel_size=3, padding=0)  # 28 -> 9
    size = conv2d_out(x_in=size, kernel_size=3, stride=1, padding=0)  # 9 -> 7
    size = conv2d_out(x_in=size, kernel_size=3, stride=1, padding=0)  # 7 -> 5
    size = conv2d_out(x_in=size, kernel_size=3, stride=1, padding=0)  # 5 -> 3
    size = max_pool2d_out(x_in=size, kernel_size=3, padding=0)  # 3 -> 1
    print(size)