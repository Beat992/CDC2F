import torch.nn.functional as F
import torch
import numpy as np

def size_restore(x, input_size=4, output_size=64):
    """
    :param x: 4*4/8*8 or 64*64 patch
    :format: bs, patch num, input_size**2
    :param input_size:
    :param output_size:
    :return:
    """
    x = x.transpose(1, 2)
    x = F.fold(x,
               kernel_size=(input_size, input_size),
               output_size=(output_size, output_size),
               dilation=1,
               padding=0,
               stride=(input_size, input_size))
    return x        # bs_16, 1, 64, 64



def zigzag_extraction(input_tensor):
    batch_size, channels, dct_size, _ = input_tensor.size()
    output_size = dct_size * dct_size

    # Reshape the input tensor to (batch_size * channels, dct_size, dct_size)
    reshaped_tensor = input_tensor.reshape(batch_size * channels, dct_size, dct_size)

    # Create zigzag indices for dct_size x dct_size matrix
    indices = torch.from_numpy(zigzag_indices(dct_size))

    # Extract the zigzag elements from the reshaped_tensor using indices
    zigzag_tensor = reshaped_tensor[:, indices[:, 0], indices[:, 1]]

    # Reshape the zigzag_tensor back to (batch_size, channels, output_size)
    output_tensor = zigzag_tensor.view(batch_size, channels, output_size)

    return output_tensor

def zigzag_indices(size):
    indices = []
    i, j = 0, 0
    for _ in range(size * size):
        indices.append([i, j])
        if (i + j) % 2 == 0:  # Diagonal movements
            if i == 0 and j < size - 1:
                j += 1
            elif j == size - 1:
                i += 1
            else:
                i -= 1
                j += 1
        else:  # Anti-diagonal movements
            if j == 0 and i < size - 1:
                i += 1
            elif i == size - 1:
                j += 1
            else:
                i += 1
                j -= 1
    return np.array(indices)


def inverse_zigzag(input_tensor, dct_size):
    batch_size, channels, _ = input_tensor.size()
    output_size = dct_size * dct_size

    # Reshape the input tensor to (batch_size * channels, output_size)
    flattened_tensor = input_tensor.reshape(batch_size * channels, output_size)

    # Create zigzag indices for dct_size x dct_size matrix
    indices = torch.from_numpy(zigzag_indices(dct_size))

    # Create an empty tensor with the desired shape
    output_tensor = torch.zeros((batch_size * channels, dct_size, dct_size), dtype=input_tensor.dtype).cuda()
    # Assign flattened values to the output tensor using zigzag indices
    output_tensor[:, indices[:, 0], indices[:, 1]] = flattened_tensor

    # Reshape the output tensor back to the original shape
    output_tensor = output_tensor.view(batch_size, channels, dct_size, dct_size)

    return output_tensor

if __name__ == '__main__':
    a = torch.rand(2, 2, 16)
    print(a)
    print(inverse_zigzag(a, dct_size=4))