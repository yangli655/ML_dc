import matplotlib.pyplot as plt


def show_image(x):
    """
    Inputs:
        x: (N, 140) matrix, N digits in an image, each digit is a (140, ) column vector.
    """
    num = x.shape[0]
    x = x.reshape(num, 14, 10).transpose(1, 0, 2).reshape(14, num * 10)
    plt.imshow(x, cmap='gray')
