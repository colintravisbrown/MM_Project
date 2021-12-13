import numpy as np

#pythran export laplace_pyth(float[:,:], bool)

def laplace_pyth(image, wrap):
    h = image.shape[0]


    if wrap:
        laplacian = np.empty((h, h))
        for i in range(h):
            for j in range(h):
                im = (i - 1 + h) % h
                ip = (i+1) % h
                jm = (j - 1 + h) % h
                jp = (j + 1) % h

                laplacian[i, j] = image[im, j] + image[ip, j] + image[i, jm] + image[i, jp] - 4*image[i, j]
    else:
        laplacian = np.empty((h-2, h-2))
        for i in range(1,h-1):
            for j in range(1,h-1):
                im = i-1
                ip = i+1
                jm = j-1
                jp = j+1
                laplacian[im, jm] = image[im, j] + image[ip, j] + image[i, jm] + image[i, jp] - 4 * image[i, j]

    return laplacian
