

from scipy.signal import convolve2d
def updateConv(mGrid):
    ep, f, q, d1, d2 = (0.03, 2.0, 0.002, 1.0, 0.6)

    grid = np.zeros_like(mGrid)
    mean = np.zeros_like(mGrid)

    kernel_avg = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])/9

    mean[0] = convolve2d(mGrid[0], kernel_avg, mode='same', boundary='fill')
    mean[1] = convolve2d(mGrid[1], kernel_avg, mode='same', boundary='fill')

    l = (mean - mGrid) / (0.1) ** 2

    x = mGrid[0]
    y = mGrid[1]

    grid[0] = mGrid[0] + 0.0001 * (d1 * l[0] + (1 / ep) * (x * (1 - x) - f * (x - q) / (q + x) * y))
    grid[1] = mGrid[1] + 0.0001 * (d2 * l[1] + x - y)

    return grid


def init_spiral(n_grid):
    ep, f, q, d1, d2 = (0.03, 2.0, 0.002, 1.0, 0.6)
    grid = np.zeros((2,n_grid, n_grid))
    ss = q*(f + 1)/(f - 1)

    for i in range(n_grid):
        for j in range(n_grid):
            x = i - int(n_grid/2)
            y = j - int(n_grid / 2)

            theta = np.arctan2(y,x) + np.pi
            if theta < 0.5 and theta >= 0:
                grid[0, i, j] = 0.8
            else:
                grid[0, i, j] = ss

            grid[1, i, j] = ss + theta/ (8 * np.pi * f)
    return grid

def init_circle(n_grid):
    ep, f, q, d1, d2 = (0.03, 2.0, 0.002, 1.0, 0.6)
    grid = np.zeros((2,n_grid, n_grid))
    ss = q*(f + 1)/(f - 1)

    for i in range(n_grid):
        for j in range(n_grid):
            x = i - int(n_grid/2)
            y = j - int(n_grid / 2)

            r = np.sqrt(x**2 + y**2)
            grid[1,i,j] = ss
            if r < 10:
                grid[0, i, j] = 0.8
            else:
                grid[0,i,j] = ss




    return grid




import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mpl.use('TkAgg')

fig = plt.figure()

n_grid = 200

grid = init_circle(n_grid)





im = plt.imshow(grid[0], animated=True, interpolation='None')


def loop2(*args):
    global grid
    for i in range(100):
        grid = updateConv(grid)
    im.set_array(grid[0])
    return im,


ani = animation.FuncAnimation(fig, loop2, blit=True)
plt.show()
# mywriter = animation.FFMpegWriter(fps=30)
# ani.save('eh.mp4',writer=mywriter)