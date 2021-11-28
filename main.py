import numpy as np
import numba as nb

n_grid = 400

grid = np.random.rand(3, n_grid, n_grid)

n_steps = 100
a, b, c = (1.2,1.2,1)


@nb.njit()
def meanGrid(grid):
    mGrid = np.zeros_like(grid)
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(3):
                i0 = (i - 1) % n_grid
                i2 = (i + 1) % n_grid
                j0 = (j - 1) % n_grid
                j2 = (j + 1) % n_grid

                r1 = grid[k, i0, j0] + grid[k, i, j0] + grid[k, i2, j0]
                r2 = grid[k, i0, j] + grid[k, i, j] + grid[k, i2, j]
                r3 = grid[k, i0, j2] + grid[k, i, j2] + grid[k, i2, j2]
                mean = 1 / 9 * (r1 + r2 + r3)
                mGrid[k, i, j] = mean

    return mGrid

@nb.jit()
def update(mGrid):
    grid = np.zeros_like(mGrid)
    grid[0] = mGrid[0] * (1 + a * mGrid[1] - c * mGrid[2])
    grid[1] = mGrid[1] * (1 + b * mGrid[2] - a * mGrid[0])
    grid[2] = mGrid[2] * (1 + c * mGrid[0] - b * mGrid[1])
    return np.clip(grid, 0, 1)


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mpl.use('TkAgg')

fig = plt.figure()

im = plt.imshow(grid[2], animated=True,  cmap='Greys_r')


def loop(*args):
    global grid
    grid = meanGrid(grid)
    grid = update(grid)
    im.set_array(grid[2])
    return im,

ani = animation.FuncAnimation(fig, loop, interval=10, blit=True, frames=300)
mywriter = animation.FFMpegWriter(fps=30)
ani.save('eh.mp4',writer=mywriter)