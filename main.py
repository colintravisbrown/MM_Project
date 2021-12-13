
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from LPythranTest import laplace_pyth
def updateConv(mGrid):
    kernel_avg = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])
    if wrap:
        l0 = laplace_pyth(mGrid[0], wrap)
        l1 = laplace_pyth(mGrid[1], wrap)
    else:
        l0 = laplace_pyth(np.pad(mGrid[0],1), wrap)/(0.125**2)
        l1 = laplace_pyth(np.pad(mGrid[1],1), wrap)/(0.125**2)



    x = mGrid[0]
    y = mGrid[1]
    mGrid[0] = mGrid[0] + 0.00025 * (d1 * l0 + 1/ep*(x + q*f*y/(q + x) - x**2 - x*f*y/(q + x)))
    mGrid[1] = mGrid[1] + 0.00025 * (d2 * l1 + x - y)

    return mGrid


def init_spiral(n_grid):
    grid = np.zeros((2,n_grid, n_grid))
    ss = q*(f+1)/(f-1)

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
    grid = np.zeros((2,n_grid, n_grid))
    ss = q*(f+1)/(f-1)

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






mpl.use('TkAgg')

n_grid = 128*2
fig, ax = plt.subplots()


ep, f, q, d1, d2 = (0.03, 2.0, 0.001, 1, 0.6)
wrap = False
ss = 1/2*(1 - f - q + np.sqrt((1 - f - q)**2 + 4*q*(1 + f)))
# grid = np.ones((2,n_grid,n_grid))*ss + (np.random.rand(2,n_grid,n_grid)-0.5)*ss*0.1
#grid = init_circle(n_grid)
grid = init_spiral(n_grid)

# for i in range(10000):
#     grid = updateConv(grid)


im = ax.imshow(grid[0], animated=True)
# cb = fig.colorbar(im)

def loop2(*args):
    global grid
    for i in range(500):
        grid = updateConv(grid)
    im.set_array(grid[0])
    # norm = mpl.colors.Normalize(vmin = grid[0].min(), vmax = grid[0].max())
    # im.set_norm(norm)
    #im.set_clim(grid[0].min(), grid[0].max())
    if np.any(grid <0):
        print('instability')
        quit()
    return im,


ani = animation.FuncAnimation(fig, loop2, blit=False, frames = 200)
plt.show()
# mywriter = animation.FFMpegWriter(fps=30)
# ani.save('eh.mp4',writer=mywriter)