import numpy as np
import numba as nb



n_steps = 400
a, b, c = (1.2,1.2,1)


@nb.njit(fastmath=True, parallel=False)
def meanGrid(grid):
    mGrid = np.zeros_like(grid)
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(2):
                i0 = (i - 1) % n_grid
                i2 = (i + 1) % n_grid
                j0 = (j - 1) % n_grid
                j2 = (j + 1) % n_grid

                if i0 > i:
                    c1 = 0
                else:
                    c1 = 1
                if i2 < i:
                    c2 = 0
                else:
                    c2 = 1
                if j0 > j:
                    c3 = 0
                else:
                    c3 = 1
                if j2 < j:
                    c4 = 0
                else:
                    c4 = 1
                # c1,c2,c3,c4 = (1,1,1,1)


                r1 = c1*c3*grid[k, i0, j0] + c3*grid[k, i, j0] + c2*c3*grid[k, i2, j0]
                r2 = c1*grid[k, i0, j] + grid[k, i, j] + c2*grid[k, i2, j]
                r3 = c1*c4*grid[k, i0, j2] + c4*grid[k, i, j2] + c2*c4*grid[k, i2, j2]
                mean = 1 / 9 * (r1 + r2 + r3)
                mGrid[k, i, j] = mean

    return mGrid

@nb.njit()
def meanGridNeighbors(grid):
    mGrid = np.zeros_like(grid)
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(3):

                i0 = (i - 1) % n_grid
                i2 = (i + 1) % n_grid
                j0 = (j - 1) % n_grid
                j2 = (j + 1) % n_grid

                if i0 > i:
                    c1 = 0
                else: c1 = 1
                if i2 < i:
                    c2 = 0
                else: c2=1
                if j0 > j:
                    c3 = 0
                else: c3=1
                if j2 < j:
                    c4 = 0
                else: c4=1

                mean = 1/4*(c1*c3*grid[k,i0,j0] + c1*c4*grid[k,i0,j2] + c2*c3*grid[k,i2,j0] +c2*c4*grid[k,i2,j2])

                mGrid[k, i, j] = mean

    return mGrid


@nb.jit()
def update(mGrid):
    grid = np.zeros_like(mGrid)

    x = mGrid[0]
    y = mGrid[1]
    z = mGrid[2]

    grid[0] = x * (1 + a * y - c * z)
    grid[1] = y * (1 + b * z - a * x)
    grid[2] = z * (1 + c * x - b * y)
    return np.clip(grid,0,1)





@nb.jit(fastmath=True)
def updateBrussel(mGrid):
    a, b = (1.0,1.5)

    grid = np.zeros_like(mGrid)

    mean = meanGrid(mGrid)

    x = mGrid[0]
    y = mGrid[1]

    l = mean - mGrid

    grid[0] = mGrid[0] + 0.01*(.5*l[0] + a + x**2 * y - b*x - x)
    grid[1] = mGrid[1] + 0.01*(20*l[1] + b*x - x**2 * y)

    return grid

@nb.jit()
def updateOrg(mGrid):
    ep, f, q, d1, d2 = (0.03, 2.0, 0.002, 0.1, 10)

    grid = np.zeros_like(mGrid)

    mean = meanGrid(mGrid)



    l = (mean - mGrid)/(0.1)**2

    x = mGrid[0]
    y = mGrid[1]

    grid[0] = mGrid[0] + 0.0001*(d1*l[0] + (1/ep)*(x*(1-x) - f*(x - q)/(q + x)*y))
    grid[1] = mGrid[1] + 0.0001*(d2*l[1] + x-y)

    return grid

def init(n_grid):
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




import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mpl.use('TkAgg')

fig = plt.figure()

n_grid = 300

# grid = init(n_grid)

grid = np.random.rand(2,n_grid,n_grid)

print(grid.shape)




for i in range(500):
    grid = updateOrg(grid)


im = plt.imshow(grid[0], animated=True)


def loop(*args):
    global grid
    grid = meanGrid(grid)
    grid = update(grid)
    im.set_array(grid[1])
    return im,

def loop2(*args):
    global grid
    for i in range(200):
        grid = updateOrg(grid)
    im.set_array(grid[0])
    return im,


ani = animation.FuncAnimation(fig, loop2, blit=True)
plt.show()
mywriter = animation.FFMpegWriter(fps=30)
ani.save('eh.mp4',writer=mywriter)