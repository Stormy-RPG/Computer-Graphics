import numpy as np
from random import randint 
from math import floor
from PIL import Image

img_mat = np.zeros((1000, 1000, 3), dtype=np.uint8)

# for i in range(1000):
#     for j in range(1000):
        # img_mat[i, j] = [0, 255, 0]
        # img_mat[i, j] = (i + j) % 256
        # img_mat[i, j] = [i % 256, j % 256, (i + j) % 256]

# x0, y0, x1, y1 = 50.3, 30.2, 987.65, 123.45


def line(img_mat, y0, x0, y1, x1, z0=0, z1=0):
    d_max = max(abs(floor(x0) - floor(x1)), abs(floor(y0) - floor(y1)))
    L = d_max + 1

    # rgb = [randint(0, 255), randint(100, 255), randint(100, 255)]
    # rgb = [x0 * y0 / 4 % 256, (y0 + 1) / 4 % 256, (x0 + 2) / 4 % 256]
    # rgb = [255, 255, 255]

    if L == 1:
        img_mat[floor(y0), floor(x0)] = z_color(z0)
    elif L > 1:
        dx = (x1 - x0) / (L - 1)
        dy = (y1 - y0) / (L - 1)
        dz = (z1 - z0) / (L - 1)
        x_f, y_f, z_f = x0, y0, z0

        for _ in range(L):
            img_mat[floor(y_f), floor(x_f)] = z_color(z_f)
            x_f += dx
            y_f += dy
            z_f += dz


def z_color(z):
    c = floor(50 + 205 * (z - z_min) / (z_max - z_min))
    return [c, c, c]

# line(img_mat, x0, y0, x1, y1)

v = []
f = []
with open("model.obj", "r") as file:
    for s in file:
        sp = s.split()
        
        if sp[0] == "v":
            v.append([float(sp[1]), float(sp[2]), float(sp[3])])

        if sp[0] == "f":
            f.append([int(sp[1].split("/")[0]), int(sp[2].split("/")[0]), int(sp[3].split("/")[0])])

    # print(f)

z_min = min(p[2] for p in v)
z_max = max(p[2] for p in v)

for i in range(len(v)):
    x, y, z = v[i]
    img_mat[floor(-y*7000+900), floor(x*7000+500)] = [i % 256, (i+1) % 256, (i+2) % 256]

for i in range(len(f)):
    x0, y0, z0 = v[f[i][0]-1]
    x1, y1, z1 = v[f[i][1]-1]
    x2, y2, z2 = v[f[i][2]-1]

    line(img_mat, -y0*7000+900, x0*7000+500, -y1*7000+900, x1*7000+500, z0, z1)
    line(img_mat, -y1*7000+900, x1*7000+500, -y2*7000+900, x2*7000+500, z1, z2)
    line(img_mat, -y2*7000+900, x2*7000+500, -y0*7000+900, x0*7000+500, z2, z0)


img = Image.fromarray(img_mat)
img.save("image.png") 