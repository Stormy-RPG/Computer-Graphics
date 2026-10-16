import numpy as np
from random import randint 
from math import floor, ceil, sin, cos, radians
from PIL import Image

img_mat = np.zeros((1000, 1000, 3), dtype=np.uint8)
z_buf = np.full((1000, 1000), float('inf'), dtype=np.float32)

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


def bar_coords(y, x, y0, x0, y1, x1, y2, x2):
    dinom = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / dinom
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / dinom
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


# Подаём на вход вещественные координаты вершин. 
# Учесть, что они положительные
def print_triangle(img_mat, z_buf, y0, x0, z0, y1, x1, z1, y2, x2, z2, canvas_width = 999, canvas_height = 999, base_color = [255, 255, 255]):
    xmin = floor(min(x0, x1, x2))
    xmax = ceil(max(x0, x1, x2))
    ymin = floor(min(y0, y1, y2))
    ymax = ceil(max(y0, y1, y2))

    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > canvas_width): xmax = canvas_width
    if (ymax > canvas_height): ymax = canvas_height

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            h0, h1, h2 = bar_coords(y, x, y0, x0, y1, x1, y2, x2)

            if (h0 >= 0) and (h1 >= 0) and (h2 >= 0):
                zpol = h0*z0 + h1*z1 + h2*z2

                if zpol < z_buf[y, x]:
                    img_mat[y, x] = base_color
                    z_buf[y, x] = zpol

def triangle_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    a = np.array([x1 - x2, y1 - y2, z1 - z2])
    b = np.array([x1 - x0, y1 - y0, z1 - z0])
    return np.cross(a, b)


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

alpha = radians(180)
for i in range(len(v)):
    x, y, z = v[i]
    v[i] = [x * cos(alpha) + z * sin(alpha), y, -x * sin(alpha) + z * cos(alpha)]


z_min = min(p[2] for p in v)
z_max = max(p[2] for p in v)

# Lab-1
# for i in range(len(v)):
#     x, y, z = v[i]
#     img_mat[floor(-y*7000+900), floor(x*7000+500)] = [i % 256, (i+1) % 256, (i+2) % 256]

for i in range(len(f)):
    x0, y0, z0 = v[f[i][0]-1]
    x1, y1, z1 = v[f[i][1]-1]
    x2, y2, z2 = v[f[i][2]-1]

    # Lab-1
    # line(img_mat, -y0*7000+900, x0*7000+500, -y1*7000+900, x1*7000+500, z0, z1)
    # line(img_mat, -y1*7000+900, x1*7000+500, -y2*7000+900, x2*7000+500, z1, z2)
    # line(img_mat, -y2*7000+900, x2*7000+500, -y0*7000+900, x0*7000+500, z2, z0)

    
    
    l = np.array((0, 0, 1))
    n = triangle_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    light_cos = np.dot(n, l) / (np.linalg.norm(n) * np.linalg.norm(l))

    if light_cos < 0:
        # print_triangle(img_mat, z_buf, -y0*7000+900, x0*7000+500, z0*7000, -y1*7000+900, x1*7000+500, z1*7000, -y2*7000+900, x2*7000+500, z2*7000, base_color=z_color((z0 + z1 + z2) / 3))
        # print_triangle(img_mat, z_buf, -y0*7000+900, x0*7000+500, z0*7000, -y1*7000+900, x1*7000+500, z1*7000, -y2*7000+900, x2*7000+500, z2*7000, base_color=[randint(0, 255), randint(100, 255), randint(100, 255)])
        print_triangle(img_mat, z_buf, -y0*7000+900, x0*7000+500, z0*7000, -y1*7000+900, x1*7000+500, z1*7000, -y2*7000+900, x2*7000+500, z2*7000, base_color=[-255*light_cos, -255*light_cos, -255*light_cos])

img = Image.fromarray(img_mat)
img.save("image.png") 