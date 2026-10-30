import numpy as np
from typing import List
from math import floor, ceil, sin, cos, radians
from PIL import Image


HEIGHT = 1000
WIDTH = 1000
SCALE = 7000
X_SHIFT = 500
Y_SHIFT = 900
ROTATION_ANGLE = 180
ALPHA = radians(ROTATION_ANGLE)
LIGHT_VECTOR = np.array((0, 0, 1))


img_mat = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
z_buf = np.full((HEIGHT, WIDTH), float('inf'), dtype=np.float32)



def bar_coords(y, x, y0, x0, y1, x1, y2, x2):
    dinom = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / dinom
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / dinom
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2

def triangle_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    a = np.array([x1 - x2, y1 - y2, z1 - z2])
    b = np.array([x1 - x0, y1 - y0, z1 - z0])
    return np.cross(a, b)

# Подаём на вход вещественные координаты вершин. 
# Учесть, что они положительные
def print_triangle(img_mat, z_buf, py0, px0, py1, px1, py2, px2, z_param: List[float], canvas_width = 999, canvas_height = 999, base_color = [255, 255, 255]):
    xmin = floor(min(px0, px1, px2))
    xmax = ceil(max(px0, px1, px2))
    ymin = floor(min(py0, py1, py2))
    ymax = ceil(max(py0, py1, py2))

    z0, z1, z2 = z_param

    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > canvas_width): xmax = canvas_width
    if (ymax > canvas_height): ymax = canvas_height

    # Сама отрисовка 
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            h0, h1, h2 = bar_coords(y, x, py0, px0, py1, px1, py2, px2)

            if (h0 >= 0) and (h1 >= 0) and (h2 >= 0):
                zpol = h0*z0 + h1*z1 + h2*z2

                if zpol < z_buf[y, x]:
                    img_mat[y, x] = base_color
                    z_buf[y, x] = zpol



v = [] # вектор вершин
f = [] # вектор граней (треугольников)
with open("model.obj", "r") as file:
    for s in file:
        sp = s.split()
        
        if sp[0] == "v":
            v.append([float(sp[1]), float(sp[2]), float(sp[3])])

        if sp[0] == "f":
            f.append([int(sp[1].split("/")[0]), int(sp[2].split("/")[0]), int(sp[3].split("/")[0])])


# Поворот модели на ALPHA градусов вокруг оси Oy
for i in range(len(v)):
    x, y, z = v[i]
    v[i] = [x * cos(ALPHA) + z * sin(ALPHA), y, -x * sin(ALPHA) + z * cos(ALPHA)]


for i in range(len(f)):
    # Мировые координаты вершин треугольника
    x0, y0, z0 = v[f[i][0]-1]
    x1, y1, z1 = v[f[i][1]-1]
    x2, y2, z2 = v[f[i][2]-1]

    # Переводим мировые координаты в экранные
    py0, px0 = -y0*SCALE+Y_SHIFT, x0*SCALE+X_SHIFT
    py1, px1 = -y1*SCALE+Y_SHIFT, x1*SCALE+X_SHIFT
    py2, px2 = -y2*SCALE+Y_SHIFT, x2*SCALE+X_SHIFT

    
    
    n = triangle_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    light_cos = np.dot(n, LIGHT_VECTOR) / (np.linalg.norm(n) * np.linalg.norm(LIGHT_VECTOR))

    if light_cos < 0:
        print_triangle(img_mat, z_buf, py0, px0, py1, px1, py2, px2, [z0, z1, z2], base_color=[-255*light_cos, -255*light_cos, -255*light_cos])

img = Image.fromarray(img_mat)
img.save("image.png") 