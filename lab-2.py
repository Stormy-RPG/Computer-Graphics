import numpy as np
from random import randint 
from math import floor, ceil
from PIL import Image

img_mat = np.zeros((1000, 1000, 3), dtype=np.uint8)



def bar_coords(y, x, y0, x0, y1, x1, y2, x2):
    dinom = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / dinom
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / dinom
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


# Подаём на вход вещественные координаты вершин. 
# Учесть, что они положительные
def print_triangle(img_mat, y0, x0, y1, x1, y2, x2, canvas_width = 999, canvas_height = 999, base_color = [255, 255, 255]):
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
            h1, h2, h3 = bar_coords(y, x, y0, x0, y1, x1, y2, x2)

            if (h1 >= 0) and (h2 >= 0) and (h3 >= 0):
                img_mat[y, x] = base_color


print_triangle(img_mat, 100.3, 100.2, 200.5, 300.7, 400.1, 150.9)
print_triangle(img_mat, 300.3, 300.2, 500.5, 600.7, 700.1, 400.9)
print_triangle(img_mat, 600.3, -8.0, 800.5, 800.7, 900.1, 550.9)
print_triangle(img_mat, -437, -8.0, -800.5, -800.7, -900.1, -550.9) 


img = Image.fromarray(img_mat)
img.save("image-lab2.png") 