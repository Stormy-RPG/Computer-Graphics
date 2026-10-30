import numpy as np
from os import PathLike
from typing import List
from math import floor, ceil, sin, cos, radians
from PIL import Image, ImageTk
import tkinter as tk
from numba import njit


def render_model(img_mat, z_buf, v, f, light_vector, scale, shift_y, shift_x):
    for i in range(len(f)):
        # Мировые координаты вершин треугольника
        x0, y0, z0 = v[f[i][0]]
        x1, y1, z1 = v[f[i][1]]
        x2, y2, z2 = v[f[i][2]]

        # Переводим мировые координаты в экранные
        py0, px0 = -y0*scale+shift_y, x0*scale+shift_x
        py1, px1 = -y1*scale+shift_y, x1*scale+shift_x
        py2, px2 = -y2*scale+shift_y, x2*scale+shift_x

        
        n = Renderer.triangle_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        light_cos = np.dot(n, light_vector) / (np.linalg.norm(n) * np.linalg.norm(light_vector))

        if light_cos < 0:
            Renderer.print_triangle(img_mat, z_buf, py0, px0, py1, px1, py2, px2, [z0, z1, z2], base_color=[-255*light_cos, -255*light_cos, -255*light_cos])

class Scene:
    def __init__(self, filename: str | bytes | PathLike, rotation_angle: float = 0, light_vector = np.array((0, 0, 1))):
        self.v = []
        self.f = []
        self.light_vector = light_vector
        self.rotation_angle = rotation_angle
        self.alpha = radians(rotation_angle)

        with open(filename, "r") as file:
            for s in file:
                sp = s.split()
                
                if sp[0] == "v":
                    self.v.append([float(sp[1]), float(sp[2]), float(sp[3])])

                if sp[0] == "f":
                    self.f.append([int(sp[1].split("/")[0]) - 1, int(sp[2].split("/")[0]) - 1, int(sp[3].split("/")[0]) - 1])
        
        # Поворот модели на ALPHA градусов вокруг оси Oy
        for i in range(len(self.v)):
            x, y, z = self.v[i]
            self.v[i] = [x * cos(self.alpha) + z * sin(self.alpha), y, -x * sin(self.alpha) + z * cos(self.alpha)]


class Renderer:
    def __init__(self, width: int = 1000, height: int = 1000, scale: int = 7000, shifts: List = [900, 500]):
        self.W = width
        self.H = height
        self.scale = scale
        self.shift_x, self.shift_y = shifts[1], shifts[0]
        self.img_mat = np.zeros((height, width, 3), dtype=np.uint8)
        self.z_buf = np.full((height, width), np.inf, dtype=np.float32)

    def render(self, scene: Scene):
        self.img_mat.fill(0)
        self.z_buf.fill(np.inf)
        render_model(self.img_mat, self.z_buf, scene.v, scene.f, scene.light_vector, self.scale, self.shift_y, self.shift_x)
        pass

    def save_image(self, filename: str | bytes | PathLike):
        img = Image.fromarray(self.img_mat)
        img.save(filename)


    @staticmethod
    def bar_coords(y, x, y0, x0, y1, x1, y2, x2):
        dinom = ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))

        lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / dinom
        lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / dinom
        lambda2 = 1.0 - lambda0 - lambda1

        return lambda0, lambda1, lambda2

    @staticmethod
    def triangle_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
        a = np.array([x1 - x2, y1 - y2, z1 - z2])
        b = np.array([x1 - x0, y1 - y0, z1 - z0])
        return np.cross(a, b)

    @staticmethod
    # Подаём на вход вещественные координаты вершин. Учесть, что они положительные.
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
                h0, h1, h2 = Renderer.bar_coords(y, x, py0, px0, py1, px1, py2, px2)

                if (h0 >= 0) and (h1 >= 0) and (h2 >= 0):
                    zpol = h0*z0 + h1*z1 + h2*z2

                    if zpol < z_buf[y, x]:
                        img_mat[y, x] = base_color
                        z_buf[y, x] = zpol


class App:
    def __init__(self, scene: Scene, renderer: Renderer, win_w = 1000, win_h = 1000):
            self.scene = scene
            self.renderer = renderer
    
            # Окно верхнего уровня
            self.root = tk.Tk()
            # Размеры окна
            self.root.geometry(f"{win_w}x{win_h}")
            # Область для рисования внутри окна
            self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
            self.canvas.pack(fill='both', expand=True)
            self.canvas.focus_set()
            # Обновление изображения при изменении размеров окна
            self.canvas.bind('<Configure>', lambda e: self._refit())
            # Привязка обработчика клавиш
            self.root.bind('<Key>', self.on_key)
    
            # Изображение внутри окна
            self.photo = None
            self.full_image = None
    
            self.render()

    def on_key(self, event):
            k = event.keysym
    
            if k == 'plus':
                self.renderer.scale *= 1.1
            elif k == 'minus':
                self.renderer.scale /= 1.1
            elif k == 'space':
                self.renderer.save_image('imgs\\app_output.png')
                return 'break'
            else:
                return
    
            self.render()
            # Не обрабатываем дальше – return 'brake'
            return 'break'

    def render(self):
        self.renderer.render(self.scene)
        self.full_image = Image.fromarray(self.renderer.img_mat)
        self._refit()

    def _refit(self):
        # Если ещё не отрендерилось, пропускаем
        if self.full_image is None: return
        # Ширина и высота области для рисования
        ww = self.canvas.winfo_width()
        wh = self.canvas.winfo_height()
        # Если вдруг сделали область слишком маленькой
        if ww < 10 or wh < 10: return
        # Вписать с сохранением пропорций
        s = min(ww / self.renderer.W, wh / self.renderer.H)
        dw = max(1, int(self.renderer.W * s))
        dh = max(1, int(self.renderer.H * s))
        disp = self.full_image.resize((dw, dh), Image.BILINEAR)
        # Очень важно сохранить ImageTk.PhotoImage(disp) в поле класса
        # Если оно будет локальным, то удалится после выхода сборщиком мусора
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(ww // 2, wh // 2, image=self.photo)   
    
    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    scene = Scene('model.obj', 180)
    renderer = Renderer(1000, 1000, 7000)
    # renderer.render(scene)
    # renderer.save_image('imgs\\output.png')
    App(scene, renderer, 500, 500).run()