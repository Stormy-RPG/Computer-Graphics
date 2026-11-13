import numpy as np
from os import PathLike
from math import floor, ceil, sin, cos, radians, sqrt
from PIL import Image, ImageTk
import tkinter as tk
from numba import njit


@njit(cache=True)
def print_triangle(img_mat, z_buf, py0, px0, py1, px1, py2, px2, z0, z1, z2, base_color, canvas_width, canvas_height):
    # Знаменатель барицентрических координат для одного треугольника
    dinom = ((px0 - px2) * (py1 - py2) - (px1 - px2) * (py0 - py2))
    if dinom == 0.0:
        return

    xmin = int(floor(min(px0, min(px1, px2))))
    xmax = int(ceil(max(px0, max(px1, px2))))
    ymin = int(floor(min(py0, min(py1, py2))))
    ymax = int(ceil(max(py0, max(py1, py2))))

    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > canvas_width - 1: xmax = canvas_width - 1
    if ymax > canvas_height - 1: ymax = canvas_height - 1

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            # Бароцентрические координаты
            h0 = ((x - px2) * (py1 - py2) - (px1 - px2) * (y - py2)) / dinom
            h1 = ((px0 - px2) * (y - py2) - (x - px2) * (py0 - py2)) / dinom
            h2 = 1.0 - h0 - h1

            if (h0 >= 0.0) and (h1 >= 0.0) and (h2 >= 0.0):
                zpol = h0*z0 + h1*z1 + h2*z2

                if zpol < z_buf[y, x]:
                    img_mat[y, x, :] = base_color
                    z_buf[y, x] = zpol


@njit(cache=True)
def render_model(img_mat, z_buf, v, f, light_vector, scale, perspective):
    H = img_mat.shape[0]
    W = img_mat.shape[1]

    lx = light_vector[0]
    ly = light_vector[1]
    lz = light_vector[2]
    l_norm = sqrt(lx * lx + ly * ly + lz * lz)

    for i in range(f.shape[0]):
        i0, i1, i2 = f[i, 0], f[i, 1], f[i, 2]

        # Мировые координаты вершин треугольника
        x0, y0, z0 = v[i0, 0], v[i0, 1], v[i0, 2]
        x1, y1, z1 = v[i1, 0], v[i1, 1], v[i1, 2]
        x2, y2, z2 = v[i2, 0], v[i2, 1], v[i2, 2]

        # Переводим мировые координаты в экранные
        if perspective:
            if z0 <= 0.0 or z1 <= 0.0 or z2 <= 0.0: 
                continue
            py0 = (-y0 * scale)/z0 + H/2; px0 = (x0 * scale)/z0 + W/2
            py1 = (-y1 * scale)/z1 + H/2; px1 = (x1 * scale)/z1 + W/2
            py2 = (-y2 * scale)/z2 + H/2; px2 = (x2 * scale)/z2 + W/2
        else:
            py0 = -y0 * scale + H/2; px0 = x0 * scale + W/2
            py1 = -y1 * scale + H/2; px1 = x1 * scale + W/2
            py2 = -y2 * scale + H/2; px2 = x2 * scale + W/2

        # Нормаль к треугольнику (векторное произведение двух сторон)
        ax = x1 - x2; ay = y1 - y2; az = z1 - z2
        bx = x1 - x0; by = y1 - y0; bz = z1 - z0
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx

        n_norm = sqrt(nx * nx + ny * ny + nz * nz)
        if n_norm == 0.0:
            continue

        light_cos = (nx * lx + ny * ly + nz * lz) / (n_norm * l_norm)

        if light_cos < 0.0:
            color = np.uint8(-255.0 * light_cos)
            print_triangle(img_mat, z_buf, py0, px0, py1, px1, py2, px2, z0, z1, z2, color, W, H)




class Scene:
    def __init__(self, filename: str | bytes | PathLike, angles = (0.0, 0.0, 0.0), scale = 1.0, t_shift = (0.0, 0.0, 0.0), light_vector = (0.0, 0.0, 1.0)):
        self.light_vector = np.array(light_vector, dtype=np.float64)
        self.shift = np.array(t_shift, dtype=np.float64)
        self.angles = list(angles)
        self.scale = scale

        v = []
        f = []

        with open(filename, "r") as file:
            for s in file:
                sp = s.split()

                if not sp:
                    continue
                
                if sp[0] == "v":
                    v.append([float(sp[1]), float(sp[2]), float(sp[3])])

                if sp[0] == "f":
                    points = [int(p.split("/")[0]) - 1 for p in sp[1:]]
                    # points = [int(sp[1].split("/")[0]) - 1, int(sp[2].split("/")[0]) - 1, int(sp[3].split("/")[0]) - 1]
                    for i in range(1, len(points) - 1):
                        f.append([points[0], points[i], points[i + 1]])

        # Однородные numpy-массивы вместо списков списков для njit
        self.v0 = np.array(v, dtype=np.float64) # исходные вершины
        self.f = np.array(f, dtype=np.int64)

        # Центрируем модель и вписываем в куб размером 1
        vmin = self.v0.min(axis=0)
        vmax = self.v0.max(axis=0)
        print(vmin, vmax)
        self.v0 = (self.v0 - (vmin + vmax) / 2) / (vmax - vmin).max()


    def rotation_matrix(self):
        a, b, g = np.radians(self.angles)

        Rx = np.array([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]])
        Ry = np.array([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]])
        Rz = np.array([[cos(g), sin(g), 0], [-sin(g), cos(g), 0], [0, 0, 1]])

        return Rx @ Ry @ Rz

    def transform(self):
        # Работаем со строкамии, поэтому матрица поворота транспонируется
        return (self.scale * self.v0) @ self.rotation_matrix().T + self.shift
    



class Renderer:
    def __init__(self, width: int = 1000, height: int = 1000, scale: int = 7000, perspective: bool = False):
        self.W = width
        self.H = height
        self.scale = float(scale)
        self.perspective = perspective
        self.img_mat = np.zeros((height, width, 3), dtype=np.uint8)
        self.z_buf = np.full((height, width), np.inf, dtype=np.float32)

    def render(self, scene: Scene):
        self.img_mat.fill(0)
        self.z_buf.fill(np.inf)
        v = scene.transform()
        render_model(self.img_mat, self.z_buf, v, scene.f, scene.light_vector, float(self.scale), self.perspective)

    def save_image(self, filename: str | bytes | PathLike):
        Image.fromarray(self.img_mat).save(filename)


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
        step = 0.01  # шаг сдвига в единицах obj

        # Масштаб экранных координат (a)
        if k in ('plus', 'equal', 'KP_Add'):
            self.renderer.scale *= 1.1
        elif k in ('minus', 'KP_Subtract'):
            self.renderer.scale /= 1.1

        # Поворот вокруг оси Y
        elif k == 'q':
            self.scene.angles[1] -= 10
        elif k == 'e':
            self.scene.angles[1] += 10

        # Поворот вокруг оси X
        elif k == 't':
            self.scene.angles[0] -= 10
        elif k == 'g':
            self.scene.angles[0] += 10

        # Поворот вокруг оси Z
        elif k == 'y':
            self.scene.angles[2] -= 10
        elif k == 'h':
            self.scene.angles[2] += 10

        # Сдвиг по X и Y
        elif k == 'Left':
            self.scene.shift[0] -= step
        elif k == 'Right':
            self.scene.shift[0] += step
        elif k == 'Up':
            self.scene.shift[1] += step
        elif k == 'Down':
            self.scene.shift[1] -= step

        # Сдвиг по Z (не даём уйти за камеру)
        elif k == 'w':
            self.scene.shift[2] += step
        elif k == 's':
            self.scene.shift[2] = max(0.1, self.scene.shift[2] - step)

        # Переключение параллельной и перспективной проекции
        elif k == 'p':
            self.renderer.perspective = not self.renderer.perspective

        # Сохранение кадра
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
    # scene = Scene('objs\\12221_Cat_v1_l3.obj', angles=(0.0, 180.0, 0.0), scale=1.0, t_shift=(0.0, -0.057, 1.0))
    scene = Scene('objs\\12221_Cat_v1_l3.obj', angles=(0.0, 180.0, 0.0), scale=1.0, t_shift=(0.0, 0.0, 2.0))
    renderer = Renderer(1000, 1000, 7000, perspective=True)
    # renderer.render(scene)
    # renderer.save_image('imgs\\output.png')
    App(scene, renderer, 500, 500).run()