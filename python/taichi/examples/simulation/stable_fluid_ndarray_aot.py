# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import argparse

import numpy as np

import taichi as ti

res = 512
dt = 0.03
p_jacobi_iters = 500  # 40 for a quicker but less accurate result
f_strength = 10000.0
curl_strength = 0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
force_radius = res / 2.0
gravity = True
debug = False
paused = False
use_sparse_matrix = False

if use_sparse_matrix:
    ti.init(arch=ti.x64)
    print('Using sparse matrix')
else:
    ti.init(arch=ti.vulkan)
    print('Using jacobi iteration')

mouse_data = ti.field(dtype=ti.f32,shape=(8))
_velocities = ti.Vector.ndarray(2, dtype=ti.f32, shape=(res,res))
_new_velocities = ti.Vector.ndarray(2, dtype=ti.f32, shape=(res,res))
velocity_divs = ti.ndarray(dtype=ti.f32, shape=(res,res))
velocity_curls = ti.ndarray(dtype=ti.f32, shape=(res,res))
_pressures = ti.ndarray(dtype=ti.f32, shape=(res,res))
_new_pressures = ti.ndarray(dtype=ti.f32, shape=(res,res))
_dye_buffer = ti.Vector.ndarray(3, dtype=ti.f32, shape=(res,res))
_new_dye_buffer = ti.Vector.ndarray(3, dtype=ti.f32, shape=(res,res))
_dye_image_buffer = ti.Vector.ndarray(4, dtype=ti.f32, shape=(res,res))
touch = ti.ndarray(dtype=ti.f32, shape=(2))

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)

if use_sparse_matrix:
    # use a sparse matrix to solve Poisson's pressure equation.
    @ti.kernel
    def fill_laplacian_matrix(A: ti.linalg.sparse_matrix_builder()):
        for i, j in ti.ndrange(res, res):
            row = i * res + j
            center = 0.0
            if j != 0:
                A[row, row - 1] += -1.0
                center += 1.0
            if j != res - 1:
                A[row, row + 1] += -1.0
                center += 1.0
            if i != 0:
                A[row, row - res] += -1.0
                center += 1.0
            if i != res - 1:
                A[row, row + res] += -1.0
                center += 1.0
            A[row, row] += center

    N = res * res
    K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
    b = ti.field(ti.f32, shape=N)

    fill_laplacian_matrix(K)
    L = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(L)
    solver.factorize(L)


@ti.func
def sample(qf: ti.template(), u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(res - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf: ti.template(), p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect(vf: ti.any_arr(element_dim=1), qf: ti.any_arr(element_dim=1), new_qf: ti.any_arr(element_dim=1)):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p) * dye_decay


@ti.kernel
def apply_impulse(vf: ti.any_arr(element_dim=1), dyef: ti.any_arr(element_dim=1)):
    g_dir = -ti.Vector([0, 9.8]) * 300
    for i, j in vf:
        omx, omy = mouse_data[2], mouse_data[3]
        mdir = ti.Vector([mouse_data[0], mouse_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)

        dc = dyef[i, j]
        a = dc.norm()

        momentum = (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt

        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15)**2)) * ti.Vector(
                [mouse_data[4], mouse_data[5], mouse_data[6]])

        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.any_arr(element_dim=1), vd: ti.any_arr()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        vd[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5


@ti.kernel
def vorticity(vf: ti.any_arr(element_dim=1), vc: ti.any_arr(element_dim=1)):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5
        # velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.any_arr(), new_pf: ti.any_arr(), vd: ti.any_arr()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = vd[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.any_arr(element_dim=1), pf: ti.any_arr()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


@ti.kernel
def enhance_vorticity(vf: ti.any_arr(element_dim=1), cf: ti.any_arr(element_dim=1)):
    # anti-physics visual enhancement...
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb),
                           abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = min(max(vf[i, j] + force * dt, -1e3), 1e3)


@ti.kernel
def apply_pressure(p_in: ti.any_arr(element_dim=1), p_out: ti.any_arr(element_dim=1)):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * res + I[1]]

@ti.kernel
def dye_to_image(df: ti.any_arr(element_dim=1), di:ti.any_arr(element_dim=1)):
    for i, j in df:
        r = df[i,j][0]
        g = df[i,j][1]
        b = df[i,j][2]
        di[i, j] = ti.Vector([r, g, b, 1.0])


def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt, velocity_divs)
        pressures_pair.swap()


def step(mouse_data):
    # generate_mouse_data(touch)
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur)

    divergence(velocities_pair.cur, velocity_divs)

    if curl_strength:
        vorticity(velocities_pair.cur, velocity_curls)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    solve_pressure_jacobi()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)


@ti.kernel
def generate_mouse_data(touch: ti.any_arr(element_dim=0)):
    direction = [ti.random(), ti.random()]
    direction[0] = direction[0]/ti.sqrt(direction[0]*direction[0] + direction[1]*direction[1])
    direction[1] = direction[1]/ti.sqrt(direction[0]*direction[0] + direction[1]*direction[1])

    color = [ti.random(), ti.random(), ti.random(), 0.0]
    mouse_data[0] = direction[0]
    mouse_data[1] = direction[1]
    mouse_data[2] = touch[0]
    mouse_data[3] = touch[1]
    mouse_data[4] = color[0]
    mouse_data[5] = color[1]
    mouse_data[6] = color[2]
    mouse_data[7] = color[3]


class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)

# visualize_d = True  #visualize dye (default)
# visualize_v = False  #visualize velocity
# visualize_c = False  #visualize curl

# gui = ti.GUI('Stable Fluid', (res, res))
# md_gen = MouseDataGen()

# while gui.running:
#     if gui.get_event(ti.GUI.PRESS):
#         e = gui.event
#         if e.key == ti.GUI.ESCAPE:
#             break
#         elif e.key == 'r':
#             paused = False
#             reset()
#         elif e.key == 's':
#             if curl_strength:
#                 curl_strength = 0
#             else:
#                 curl_strength = 7
#         elif e.key == 'g':
#             gravity = not gravity
#         elif e.key == 'v':
#             visualize_v = True
#             visualize_c = False
#             visualize_d = False
#         elif e.key == 'd':
#             visualize_d = True
#             visualize_v = False
#             visualize_c = False
#         elif e.key == 'c':
#             visualize_c = True
#             visualize_d = False
#             visualize_v = False
#         elif e.key == 'p':
#             paused = not paused
#         elif e.key == 'd':
#             debug = not debug

#     if not paused:
#         mouse_data.from_numpy(md_gen(gui))
#         # mouse_data = md_gen(gui)
#         step(mouse_data)
#     if visualize_c:
#         vorticity(velocities_pair.cur, velocity_curls)
#         gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
#     elif visualize_d:
#         dye_to_image(dyes_pair.cur, _dye_image_buffer)
#         gui.set_image(_dye_image_buffer.to_numpy())
#     elif visualize_v:
#         gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
#     gui.show()

# Run the AOT module builder
def aot():
    m = ti.aot.Module(ti.vulkan)
    m.add_kernel(advect, (velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt))
    m.add_kernel(apply_impulse, (velocities_pair.cur, dyes_pair.cur))
    m.add_kernel(divergence, (velocities_pair.cur, velocity_divs))
    m.add_kernel(pressure_jacobi, (pressures_pair.cur, pressures_pair.nxt, velocity_divs))
    m.add_kernel(subtract_gradient, (velocities_pair.cur, pressures_pair.cur))
    m.add_kernel(dye_to_image, (dyes_pair.cur, _dye_image_buffer))
    m.add_field('mouse_data',mouse_data)
    m.add_kernel(generate_mouse_data, (touch,))
    dir_name = 'stable_fluids_aot'
    m.save(dir_name, 'stable_fluids')

aot()