import taichi as ti

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

@ti.kernel
def substep_reset_grid(grid_v : ti.any_arr(element_dim=1), grid_m : ti.any_arr()):
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

@ti.kernel
def substep_p2g(x : ti.any_arr(element_dim=1),
                v : ti.any_arr(element_dim=1),
                C : ti.any_arr(element_dim=2),
                J : ti.any_arr(),
                grid_v : ti.any_arr(element_dim=1),
                grid_m : ti.any_arr()):
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

@ti.kernel
def substep_update_grid_v(grid_v : ti.any_arr(element_dim=1), grid_m : ti.any_arr()):
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0

@ti.kernel
def substep_g2p(x : ti.any_arr(element_dim=1),
                v : ti.any_arr(element_dim=1),
                C : ti.any_arr(element_dim=2),
                J : ti.any_arr(),
                grid_v : ti.any_arr(element_dim=1)):
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

@ti.kernel
def generate_vbo(x : ti.any_arr(element_dim=1), vertices : ti.any_arr(element_dim=1)):
    for p in x:
        vertices[p] = x[p]

@ti.kernel
def init_particles(x : ti.any_arr(element_dim=1),
                   v : ti.any_arr(element_dim=1),
                   J : ti.any_arr(),
                   x_init : ti.any_arr(element_dim=1),
                   v_init : ti.any_arr(element_dim=1)):
    for i in range(n_particles):
        x[i] = x_init[i]
        v[i] = v_init[i]
        J[i] = 1

# g = cgraph.Module()

g_init = taichi.Graph('init')
g_init.emplace(init_particles, cgraph.Arg('x'), cgraph.Arg('v'), cgraph.Arg('J'), cgraph.Arg('x_init'), cgraph.Arg('v_init'))

g_update = taichi.Graph('update')
substep = g_init.create_sequential()
substep.emplace(substep_reset_grid, cgraph.Arg('grid_v'), cgraph.Arg('grid_m'))
substep.emplace(substep_p2g, cgraph.Arg('x'), cgraph.Arg('v'), cgraph.Arg('C'), cgraph.Arg('J'), cgraph.Arg('grid_v'), cgraph.Arg('grid_m'))
substep.emplace(substep_update_grid_v, cgraph.Arg('grid_v'), cgraph.Arg('grid_m'))
substep.emplace(substep_g2p, cgraph.Arg('x'), cgraph.Arg('v'), cgraph.Arg('C'), cgraph.Arg('J'), cgraph.Arg('grid_v'))

# Static loop??
# g_update = g.new_graph('update')

for i in range(50):
    g_update.append(substep)

g_update.emplace(generate_vbo, cgraph.Arg('x'), cgraph.Arg('vertices'))

g_init.compile()
g_update.compile()

import numpy as np

ti.init(arch=ti.vulkan)

vertices_np = (np.random.random((n_particles, 2)) * 0.4 + 0.2).astype(np.single)
init_v_np = np.zeros((n_particles, 2)).astype(np.single)
vertices = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))
init_v = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))
vertices.from_numpy(vertices_np)
init_v.from_numpy(init_v_np)

x = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))
v = ti.Vector.ndarray(2, ti.f32, shape=(n_particles))
C = ti.Matrix.ndarray(2, 2, ti.f32, shape=(n_particles))
J = ti.ndarray(ti.f32, shape=(n_particles))
grid_v = ti.Vector.ndarray(2, ti.f32, shape=(n_grid, n_grid))
grid_m = ti.ndarray(ti.f32, shape=(n_grid, n_grid))

# compiled_g_init = g.get_graph('init')
# compiled_g_init.bind()
g_init.run(('x_init', vertices), ('v_init', init_v), ('x', x), ('v', v), ('J', J))

# g_update = ti.Graph('update')
# TODO: support graph.bind()
# g_update.bind(('x', x), ('v', v), ('C', C), ('J', J), ('grid_v', grid_v), ('grid_m', grid_m))

gui = ti.GUI('MPM88')
while gui.running:
    g_update.run(('vertices', vertices),('x', x), ('v', v), ('C', C), ('J', J), ('grid_v', grid_v), ('grid_m', grid_m))
    gui.clear(0x112F41)
    gui.circles(vertices.to_numpy(), radius=1.5, color=0x068587)
    gui.show()
