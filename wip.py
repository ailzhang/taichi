import taichi as ti

ti.init(ti.cpu)

n = 4

# vec3f = ti.types.vector(3, float)
line3f = ti.types.struct(linedir=float, length=float)


@ti.kernel
def init(x: line3f):
    print(x.length)
    print(x.linedir)


# v = vec3f(1)
line = line3f(linedir=3.0, length=2.0)
init(line)
