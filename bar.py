import taichi as ti
import numpy as np

ti.init(arch=ti.cuda, ad_stack_size=512, print_ir=True)

# def test_shfl_xor_i32():
#     a = ti.field(dtype=ti.i32, shape=32)

#     @ti.kernel
#     def foo():
#         ti.loop_config(block_dim=32)
#         for i in range(32):
#             for j in range(5):
#                 offset = 1 << j
#                 a[i] += ti.simt.warp.shfl_xor_i32(ti.u32(0xFFFFFFFF), a[i], offset)

#     value = 0
#     for i in range(32):
#         a[i] = i
#         value += i

#     foo()


#     for i in range(32):
#         print(a[i], value)
#         # assert a[i] == value
def test_shfl_xor_i32():
    a = ti.field(dtype=ti.i32, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=32)
        for i in range(32):
            b = 32 - i
            c = ti.simt.warp.shfl_xor_i32(ti.u32(0xFFFFFFFF), b, 1)
            print(i, c)

    # value = 0
    # for i in range(32):
    #     a[i] = i
    #     # value += i

    foo()

    # for i in range(32):
    #     print(a[i], value)
    #     # assert a[i] == value


test_shfl_xor_i32()
