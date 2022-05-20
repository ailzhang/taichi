from taichi.lang._ndarray import ScalarNdarray
from taichi.lang.enums import Layout
from taichi.lang.exception import TaichiCompilationError
from taichi.lang.matrix import MatrixNdarray, VectorNdarray
from taichi.types.annotations import template
from taichi.types.ndarray_type import NdarrayType
from taichi.types.primitive_types import f32


def produce_injected_args(kernel, template_args=None, symbolic_args=None):
    injected_args = []
    template_types = (NdarrayType, template)
    num_template_args = len([
        arg.annotation for arg in kernel.arguments
        if isinstance(arg.annotation, template_types)
    ])
    if template_args is not None and num_template_args != len(template_args):
        raise TaichiCompilationError(
            f'Need {num_template_args} inputs to instantiate the template '
            f'parameters, got {len(template_args)}')
    i = 0
    for j, arg in enumerate(kernel.arguments):
        anno = arg.annotation
        if isinstance(anno, template_types):
            if template_args:
                injected_args.append(template_args[arg.name])
            else:
                if not isinstance(anno, NdarrayType):
                    raise TaichiCompilationError(
                        f'Expected Ndaray type, got {anno}')
                if symbolic_args is not None:
                    anno.element_shape = symbolic_args[j].element_shape
                    anno.element_dim = len(anno.element_shape)

                if anno.element_shape is None or anno.field_dim is None:
                    raise TaichiCompilationError(
                        'Please either specify both `element_shape` and `field_dim` '
                        'in the param annotation, or provide an example '
                        f'ndarray for param={arg.name}')
                if anno.element_dim == 0:
                    injected_args.append(
                        ScalarNdarray(f32, (2, ) * anno.field_dim))
                elif anno.element_dim == 1:
                    injected_args.append(
                        VectorNdarray(anno.element_shape[0],
                                      dtype=f32,
                                      shape=(2, ) * anno.field_dim,
                                      layout=Layout.AOS))
                elif anno.element_dim == 2:
                    injected_args.append(
                        MatrixNdarray(anno.element_shape[0],
                                      anno.element_shape[1],
                                      dtype=f32,
                                      shape=(2, ) * anno.field_dim,
                                      layout=Layout.AOS))
                else:
                    raise RuntimeError('')
            i = i + 1
        else:
            # For primitive types, we can just inject a dummy value.
            injected_args.append(0)
    return injected_args
