from taichi._lib import core as _ti_core
from taichi.aot.utils import produce_injected_args
from taichi.lang import kernel_impl

ArgKind = _ti_core.ArgKind
Arg = _ti_core.Arg


class Sequential:
    def __init__(self, seq):
        self.seq_ = seq

    def emplace(self, kernel_fn, *args):
        kernel = kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        injected_args = produce_injected_args(kernel,
                                              template_args=None,
                                              symbolic_args=args)
        # FIXME: move compilation down to the compile() below
        kernel.ensure_compiled(*injected_args)
        self.seq_.emplace(kernel.kernel_cpp, args)


class Graph:
    def __init__(self, name):
        self._graph = _ti_core.Graph(name)

    def emplace(self, kernel_fn, *args):
        kernel = kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        injected_args = produce_injected_args(kernel,
                                              template_args=None,
                                              symbolic_args=args)
        kernel.ensure_compiled(*injected_args)
        self._graph.emplace(kernel.kernel_cpp, args)

    def create_sequential(self):
        return Sequential(self._graph.create_sequential())

    def append(self, node):
        # FIXME: support this in a cleaner way
        if isinstance(node, Sequential):
            self._graph.seq().append(node.seq_)

    def compile(self):
        self._graph.compile()

    def run(self, args):
        arg_ptrs = {k: args[k].arr for k in args.keys()}
        self._graph.run(arg_ptrs)


__all__ = ['Graph', 'Arg', 'ArgKind']
