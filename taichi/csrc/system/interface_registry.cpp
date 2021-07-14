#include <functional>

#include "pybind11/pybind11.h"
#include "taichi/csrc/common/interface.h"
#include "taichi/csrc/common/task.h"
#include "taichi/csrc/system/benchmark.h"

namespace taichi {

TI_INTERFACE_DEF(Benchmark, "benchmark")
TI_INTERFACE_DEF(Task, "task")

}  // namespace taichi
