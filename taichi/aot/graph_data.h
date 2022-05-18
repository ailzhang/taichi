#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "taichi/program/graph_utils.h"
#include "taichi/aot/module_data.h"

namespace taichi {
namespace lang {
class AotModuleBuilder;
namespace aot {
class TI_DLL_EXPORT Kernel {
 public:
  // Rule of 5 to make MSVC happy
  Kernel() = default;
  virtual ~Kernel() = default;
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;
  Kernel(Kernel &&) = default;
  Kernel &operator=(Kernel &&) = default;

  /**
   * @brief Launches the kernel to the device
   *
   * This does not manage the device to host synchronization.
   *
   * @param ctx Host context
   */
  virtual void launch(RuntimeContext *ctx) = 0;

  virtual void save_to_module(AotModuleBuilder *builder) {
    TI_NOT_IMPLEMENTED;
  }
};

struct CompiledDispatch {
  std::string kernel_name;
  std::vector<Arg> symbolic_args;
  Kernel *compiled_kernel{nullptr};

  TI_IO_DEF(kernel_name, symbolic_args);
};

struct CompiledGraph {
  std::vector<CompiledDispatch> dispatches;

  TI_IO_DEF(dispatches);
};

}  // namespace aot
}  // namespace lang
}  // namespace taichi
