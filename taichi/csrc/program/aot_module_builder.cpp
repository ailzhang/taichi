#include "taichi/csrc/program/aot_module_builder.h"

#include "taichi/csrc/program/kernel.h"

namespace taichi {
namespace lang {

void AotModuleBuilder::add(const std::string &identifier, Kernel *kernel) {
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend(identifier, kernel);
}

}  // namespace lang
}  // namespace taichi
