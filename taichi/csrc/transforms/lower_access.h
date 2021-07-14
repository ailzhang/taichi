#pragma once

#include "taichi/csrc/ir/pass.h"

namespace taichi {
namespace lang {

class LowerAccessPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    std::vector<SNode *> kernel_forces_no_activate;
    bool lower_atomic;
  };
};

}  // namespace lang
}  // namespace taichi
