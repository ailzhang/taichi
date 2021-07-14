#pragma once

#include "taichi/csrc/ir/pass.h"
#include "taichi/csrc/program/program.h"

namespace taichi {
namespace lang {

class ConstantFoldPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    Program *program;
  };
};

}  // namespace lang
}  // namespace taichi
