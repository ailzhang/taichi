#pragma once
#include <string>
#include <vector>
#include "taichi/ir/type.h"
#include "taichi/inc/constants.h"

namespace taichi {
namespace lang {
class Ndarray;

enum ArgKind { SCALAR, NDARRAY, UNKNOWN };
struct Arg {
  std::string name;
  // TODO: real element dtype = dtype + element_shape
  std::string dtype_name;
  ArgKind tag;
  std::vector<int> element_shape;

  TI_IO_DEF(name, dtype_name, tag, element_shape);
};

struct IValue {
 public:
  uint64 val;
  ArgKind tag;

  static IValue create(const Ndarray &ndarray) {
    return IValue(reinterpret_cast<intptr_t>(&ndarray), ArgKind::NDARRAY);
  }

  template <typename T>
  static IValue create(T v) {
    return IValue(taichi_union_cast_with_different_sizes<uint64>(v),
                  ArgKind::SCALAR);
  }

 private:
  IValue(uint64 val, ArgKind tag) : val(val), tag(tag) {
  }
};

}  // namespace lang
}  // namespace taichi
