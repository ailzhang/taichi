#pragma once
#include <string>
#include <vector>
#include "taichi/ir/type.h"

namespace taichi {
namespace lang {
class Ndarray;

enum ArgKind { SCALAR, NDARRAY };
struct Arg {
  std::string name;
  // TODO: real element dtype = dtype + element_shape
  DataType dtype;
  ArgKind tag;
  std::vector<int> element_shape;

  Arg(std::string name, const DataType dtype)
      : name(name), dtype(dtype), tag(ArgKind::SCALAR) {
  }

  Arg(std::string name,
      const DataType dtype,
      const std::vector<int> &element_shape)
      : name(name),
        dtype(dtype),
        tag(ArgKind::NDARRAY),
        element_shape(element_shape) {
  }
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
