#include "taichi/csrc/struct/snode_tree.h"

#include "taichi/csrc/struct/struct.h"

namespace taichi {
namespace lang {

SNodeTree::SNodeTree(int id, std::unique_ptr<SNode> root)
    : id_(id), root_(std::move(root)) {
  infer_snode_properties(*root_);
}

}  // namespace lang
}  // namespace taichi
