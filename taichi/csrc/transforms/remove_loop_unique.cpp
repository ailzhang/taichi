#include "taichi/csrc/ir/ir.h"
#include "taichi/csrc/ir/statements.h"
#include "taichi/csrc/ir/transforms.h"
#include "taichi/csrc/ir/visitors.h"
#include "taichi/csrc/system/profiler.h"

TLANG_NAMESPACE_BEGIN

namespace {

// Remove all the loop_unique statements.

class RemoveLoopUnique : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  void visit(LoopUniqueStmt *stmt) override {
    stmt->replace_with(stmt->input);
    modifier.erase(stmt);
  }

  static bool run(IRNode *node) {
    RemoveLoopUnique pass;
    node->accept(&pass);
    return pass.modifier.modify_ir();
  }
};

}  // namespace

namespace irpass {

bool remove_loop_unique(IRNode *root) {
  TI_AUTO_PROF;
  return RemoveLoopUnique::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
