#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/system/profiler.h"

#include <deque>
#include <set>

namespace taichi::lang {

class DemoteAtomics : public BasicStmtVisitor {
 private:
  std::unordered_map<const SNode *, GlobalPtrStmt *> loop_unique_ptr_;
  std::unordered_map<int, ExternalPtrStmt *> loop_unique_arr_ptr_;

 public:
  using BasicStmtVisitor::visit;

  OffloadedStmt *current_offloaded;
  RangeForStmt *current_range_for{nullptr};
  DelayedIRModifier modifier;

  DemoteAtomics() {
    current_offloaded = nullptr;
    current_range_for = nullptr;
  }

  void visit(RangeForStmt *stmt) override {
    preprocess_container_stmt(stmt);
    if (current_offloaded) {
      current_range_for = stmt;
    }
    if (stmt->body) {
      stmt->body->accept(this);
    }
    if (current_offloaded) {
      current_range_for = nullptr;
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    bool demote = false;
    bool is_local = false;

    if (current_offloaded) {
      if (stmt->dest->is<ThreadLocalPtrStmt>()) {
        demote = true;
      }
      if (current_offloaded->task_type == OffloadedTaskType::serial) {
        demote = true;
      }
      if (!demote &&
          (current_offloaded->task_type == OffloadedTaskType::range_for ||
           current_offloaded->task_type == OffloadedTaskType::mesh_for ||
           current_offloaded->task_type == OffloadedTaskType::struct_for)) {
        // Handle loop-unique GlobalPtrStmt
        bool is_global_ptr_stmt = false;
        GlobalPtrStmt *dest = nullptr;
        if (stmt->dest->is<GlobalPtrStmt>()) {
          is_global_ptr_stmt = true;
          dest = stmt->dest->as<GlobalPtrStmt>();
        } else if (stmt->dest->is<MatrixPtrStmt>() &&
                   stmt->dest->as<MatrixPtrStmt>()
                       ->origin->is<GlobalPtrStmt>()) {
          is_global_ptr_stmt = true;
          dest = stmt->dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
        }

        if (is_global_ptr_stmt) {
          demote = true;
          auto snode = dest->snode;
          if (loop_unique_ptr_[snode] == nullptr ||
              loop_unique_ptr_[snode]->indices.empty()) {
            // not uniquely accessed
            demote = false;
          }
          if (current_offloaded->mem_access_opt.has_flag(
                  snode, SNodeAccessFlag::block_local) ||
              current_offloaded->mem_access_opt.has_flag(
                  snode, SNodeAccessFlag::mesh_local)) {
            // BLS does not support write access yet so we keep atomic_adds.
            demote = false;
          }
          // demote from-end atomics
          if (current_offloaded->task_type == OffloadedTaskType::mesh_for) {
            if (dest->indices.size() == 1 &&
                dest->indices[0]->is<MeshIndexConversionStmt>()) {
              auto idx = dest->indices[0]->as<MeshIndexConversionStmt>()->idx;
              while (idx->is<MeshIndexConversionStmt>()) {  // special case: l2g
                                                            // + g2r
                idx = idx->as<MeshIndexConversionStmt>()->idx;
              }
              if (idx->is<LoopIndexStmt>() &&
                  idx->as<LoopIndexStmt>()->is_mesh_index() &&
                  loop_unique_ptr_[dest->snode] != nullptr) {
                demote = true;
              }
            }
          }
        }

        // Handle loop-unique ExternalPtrStmt
        bool is_external_ptr_stmt = false;
        ExternalPtrStmt *dest_ptr = nullptr;
        if (stmt->dest->is<ExternalPtrStmt>()) {
          is_external_ptr_stmt = true;
          dest_ptr = stmt->dest->as<ExternalPtrStmt>();
        } else if (stmt->dest->is<MatrixPtrStmt>() &&
                   stmt->dest->as<MatrixPtrStmt>()
                       ->origin->is<ExternalPtrStmt>()) {
          is_external_ptr_stmt = true;
          dest_ptr =
              stmt->dest->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
        }

        if (is_external_ptr_stmt) {
          demote = true;
          if (dest_ptr->indices.empty()) {
            demote = false;
          }
          ArgLoadStmt *arg_load_stmt = dest_ptr->base_ptr->as<ArgLoadStmt>();
          int arg_id = arg_load_stmt->arg_id;
          if (loop_unique_arr_ptr_[arg_id] == nullptr) {
            // Not loop unique
            demote = false;
          }
          // TODO: Is BLS / Mem Access Opt a thing for any_arr?
        }
      }
    }

    if (stmt->dest->is<AllocaStmt>()) {
      // Except shared array
      if (!stmt->dest->as<AllocaStmt>()->is_shared) {
        demote = true;
        is_local = true;
      }
    }

    if (stmt->dest->is<MatrixPtrStmt>() &&
        stmt->dest->cast<MatrixPtrStmt>()->origin->is<AllocaStmt>()) {
      // Except shared array
      if (!stmt->dest->cast<MatrixPtrStmt>()
               ->origin->as<AllocaStmt>()
               ->is_shared) {
        demote = true;
        is_local = true;
      }
    }

    if (auto dest_pointer_type = stmt->dest->ret_type->cast<PointerType>()) {
      if (dest_pointer_type->get_pointee_type()->is<QuantFloatType>()) {
        TI_WARN(
            "AtomicOp on QuantFloatType is not supported. "
            "Demoting to non-atomic RMW.\n{}",
            stmt->tb);
        demote = true;
      }
    }

    if (demote) {
      if (current_range_for && stmt->dest->is<GlobalPtrStmt>() && !is_local &&
          stmt->op_type == AtomicOpType::add) {
        // if (false) {
        for (auto &st : stmt->dest->as<GlobalPtrStmt>()->indices) {
          // FIXME: inside the range for.
          if (st->parent == current_range_for->body.get()) {
            return;
          }
        }
        std::cout << "MY HACK" << std::endl;
        // insert alloca to parent
        auto before_for = VecStatement();
        auto alloca = before_for.push_back<AllocaStmt>(stmt->ret_type);
        // insert add inside range_for, replace old
        auto inside_for = VecStatement();
        auto add = inside_for.push_back<AtomicOpStmt>(stmt->op_type, alloca,
                                                      stmt->val);
        stmt->replace_usages_with(add);
        // insert add_back after the block
        auto after_for = VecStatement();
        auto load = after_for.push_back<LocalLoadStmt>(alloca);
        // auto add_back
        auto new_ptr = after_for.push_back(stmt->dest->clone());
        after_for.push_back<AtomicOpStmt>(stmt->op_type, new_ptr, load);
        modifier.insert_before(current_range_for, std::move(before_for));
        modifier.insert_before(stmt, std::move(inside_for));
        std::cout << stmt->parent << " " << add->parent << std::endl;
        modifier.insert_after(current_range_for, std::move(after_for));
        modifier.erase(stmt);
        return;
      }
      // replace atomics with load, add, store
      auto bin_type = atomic_to_binary_op_type(stmt->op_type);
      auto ptr = stmt->dest;
      auto val = stmt->val;

      auto new_stmts = VecStatement();
      Stmt *load;
      if (is_local) {
        load = new_stmts.push_back<LocalLoadStmt>(ptr);
        auto bin = new_stmts.push_back<BinaryOpStmt>(bin_type, load, val);
        new_stmts.push_back<LocalStoreStmt>(ptr, bin);
      } else {
        load = new_stmts.push_back<GlobalLoadStmt>(ptr);
        auto bin = new_stmts.push_back<BinaryOpStmt>(bin_type, load, val);
        new_stmts.push_back<GlobalStoreStmt>(ptr, bin);
      }
      // For a taichi program like `c = ti.atomic_add(a, b)`, the IR looks
      // like the following
      //
      // $c  = # lhs memory
      // $d  = atomic add($a, $b)
      // $e  : store [$c <- $d]
      //
      // If this gets demoted, the IR is translated into:
      //
      // $c  = # lhs memory
      // $d' = load $a             <-- added by demote_atomic
      // $e' = add $d' $b
      // $f  : store [$a <- $e']   <-- added by demote_atomic
      // $g  : store [$c <- ???]   <-- store the old value into lhs $c
      //
      // Naively relying on Block::replace_with() would incorrectly fill $f
      // into ???, because $f is a store stmt that doesn't have a return
      // value. The correct thing is to replace |stmt| $d with the loaded
      // old value $d'.
      // See also: https://github.com/taichi-dev/taichi/issues/332
      stmt->replace_usages_with(load);
      modifier.replace_with(stmt, std::move(new_stmts),
                            /*replace_usages=*/false);
    }
  }

  void visit(OffloadedStmt *stmt) override {
    current_offloaded = stmt;
    if (stmt->task_type == OffloadedTaskType::range_for ||
        stmt->task_type == OffloadedTaskType::mesh_for ||
        stmt->task_type == OffloadedTaskType::struct_for) {
      auto uniquely_accessed_pointers =
          irpass::analysis::gather_uniquely_accessed_pointers(stmt);
      loop_unique_ptr_ = std::move(uniquely_accessed_pointers.first);
      loop_unique_arr_ptr_ = std::move(uniquely_accessed_pointers.second);
    }
    // We don't need to visit TLS/BLS prologues/epilogues.
    if (stmt->body) {
      stmt->body->accept(this);
    }
    current_offloaded = nullptr;
  }

  static bool run(IRNode *node) {
    DemoteAtomics demoter;
    bool modified = false;
    while (true) {
      // std::cout << "running demote once" << std::endl;
      node->accept(&demoter);
      if (demoter.modifier.modify_ir()) {
        modified = true;
      } else {
        break;
      }
    }
    return modified;
  }
};

namespace irpass {

bool demote_atomics(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  bool modified = DemoteAtomics::run(root);
  type_check(root, config);
  return modified;
}

}  // namespace irpass

}  // namespace taichi::lang
