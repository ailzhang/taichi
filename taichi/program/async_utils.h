#pragma once

#include <unordered_map>
#include <unordered_set>
#include <atomic>

#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

struct TaskMeta;

class IRHandle {
 public:
  IRHandle() : ir_(nullptr), hash_(0) {
  }

  IRHandle(const IRNode *ir, uint64 hash) : ir_(ir), hash_(hash) {
  }

  std::unique_ptr<IRNode> clone() const;

  const IRNode *ir() const {
    return ir_;
  }

  uint64 hash() const {
    return hash_;
  }

  bool empty() const {
    return ir_ == nullptr;
  }

  // Two IRHandles are considered the same iff their hash values are the same.
  bool operator==(const IRHandle &other_ir_handle) const {
    return hash_ == other_ir_handle.hash_;
  }

  bool operator<(const IRHandle &other_ir_handle) const {
    return hash_ < other_ir_handle.hash_;
  }

 private:
  const IRNode *ir_;  // not owned
  uint64 hash_;
};

// Records the necessary data for launching an offloaded task.
class TaskLaunchRecord {
 public:
  Context context;
  Kernel *kernel;  // TODO: remove this
  IRHandle ir_handle;
  int id;

  TaskLaunchRecord();

  TaskLaunchRecord(Context context, Kernel *kernel, IRHandle ir_handle);

  OffloadedStmt *stmt() const;

  bool empty() const;

 private:
  static std::atomic<int> task_counter;
};

struct AsyncState {
  enum class Type { mask, value, list };

  AsyncState(SNode *snode, Type type) : snode(snode), type(type) {
  }

  SNode *snode;
  Type type;

  bool operator<(const AsyncState &other) const {
    return snode < other.snode || (snode == other.snode && type < other.type);
  }

  bool operator==(const AsyncState &other) const {
    return snode == other.snode && type == other.type;
  }

  std::string name() const {
    std::string type_name;
    switch (type) {
      case Type::mask:
        type_name = "mask";
        break;
      case Type::value:
        type_name = "value";
        break;
      case Type::list:
        type_name = "list";
        break;
    }
    return snode->get_node_type_name_hinted() + "_" + type_name;
  }
};

struct TaskFusionMeta {
  // meta for task fusion
  OffloadedStmt::TaskType type{OffloadedStmt::TaskType::serial};
  SNode *snode{nullptr};  // struct-for only
  int block_dim{0};       // struct-for only
  int32 begin_value{0};   // range-for only
  int32 end_value{0};     // range-for only

  // Merging kernels with different signatures will break invariants.
  // E.g.
  // https://github.com/taichi-dev/taichi/blob/a6575fb97557267e2f550591f43b183076b72ac2/taichi/transforms/type_check.cpp#L326
  //
  // TODO: we could merge different kernels if their args are the
  // same. But we have no way to check that for now.

  // We use nullptr for kernels without arguments or return value.
  // Otherwise, we only fuse tasks with the same kernel.
  Kernel *kernel{nullptr};

  // If fusible is false, this task can't be fused with any other tasks.
  bool fusible{false};

  bool operator==(const TaskFusionMeta &other) const {
    return type == other.type && snode == other.snode &&
           block_dim == other.block_dim && begin_value == other.begin_value &&
           end_value == other.end_value && kernel == other.kernel &&
           fusible == other.fusible;
  }

  bool operator!=(const TaskFusionMeta &other) const {
    return !(*this == other);
  }
};

TLANG_NAMESPACE_END

namespace std {
template <>
struct hash<taichi::lang::IRHandle> {
  std::size_t operator()(const taichi::lang::IRHandle &ir_handle) const
      noexcept {
    return ir_handle.hash();
  }
};

template <>
struct hash<std::pair<taichi::lang::IRHandle, taichi::lang::IRHandle>> {
  std::size_t operator()(
      const std::pair<taichi::lang::IRHandle, taichi::lang::IRHandle>
          &ir_handles) const noexcept {
    return ir_handles.first.hash() * 100000007UL + ir_handles.second.hash();
  }
};

template <>
struct hash<taichi::lang::AsyncState> {
  std::size_t operator()(const taichi::lang::AsyncState &s) const noexcept {
    return (std::size_t)s.snode ^ (std::size_t)s.type;
  }
};

template <>
struct hash<taichi::lang::TaskFusionMeta> {
  std::size_t operator()(const taichi::lang::TaskFusionMeta &t) const noexcept {
    std::size_t result = (t.type << 1) ^ t.fusible ^ (std::size_t)t.kernel;
    result ^= (std::size_t)t.block_dim * 100000007UL + (std::size_t)t.snode;
    result ^= ((std::size_t)t.begin_value << 32) ^ t.end_value;
    return result;
  }
};

}  // namespace std

TLANG_NAMESPACE_BEGIN

struct TaskMeta {
  std::string name;
  OffloadedStmt::TaskType type{OffloadedStmt::TaskType::serial};
  SNode *snode{nullptr};  // struct-for and listgen only
  std::unordered_set<AsyncState> input_states;
  std::unordered_set<AsyncState> output_states;
  std::unordered_map<SNode *, bool> element_wise;

  void print() const;
};

class IRBank;

TaskMeta *get_task_meta(IRBank *bank, const TaskLaunchRecord &t);

TaskFusionMeta get_task_fusion_meta(IRBank *bank, const TaskLaunchRecord &t);

TLANG_NAMESPACE_END
