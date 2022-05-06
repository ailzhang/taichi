#include "taichi/aot/graph.h"
#define TI_RUNTIME_HOST
#include "taichi/common/core.h"
#include "taichi/program/context.h"

namespace taichi {
namespace lang {

namespace aot {
  void Dispatch::run(taichi::lang::RuntimeContext* host_ctx, std::unordered_map<std::string, IValue>& args) {


    for (int i = 0; i < symbolic_args_.size(); ++i) {
      IValue& ival = args.at(symbolic_args_[i].name);
      if (ival.tag == IValue::Tag::DEVALLOC) {
        host_ctx->set_arg_devalloc(i, *(reinterpret_cast<DeviceAllocation*>(ival.val)), ival.shape_);
      } else {
        host_ctx->set_arg<uint64>(i, ival.val);
      }
    }
    kernel_->launch(host_ctx);
  }

  void Sequential::append(Node *node) {
    sequence_.push_back(node);
  }

  void Sequential::emplace(Kernel *kernel, const std::vector<Arg>& args) {
    Node* n = owning_graph_->create_dispatch(kernel, args);
    sequence_.push_back(n);
  }

  void Sequential::run(taichi::lang::RuntimeContext* host_ctx, std::unordered_map<std::string, IValue>& args) {
    for (Node* n: sequence_) {
      n->run(host_ctx, args);
    }
  }

  Graph::Graph() {
    seq_ = std::make_unique<Sequential>(this);
  }

  Node* Graph::create_dispatch(Kernel *kernel, const std::vector<Arg>& args) {
    Node* n = new Dispatch(kernel, args);
    all_nodes_.insert(n);
    return n;
  }
    Sequential* Graph::seq() {
    return seq_.get();
  }

  void Graph::run(std::unordered_map<std::string, IValue>& args) {

    taichi::lang::RuntimeContext host_ctx;
    memset(&host_ctx, 0, sizeof(taichi::lang::RuntimeContext));
    seq_->run(&host_ctx, args);
  }
}
}
}