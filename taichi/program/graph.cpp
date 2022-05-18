#include "taichi/program/graph.h"
#include "taichi/program/kernel.h"
#include "taichi/aot/module_builder.h"
#include "spdlog/fmt/fmt.h"

#include <fstream>

namespace taichi {
namespace lang {

void Dispatch::compile() {
  compiled_kernel_ = kernel_->compile_to_aot_kernel();
}

void Dispatch::run(RuntimeContext *ctx,
                   std::unordered_map<std::string, IValue> &args) {
  TI_ERROR_IF(!compiled_kernel_, "Did you forget to call graph.compile()?");

  memset(ctx, 0, sizeof(RuntimeContext));
  for (int i = 0; i < symbolic_args_.size(); ++i) {
    auto &symbolic_arg = symbolic_args_[i];
    auto found = args.find(symbolic_arg.name);
    TI_ERROR_IF(found == args.end(), "Missing runtime value for ",
                symbolic_arg.name);
    IValue &ival = found->second;
    if (ival.tag == ArgKind::NDARRAY) {
      Ndarray *arr = reinterpret_cast<Ndarray *>(ival.val);
      TI_ERROR_IF((symbolic_arg.tag != ival.tag) ||
                      (symbolic_arg.element_shape != arr->shape),
                  "Mismatched shape information for argument",
                  symbolic_arg.name);
      set_runtime_ctx_ndarray(ctx, i, arr);
    } else {
      TI_ERROR_IF(symbolic_arg.tag != ArgKind::SCALAR,
                  "Required a scalar for argument", symbolic_arg.name);
      ctx->set_arg(i, ival.val);
    }
  }

  compiled_kernel_->launch(ctx);
}

void Sequential::compile() {
  // In the future we can do more across-kernel optimization here.
  for (Node *n : sequence_) {
    n->compile();
  }
}

void Sequential::append(Node *node) {
  sequence_.push_back(node);
}

void Sequential::emplace(Kernel *kernel, const std::vector<Arg> &args) {
  Node *n = owning_graph_->create_dispatch(kernel, args);
  sequence_.push_back(n);
}

void Sequential::run(RuntimeContext *ctx,
                     std::unordered_map<std::string, IValue> &args) {
  for (Node *n : sequence_) {
    n->run(ctx, args);
  }
}

Graph::Graph(std::string name) : name_(name) {
  seq_ = std::make_unique<Sequential>(this);
}
Node *Graph::create_dispatch(Kernel *kernel, const std::vector<Arg> &args) {
  all_nodes_.push_back(std::make_unique<Dispatch>(kernel, args));
  return all_nodes_.back().get();
}

Sequential *Graph::create_sequential() {
  all_nodes_.push_back(std::make_unique<Sequential>(this));
  auto res = static_cast<Sequential *>(all_nodes_.back().get());

  return res;
}

void Graph::compile() {
  seq()->compile();
}

Sequential *Graph::seq() const {
  return seq_.get();
}

void Graph::emplace(Kernel *kernel, const std::vector<Arg> &args) {
  seq()->emplace(kernel, args);
}

void Graph::run(std::unordered_map<std::string, IValue> &args) {
  RuntimeContext host_ctx;
  memset(&host_ctx, 0, sizeof(RuntimeContext));
  seq_->run(&host_ctx, args);
}

}  // namespace lang
}  // namespace taichi
