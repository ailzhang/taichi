#include "taichi/program/graph_module.h"
#include "taichi/program/kernel.h"
#include "taichi/aot/module_builder.h"
#include "spdlog/fmt/fmt.h"

#include <fstream>

namespace taichi {
namespace lang {

  void Dispatch::compile() {
    kernel_->compile();
  }

  void Dispatch::run(std::unordered_map<std::string, IValue>& args) {
    auto launch_ctx = kernel_->make_launch_context();
    for (int i = 0; i < symbolic_args_.size(); ++i) {
      TI_ASSERT(args.count(symbolic_args_[i].name) == 1);
      IValue& ival = args.at(symbolic_args_[i].name);
      if (ival.tag == IValue::Tag::NDARRAY) {
      launch_ctx.set_arg_ndarray(i, *(reinterpret_cast<Ndarray*>(ival.val)));
      } else if (ival.tag == IValue::Tag::DEVALLOC) {
        launch_ctx.set_arg_external_array_with_shape(i, ival.val, ival.size_, ival.shape_);
      } else {
        launch_ctx.set_arg_raw(i, ival.val);
      }
    }
    (*kernel_)(launch_ctx);
  }

  std::vector<aot::SymbolicDispatch> Dispatch::serialize(AotModuleBuilder* aot_builder) const {
    aot_builder->add(kernel_->get_name(), kernel_);
    std::vector<aot::SymbolicArg> arg_names;
    for (const auto& arg: symbolic_args_){
      arg_names.push_back(aot::SymbolicArg{arg.name});
    }
    aot::SymbolicDispatch dispatch{kernel_->get_name(), arg_names};
    
    return {dispatch};
  }

  void Sequential::compile() {
    // In the future we can do more across-kernel optimization here.
    for (Node* n: sequence_) {
      n->compile();
    }
  }

  
  void Sequential::append(Node *node) {
    sequence_.push_back(node);
  }

  void Sequential::emplace(Kernel *kernel, const std::vector<Arg>& args) {
    Node* n = owning_graph_->create_dispatch(kernel, args);
    sequence_.push_back(n);
  }

  void Sequential::run(std::unordered_map<std::string, IValue>& args) {
    for (Node* n: sequence_) {
      n->run(args);
    }
  }

  std::vector<aot::SymbolicDispatch> Sequential::serialize(AotModuleBuilder* aot_builder) const {
    std::vector<aot::SymbolicDispatch> res;
    for (const auto& node: sequence_) {
      auto dispatches = node->serialize(aot_builder);
      res.insert(res.end(), dispatches.begin(), dispatches.end());
    }
    return res;
  }


  Graph::Graph(std::string name) : name_(name) {
    seq_ = std::make_unique<Sequential>(this);
  }
  Node* Graph::create_dispatch(Kernel *kernel, const std::vector<Arg>& args) {
    Node* n = new Dispatch(kernel, args);
    all_nodes_.insert(n);
    return n;
  }
  
  // TODO: compile can take in Arch argument
  void Graph::compile() {
    seq()->compile();
  }

  Sequential* Graph::seq() const {
    return seq_.get();
  }

  void Graph::run(std::unordered_map<std::string, IValue>& args) {
    seq_->run(args);
  }

  void Graph::serialize(AotModuleBuilder* aot_builder)  const{
    aot_builder->add_graph(name_, aot::DispatchSeq{this->seq()->serialize(aot_builder)});
  }
}
}
