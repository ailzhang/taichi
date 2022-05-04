#include "taichi/program/graph_module.h"
// #include "taichi/program/kernel.h"

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
      } else {
        launch_ctx.set_arg_raw(i, ival.val);
      }
    }
    (*kernel_)(launch_ctx);
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


  Graph::Graph() {
    seq_ = std::make_unique<Sequential>(this);
  }
  Node* Graph::create_dispatch(Kernel *kernel, const std::vector<Arg>& args) {
    Node* n = new Dispatch(kernel, args);
    all_nodes_.insert(n);
    return n;
  }
  
  void Graph::compile() {
    seq()->compile();
  }

  Sequential* Graph::seq() {
    return seq_.get();
  }

  void Graph::run(std::unordered_map<std::string, IValue>& args) {
    seq_->run(args);
  }

  Sequential* GraphModule::new_graph(std::string name) {
    TI_ASSERT(graphs_.count(name) == 0);
    graphs_[name] = std::make_unique<Graph>();
    return graphs_[name]->seq();
  }

  Graph* GraphModule::get_graph(std::string name) {
    TI_ASSERT(graphs_.count(name) != 0);
    return graphs_[name].get();
  }

  void GraphModule::compile() {
    for (auto & graph: graphs_) {
      graph.second->compile();
    }
  }
}
}
