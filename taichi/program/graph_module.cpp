#include "taichi/program/graph_module.h"
// #include "taichi/program/kernel.h"
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

  void Dispatch::save(std::ofstream& fs) {
    std::string args;
    for (int i = 0; i < symbolic_args_.size(); ++i) {
      args += symbolic_args_[i].name;
      if (i != symbolic_args_.size() - 1) {
        args += ", ";
      }
    }
    std::string dispatch_str = fmt::format("{}({})", name, args);
    fs << dispatch_str << std::endl;
  }

  void Dispatch::add_to_aot_module(AotModuleBuilder* aot_builder) const {
    aot_builder->add(name, kernel_);
  }

  void Sequential::compile() {
    // In the future we can do more across-kernel optimization here.
    for (Node* n: sequence_) {
      n->compile();
    }
  }

  void Sequential::save(std::ofstream& fs) {
    for (const auto& node: sequence_) {
      node->save(fs);
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

  void Sequential::add_to_aot_module(AotModuleBuilder* aot_builder) const {
    // No-op
  }


  Graph::Graph() {
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

  Sequential* Graph::seq() {
    return seq_.get();
  }

  void Graph::run(std::unordered_map<std::string, IValue>& args) {
    seq_->run(args);
  }

  void Graph::save(std::string dst_file) {
    std::ofstream fs(dst_file);
    seq()->save(fs);
    fs.close();
  }

  void Graph::add_to_aot_module(AotModuleBuilder* aot_builder)  const{
    for (const auto& node: all_nodes_) {
      node->add_to_aot_module(aot_builder);
    }
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

  void GraphModule::save(std::string dst_folder, Program* prog) {
    auto aot_builder = prog->make_aot_module_builder(Arch::vulkan);

    for (auto& graph: graphs_) {
      std::string graph_path = fmt::format("{}/graph_{}.txt", dst_folder, graph.first);
      graph.second->save(graph_path); 
      graph.second->add_to_aot_module(aot_builder.get());
    }
    aot_builder->dump(dst_folder, "");
  }
}
}
