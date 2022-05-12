#include "taichi/aot/module_builder.h"
#include "taichi/program/kernel.h"

namespace taichi {
namespace lang {

void AotModuleBuilder::add(const std::string &identifier, Kernel *kernel) {
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend(identifier, kernel);
}

void AotModuleBuilder::add_field(const std::string &identifier,
                                 const SNode *rep_snode,
                                 bool is_scalar,
                                 DataType dt,
                                 std::vector<int> shape,
                                 int row_num,
                                 int column_num) {
  add_field_per_backend(identifier, rep_snode, is_scalar, dt, shape, row_num,
                        column_num);
}

void AotModuleBuilder::add_kernel_template(const std::string &identifier,
                                           const std::string &key,
                                           Kernel *kernel) {
  if (!kernel->lowered() && Kernel::supports_lowering(kernel->arch)) {
    kernel->lower();
  }
  add_per_backend_tmpl(identifier, key, kernel);
}

bool AotModuleBuilder::all_fields_are_dense_in_container(
    const SNode *container) {
  for (const auto &ch : container->ch) {
    if (ch->type != SNodeType::place) {
      return false;
    }
  }
  const auto *parent = container->parent;
  if (!parent) {
    return false;
  }
  if (parent->type != SNodeType::root) {
    return false;
  }
  return true;
}

void AotModuleBuilder::load(const std::string &output_dir) {
  TI_ERROR("Aot loader not supported");
}

void AotModuleBuilder::dump_graph(std::string output_dir) const {
  for (auto &graph : graphs_) {
    std::string graph_path =
        fmt::format("{}/graph_{}.txt", output_dir, graph.first);
    // graph.second->save(graph_path);
    // graph.second->add_to_aot_module(this);
    std::ofstream fs(graph_path);

    for (int j = 0; j < graph.second.dispatch_seq.size(); ++j) {
      auto &dispatch = graph.second.dispatch_seq[j];
      std::string args;
      for (int i = 0; i < dispatch.args.size(); ++i) {
        args += dispatch.args[i].name;
        if (i != dispatch.args.size() - 1) {
          args += ", ";
        }
      }
      std::string dispatch_str = fmt::format("{}({})", dispatch.name, args);
      fs << dispatch_str << std::endl;
    }
    fs.close();
  }
  // aot_builder->dump(dst_folder, "");
}

void AotModuleBuilder::add_graph(std::string name,
                                 const aot::DispatchSeq &seq) {
  if (graphs_.count(name) != 0) {
    TI_ERROR("Graph {} already exists", name);
  }
  graphs_[name] = seq;
}
}  // namespace lang
}  // namespace taichi
