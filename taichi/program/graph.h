#pragma once

#include <string>
#include <vector>
#include <unordered_set>

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/ir/type.h"
#include "taichi/aot/module_loader.h"
#include "taichi/program/graph_data.h"

namespace taichi {
namespace lang {
class Kernel;
class Graph;
struct IValue;

class Node {
 public:
  Node() = default;
  virtual ~Node() = default;
  Node(const Node &) = delete;
  Node &operator=(const Node &) = delete;
  Node(Node &&) = default;
  Node &operator=(Node &&) = default;
  virtual void compile() = 0;
  virtual void run(RuntimeContext *ctx,
                   std::unordered_map<std::string, IValue> &args) = 0;
};
class Dispatch : public Node {
 public:
  explicit Dispatch(Kernel *kernel, const std::vector<Arg> &args)
      : kernel_(kernel), symbolic_args_(args) {
  }
  void compile() override;
  void run(RuntimeContext *ctx,
           std::unordered_map<std::string, IValue> &args) override;

 private:
  Kernel *kernel_{nullptr};
  std::unique_ptr<aot::Kernel> compiled_kernel_;
  std::vector<Arg> symbolic_args_;
};

class Sequential : public Node {
 public:
  explicit Sequential(Graph *graph) : owning_graph_(graph) {
  }
  void append(Node *node);
  void emplace(Kernel *kernel, const std::vector<Arg> &args);
  void compile() override;
  void run(RuntimeContext *ctx,
           std::unordered_map<std::string, IValue> &args) override;

 private:
  std::vector<Node *> sequence_;
  Graph *owning_graph_{nullptr};
};

enum NodeKind { DISPATCH, SEQUENTIAL, COND };

class Graph {
 public:
  Graph(std::string name);
  // TODO: compile can take in Arch argument
  void compile();
  void bind();
  void run(std::unordered_map<std::string, IValue> &args);
  Node *create_dispatch(Kernel *kernel, const std::vector<Arg> &args);
  Sequential *create_sequential();
  void emplace(Kernel *kernel, const std::vector<Arg> &args);
  Sequential *seq() const;

 private:
  std::string name_;
  std::unique_ptr<Sequential> seq_{nullptr};
  std::vector<std::unique_ptr<Node>> all_nodes_;
  std::unordered_map<std::string, IValue> bound_args;
};

}  // namespace lang
}  // namespace taichi
