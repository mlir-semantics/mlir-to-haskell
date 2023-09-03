#include <optional>

// #include "llvm/IR/InstVisitor.h"
#include "llvm/ADT/StringSet.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace {

typedef llvm::raw_ostream stream_t; 

struct ComputeDialectsPass
    : public PassWrapper<ComputeDialectsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComputeDialectsPass)

  StringRef getArgument() const final { return "compute-dialects"; }
  StringRef getDescription() const final { return "Computes the dialect used within each region, and save these as attributes."; }

  // Entry point for the pass.
  void runOnOperation() override {}
};
} // namespace

namespace mlir {
void registerComputeDialectsPass() {
  PassRegistration<ComputeDialectsPass>();
}
} // namespace mlir