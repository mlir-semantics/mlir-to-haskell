#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace {

/// This pass illustrates the IR nesting through printing.
struct HaskellPrintingPass
    : public PassWrapper<HaskellPrintingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HaskellPrintingPass)

  StringRef getArgument() const final { return "to-haskell"; }
  StringRef getDescription() const final { return "Print into Haskell embedding."; }

  /// The three methods below are mutually recursive and follow the nesting of
  /// the IR: operation->region->block->operation->...

  void printOperation(Operation *op) {
    // print the operation's attached region
    // for (Region &region : op->getRegions())
    //     printRegion(region);

    // // Print the operation itself and some of its properties
	auto& stream = printOpLine(printIndent(), op) << " ";
    stream << "visiting op: '" << op->getName() << "' with "
		   << op->getNumOperands() << " operands and "
		   << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      for (NamedAttribute attr : op->getAttrs())
        printIndent() << " - '" << attr.getName().getValue() << "' : '"
                      << attr.getValue() << "'\n";
    }

    // Recurse into each of the regions attached to the operation.
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    auto indent = pushIndent();
    for (Region &region : op->getRegions())
      printRegion(region);
  }

  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }

  void printBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // Block main role is to hold a list of Operations: let's recurse.
    auto indent = pushIndent();
    for (Operation &op : block.getOperations())
      printOperation(&op);
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };

  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }

  // Entry point for the pass.
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }

private:
  llvm::raw_ostream& printOpLine(llvm::raw_ostream &stream, Operation *op) {
	// print the operation, polymorphic on operation type
	llvm::StringRef opName {op->getName().getStringRef()};

	if (opName.equals("arith.addf")) hp.print(stream, llvm::dyn_cast<arith::AddFOp>(op));
	else if (opName.equals("func.func")) hp.print(stream, llvm::dyn_cast<func::FuncOp>(op));
	else if (opName.equals("func.call")) hp.print(stream, llvm::dyn_cast<func::CallOp>(op));
	else if (opName.equals("func.return")) hp.print(stream, llvm::dyn_cast<func::ReturnOp>(op));
	else stream << "UNIMPLEMENTED OP " << op->getName();

	return stream;
  }

  struct HaskellOpPrinter {
  public:
	void print(llvm::raw_ostream &stream, arith::AddFOp op) {
		stream << "add "; 
		op.getLhs().printAsOperand(stream, OpPrintingFlags());
		stream << " ";
		op.getRhs().printAsOperand(stream, OpPrintingFlags());
	};

	void print(llvm::raw_ostream &stream, func::FuncOp op) {
		// TODO: add function signature
		stream << op.getSymName().str() << " ";

		// print arg names, which are arguments to the first block in the attached region
		Block& firstBlock { op.getRegion().front() };
		for (unsigned i = 0; i < firstBlock.getNumArguments(); i++) {
			firstBlock.getArgument(i).printAsOperand(stream, OpPrintingFlags());
			stream << " ";
		}
		
		// print bit before start of region
		stream << "= do\n";
	};

	void print(llvm::raw_ostream &stream, func::CallOp op) {
		stream << op.getCallee().str() << " ";
		for (unsigned i = 0; i < op.getNumOperands(); i++) {
			op.getOperand(i).printAsOperand(stream, OpPrintingFlags());
			stream << " ";
		}
	};

	void print(llvm::raw_ostream &stream, func::ReturnOp op) {
		stream << "return ";

		unsigned nOperands = op.getNumOperands(); 
		if (nOperands == 0) {
			stream << "()";
		} else if (nOperands == 1) {
			op.getOperand(0).printAsOperand(stream, OpPrintingFlags());
		} else {
			stream << "(";
			for (unsigned i = 0; i < nOperands-1; i++) {
				op.getOperand(i).printAsOperand(stream, OpPrintingFlags());
				stream << ", ";
			}
			op.getOperand(nOperands-1).printAsOperand(stream, OpPrintingFlags());
			stream << ")";
		}
	};
  };
  HaskellOpPrinter hp;
};
} // namespace

namespace mlir {
void registerHaskellPrintingPass() {
  PassRegistration<HaskellPrintingPass>();
}
} // namespace mlir