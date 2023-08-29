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
    // Print the operation itself and some of its properties
	printOpLine(printIndent(), op) << "\n";

    // Recurse into each of the regions attached to the operation.
    for (Region &region : op->getRegions())
      printRegion(region);
  }

  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
	auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }

  void printBlock(Block &block) {
    // Block main role is to hold a list of Operations: let's recurse.
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
      llvm::outs() << "\t";
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

	HaskellOpPrinter hp{stream};
	if (hp.isStandard(opName)) hp.printStandard(opName, op);
	else if (hp.isTerminator(opName)) hp.printTerminator(opName, op);
	else if (opName.equals("func.func")) hp.print(llvm::dyn_cast<func::FuncOp>(op));
	else if (opName.equals("func.call")) hp.print(llvm::dyn_cast<func::CallOp>(op));
	else stream << "UNIMPLEMENTED OP " << op->getName();

	return stream;
  }

  struct HaskellOpPrinter {
  public:
	HaskellOpPrinter(llvm::raw_ostream &stream) : pStream{&stream} {}; 

  private:
	llvm::raw_ostream *pStream;

	/* access stream by reference */
	llvm::raw_ostream &stream() { return *pStream; }; 

	/* operations to be printed in the "Standard" form */
	static llvm::StringMap<std::string> standardOps() {
		return {
			{"arith.addf", "add"},
			{"arith.addi", "add"},
			{"arith.mulf", "mult"},
			{"arith.negf", "negative"},
			{"arith.divf", "divide"},
			{"arith.uitofp", "intToFp"},
			{"index.sub", "sub"},
			{"memref.dim", "dim"},
			{"memref.dealloc", "deallocMemref"}
		};
	};

	/* terminators ops, they all are printed in similar ways */
	static llvm::StringMap<std::string> terminatorOps() {
		return {
			{"func.return", "return"},
			{"scf.yield", "scfYield"},
			{"affine.yield", "affineYield"}
		};
	};

	template <typename Container>
	void printValuesInterleave(const std::string &sep, const Container &c) {
		llvm::interleave(c, stream(), [&](mlir::Value arg) { arg.printAsOperand(stream(), OpPrintingFlags()); }, sep.c_str());
	}

	void printOperandsInterleave(const std::string sep, Operation *op) {
		printValuesInterleave(sep, op->getOperands());
	}

  public:
	/* Check against predefined list of ops printed in the "Standard" form. */
	bool isStandard(llvm::StringRef opName) { return standardOps().contains(opName); };

	/* predefined list of terminator ops, which have the same printing form */
	bool isTerminator(llvm::StringRef opName) { return terminatorOps().contains(opName); };

	/* 
	"Standard" case: prints into the form:
		{standardOps[opName]} <operand0> <operand1> ... <operandn>
	*/
	void printStandard(llvm::StringRef opName, Operation *op) {
		assert(isStandard(opName));
		stream() << standardOps().at(opName) << " ";
		printOperandsInterleave(" ", op);
	};

	/* 
	"Terminators" case: prints into the form:
		0 or >1 args:
		{terminatorOps[opName]} (<operand0>, <operand1>, ... <operandn>)
		1 arg:
		{terminatorOps[opName]} operand0
	*/
	void printTerminator(llvm::StringRef opName, Operation *op) {
		assert(isTerminator(opName));
		stream() << terminatorOps().at(opName) << " ";

		unsigned nOperands = op->getNumOperands(); 
		if (nOperands == 1) {
			op->getOperand(0).printAsOperand(stream(), OpPrintingFlags());
		} else {
			stream() << "(";
			printOperandsInterleave(", ", op);
			stream() << ")";
		}
	};

	void print(arith::AddFOp op) {
		stream() << "add "; 
		printOperandsInterleave(" ", op);
	};

	void print(func::FuncOp op) {
		// TODO: add function signature
		stream() << op.getSymName().str() << " ";

		// print arg names, which are arguments to the first block in the attached region
		Block& firstBlock { op.getRegion().front() };
		printValuesInterleave(" ", firstBlock.getArguments());
		
		// print bit before start of region
		stream() << "= do";
		// region will be printed in the parent call
	};

	void print(func::CallOp op) {
		stream() << op.getCallee().str() << " ";
		printOperandsInterleave(" ", op);
	};
  };
};
} // namespace

namespace mlir {
void registerHaskellPrintingPass() {
  PassRegistration<HaskellPrintingPass>();
}
} // namespace mlir