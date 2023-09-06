#include <optional>
#include <numeric>

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

const std::string DIALECT_ATTR = "__hask.dialects";

typedef llvm::raw_ostream stream_t; 

stream_t &printIndent(stream_t &stream, int indent) {
	for (int i = 0; i < indent; ++i)
		stream << "    ";
	return stream;
}

struct HaskellPrintingPass
    : public PassWrapper<HaskellPrintingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HaskellPrintingPass)

  StringRef getArgument() const final { return "to-haskell"; }
  StringRef getDescription() const final { return "Print into Haskell embedding."; }

  /// Methods below follow the nesting of the IR: operation->region->block->operation->...

  void printOperation(Operation *op) {
    // Print the operation itself and some of its properties
	std::optional<std::string> suffix { printOpLine(printIndent(stream(), indent), op) };
	stream() << "\n";
	
	// Recurse through all regions
	// But don't add indent if current op is builtin.module.
	if (llvm::isa<ModuleOp>(op)) {
		for (Region &region : op->getRegions()) 
			printRegion(region);
	} else {
		auto indent = pushIndent();
    	for (Region &region : op->getRegions()) 
			printRegion(region);
	}

	if (suffix) {
		printIndent(stream(), indent) << *suffix;
		stream() << "\n";
	}
  }

  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    for (Block &block : region.getBlocks())
      printBlock(block);
  }

  void printBlock(Block &block) {
    // Block main role is to hold a list of Operations: let's recurse.
    for (Operation &op : block.getOperations())
      printOperation(&op);
  }

  stream_t &stream() { return llvm::outs(); };

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };

  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  // Entry point for the pass.
  void runOnOperation() override {
	stream() << "---- START OF HASKELL ----\n";

    Operation *op = getOperation();
    resetIndent();
    printOperation(op);

	stream() << "---- END OF HASKELL ----\n";
  }

private:
  std::optional<std::string> printOpLine(stream_t &stream, Operation *op) {
	// print the operation, polymorphic on operation type
	llvm::StringRef opName {op->getName().getStringRef()};

	HaskellOpPrinter hp(stream, indent);
	if (hp.isStandard(opName)) hp.printStandard(opName, op);
	else if (hp.supportedCasting(opName)) hp.printCasting(op);
	else if (op->hasTrait<mlir::OpTrait::IsTerminator>()) hp.printTerminator(opName, op);
	else if (op->hasTrait<mlir::OpTrait::ConstantLike>()) hp.printConstantLike(op);
	else if (opName.equals("func.func")) hp.print(llvm::dyn_cast<func::FuncOp>(op)); // creates function SSA value
	else if (opName.equals("func.call")) hp.print(llvm::dyn_cast<func::CallOp>(op)); // uses function SSA value
	else if (opName.equals("scf.for")) hp.print(llvm::dyn_cast<scf::ForOp>(op)); // variable binding
	else if (opName.equals("memref.alloc")) hp.print(llvm::dyn_cast<memref::AllocOp>(op)); // dependent return type
	else if (opName.equals("memref.load")) hp.print(llvm::dyn_cast<memref::LoadOp>(op)); // variadic operands
	else if (opName.equals("memref.store")) hp.print(llvm::dyn_cast<memref::StoreOp>(op)); // variadic operands
	else stream << "UNIMPLEMENTED " << op->getName();

	return hp.getSuffix();
  }

  class HaskellOpPrinter {
  public:
	HaskellOpPrinter(stream_t &stream, int indent) : pStream{&stream}, suffix(), indent{indent} {}; 

  private:
	stream_t *pStream;

	/* access stream by reference */
	stream_t &stream() { return *pStream; }; 

	std::optional<std::string> suffix;
	void setSuffix(std::string newSuffix) { suffix = newSuffix; };

	int indent;

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

	/* casting ops have trivial embedding in Haskell, and will be translated as `return` */
	static llvm::StringSet<> castingOps() {
		return {
			"index.casts",
			"memref.cast"
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

	/* dialects printing */
	static std::optional<std::string> embedDialect(const StringAttr &dialect) {
		llvm::StringMap<std::optional<std::string>> specialDialects = {
			{"memref", "Memref s"},
			{"func", std::nullopt},
			{"index", std::nullopt} // nullopt for now as is only type. in future effects will be added to index embedding 
		};
		if (specialDialects.contains(dialect.str())) 
			return specialDialects.at(dialect.str());
		if (dialect.str().empty()) 
			return std::nullopt;
		
		// default printing method: capitalize first letter
		std::string dialectStr { dialect.str() };
		dialectStr.front() = std::toupper(dialectStr.front());
		return { dialectStr };
	};

	/* type printing */
	static std::string embedType(const mlir::Type t) {
		if (t.isa<mlir::Float32Type>()) return "Float";
		else if (t.isa<mlir::Float64Type>()) return "Double";
		else if (t.isa<mlir::IndexType>()) return "IxType";
		else if (auto mt = t.dyn_cast<mlir::MemRefType>()) return "MemrefType s " + embedType(mt.getElementType());
		else return "UNSUPPORTED_TYPE";
	}

	void printValue(mlir::Value arg) { arg.printAsOperand(stream(), OpPrintingFlags()); };

	/*
	Container should be an instance of:
		indexed_accessor_range_base<>
	*/
	template <typename Container>
	void printValuesInterleave(const std::string &sep, const Container &c) {
		llvm::interleave(c, stream(), [&](mlir::Value arg) { printValue(arg); }, sep.c_str());
	};

	template <typename Container>
	void printValues(const Container &c,
					 const std::string sep = ", ",
					 const std::string openB = "", const std::string closeB = "") {
		stream() << openB;
		printValuesInterleave(sep, c);
		stream() << closeB;
	}

	void printOperands(Operation *op, const std::string sep = " ") { printValues(op->getOperands(), sep); };

	/*
	Container should be an instance of:
		indexed_accessor_range_base<>

	if single value, print:
		v0 
	else print:
		(v0, v1, ..., vn)
	*/
	template <typename Container>
	void printValueTuple(const Container &c) {
		if (c.size() == 1) printValues(c, "");
		else printValues(c, ", ", "(", ")");
	};

	template <typename Container>
	void printIndices(const Container &c) { printValues(c, ", ", "[", "]"); };

	/*
	print operation out in the form:
		[(return_vals) <-] <op_name> memref_val[i0, i1, ..., in] 

	MemRefIndexingOp must have the following functions:
		- getMemref()
		- getIndices()
	*/
	template <typename MemRefIndexingOp>
	void printMemRefIndexingOp(MemRefIndexingOp op, const std::string opName) {
		printResultsWithAssign(op);
		stream() << opName << " ";
		printValue(op.getMemref());
		stream() << " ";
		printIndices(op.getIndices());
	}

	/* 
	results section, for operations that creates SSA values 
	returns boolean, if any SSA values were created. 
	*/
	bool printResults(Operation *op) {
		unsigned nValues = op->getNumResults();
		if (nValues == 0) return false;
		printValueTuple(op->getResults());
		return true;
	};

	void printResultsWithAssign(Operation *op) { if (printResults(op)) stream() << " <- "; };

	/*
	"Constant"-like case: prints the single value attribute as a return.

	Requirements on ConstantLikeOp:
	- Subclass of Operation*
	- Should have a `::getValueAttr()` function.
	*/
	template <typename ConstantLikeOp>
	void printConstantLike_(ConstantLikeOp op) {
		// TODO: maybe also add type printing here
		printResultsWithAssign(op); 
		stream() << "return ";
		op.getValueAttr().print(stream(), true);
	}

  public:
	/* if the operation printed needs a suffix printed too */
	std::optional<std::string> getSuffix() { return suffix; };

	/* Check against predefined list of ops embedded in the "Standard" form. */
	bool isStandard(llvm::StringRef opName) { return standardOps().contains(opName); };

	/* Checks against predefined list of supported casting operations */
	bool supportedCasting(llvm::StringRef opName) { return castingOps().contains(opName); };

	/* supported terminator operations */
	bool supportedTerminator(llvm::StringRef opName) { return terminatorOps().contains(opName); };

	/* 
	"Standard" case: prints into the form:
		{standardOps[opName]} <operand0> <operand1> ... <operandn>
	*/
	void printStandard(llvm::StringRef opName, Operation *op) {
		assert(isStandard(opName) && "printStandard called on operation not within isStandard() list.");
		printResultsWithAssign(op);
		stream() << standardOps().at(opName) << " ";
		printOperands(op);
	};

	/* 
	"Terminators" case: prints into the form:
		0 or >1 args:
		{terminatorOps[opName]} (<operand0>, <operand1>, ... <operandn>)
		1 arg:
		{terminatorOps[opName]} operand0
	*/
	void printTerminator(llvm::StringRef opName, Operation *op) {
		assert(supportedTerminator(opName) && "printTerminator() called on unsupported operation");
		stream() << terminatorOps().at(opName) << " ";
		printValueTuple(op->getOperands());
	};

	void printConstantLike(Operation *op) {
		assert(op->hasTrait<mlir::OpTrait::ConstantLike>() && "expected operation with CosntantLike trait");
		llvm::StringRef opName {op->getName().getStringRef()};
		if (opName.equals("index.constant")) printConstantLike_(llvm::dyn_cast<index::ConstantOp>(op));
		else if (opName.equals("arith.constant")) printConstantLike_(llvm::dyn_cast<arith::ConstantOp>(op));
		else stream() << "UNSUPPORTED CONSTANT OPERATION: " << opName;
	};

	void printCasting(Operation *op) {
		const llvm::StringRef opName = op->getName().getStringRef();
		assert(supportedCasting(opName) && "expected a supported casting operation");
		// check for self, ensure Op has supported form to be casted (single input & output)
		assert(op->hasTrait<mlir::CastOpInterface::Trait>() && 
			   op->hasTrait<mlir::OpTrait::OneResult>() && 
			   op->hasTrait<mlir::OpTrait::OneOperand>() &&
			   ("operation '" + opName + "' does not look like a cast. double check?").str().c_str());

		printResultsWithAssign(op);
		stream() << "return ";
		printOperands(op); 
	};

	/* Special cases */
	
	void print(memref::AllocOp op) {
		mlir::MemRefType memRefType = op.getType();

		printResults(op); // SSA value
		stream() << " :: " << embedType(memRefType) << " <- allocND [";
		llvm::interleave( // print memref sizes
			memRefType.getShape(), 
			stream(), 
			[&](int64_t arg) { stream() << arg; }, 
			", "
		); 
		stream() << "]";
	};

	void print(memref::LoadOp op) {	printMemRefIndexingOp(op, "load"); };

	void print(memref::StoreOp op) { printMemRefIndexingOp(op, "store"); };

	void print(scf::ForOp op) {
		printResultsWithAssign(op);
		stream() << "for ";
		printValue(op.getLowerBound());
		stream() << " ";
		printValue(op.getUpperBound());
		stream() << " ";
		printValue(op.getStep());
		stream() << " ";
		printValues(op.getInitArgs(), " ", "(", ")");
		stream() << " (\\";
		printValue(op.getInductionVar());
		stream() << " ";
		printValueTuple(op.getRegionIterArgs());
		stream() << " -> do";

		setSuffix(")");
	}

	void print(func::FuncOp op) {		
		// signature, typeclass
		stream() << op.getSymName().str() << " :: ";
		assert(op->hasAttrOfType<ArrayAttr>(DIALECT_ATTR));
		auto dialects = op->getAttrOfType<ArrayAttr>(DIALECT_ATTR).getAsRange<StringAttr>();
		std::vector<std::string> dialectEmbedding;
		for (const auto& dialect : dialects) {
			std::optional<std::string> embedded { embedDialect(dialect) };
			if (embedded) dialectEmbedding.push_back(*embedded);
		}
		if (!dialectEmbedding.empty()) {
			stream() 
				<< "(Members '["
				<< std::accumulate(
					std::next(dialectEmbedding.begin()), 
					dialectEmbedding.end(), 
					dialectEmbedding[0], 
					[](std::string a, std::string b) {
						return a + ", " + b;
				   }) 
				<< "] r) => ";
		}

		// TODO next: make computeDialects assign dialects to builtin.module so we know what to import
		// TODO next: make computeDialects more generic for functions

		// signature, inputs
		FunctionType fnType { op.getFunctionType() };
		for (const auto& type : fnType.getInputs())
			stream() << embedType(type) << " -> ";

		// signature, outputs (as tuple):
		llvm::ArrayRef<Type> outputs { fnType.getResults() };
		stream() << "Sem r (";
		if (!outputs.empty()) {
			for (auto it = outputs.begin(); it < outputs.end() - 1; ++it)
				stream() << embedType(*it) << ", ";
			stream() << embedType(*(outputs.end() - 1));
		}
		stream() << ")\n";

		// function definition
		printIndent(stream(), indent) << op.getSymName().str() << " ";

		// print arg names, which are arguments to the first block in the attached region
		Block& firstBlock { op.getRegion().front() };
		printValuesInterleave(" ", firstBlock.getArguments());
		
		// print bit before start of region
		stream() << " = do";
		// region will be printed in the parent call
	};

	void print(func::CallOp op) {
		printResults(op);
		stream() << op.getCallee().str() << " ";
		printOperands(op);
	};
  };
};
} // namespace

namespace mlir {
void registerHaskellPrintingPass() {
  PassRegistration<HaskellPrintingPass>();
}
} // namespace mlir