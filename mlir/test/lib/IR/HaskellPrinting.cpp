#include <optional>
#include <numeric>
#include <set>

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

static const std::string DIALECT_ATTR = "__hask.dialects";
static const std::string TAB = "    ";
/* suffix for MLIR functions (to distinguish from Haskell namespace) */
static inline std::string FUNCTIONS_SUFFIX = "_mlir";  

typedef llvm::raw_ostream stream_t; 

static stream_t &printIndent(stream_t &stream, int indent) {
	for (int i = 0; i < indent; ++i)
		stream << TAB;
	return stream;
}

static std::string capitalise(const std::string &word) {
	std::string newWord { word };
	newWord.front() = std::toupper(newWord.front());
	return { newWord };
}

/* dialects printing */
static std::optional<std::string> embedDialect(const std::string &dialect) {
	llvm::StringMap<std::optional<std::string>> specialDialects = {
		{"memref", "Memref s"},
		{"func", std::nullopt}
	};
	if (specialDialects.contains(dialect)) 
		return specialDialects.at(dialect);
	if (dialect.empty()) 
		return std::nullopt;
	
	// default printing method: capitalise first letter
	return { capitalise(dialect) };
}

/* type printing */
static std::string embedType(const mlir::Type t) {
	if (t.isa<mlir::Float32Type>()) return "Float";
	else if (t.isa<mlir::Float64Type>()) return "Double";
	else if (t.isa<mlir::IndexType>()) return "IxType";
	else if (auto mt = t.dyn_cast<mlir::MemRefType>()) return "MemrefType s " + embedType(mt.getElementType());
	else if (t.isa<mlir::IntegerType>()) {
		// TODO: use dynamic width / signedness integers embedded types instead. 
		// Right now just embeds all widths into Int (32 bits), apart from width 1, which goes into Boolean.
		auto tInt = t.dyn_cast<mlir::IntegerType>();
		if (tInt.getWidth() == 1) return "Bool";
		else return "Int"; 
	}
	else return "UNSUPPORTED_TYPE";
}

/*
Extract the region's dialects, stored within the attributes, as a vector
*/
static std::vector<std::string> getRegionDialects(Operation *op) {
	std::vector<std::string> dialects;
	for (const auto& dialect : op->getAttrOfType<ArrayAttr>(DIALECT_ATTR).getAsRange<StringAttr>())
		dialects.push_back(dialect.str());
	return dialects;
}

struct Interpreter {
	std::string fn;
	int priority; // higher priority => top of the interpreter stack, applied last
	Interpreter(std::string fn, int priority = 0) : fn{fn}, priority{priority} {}
};

bool operator==(const Interpreter& lhs, const Interpreter& rhs) {
	return lhs.priority == rhs.priority && lhs.fn == rhs.fn;
}

struct InterpreterLT {
bool operator()(const Interpreter& lhs, const Interpreter& rhs) {
	return (lhs.priority <= rhs.priority) || 
			((lhs.priority == rhs.priority) && (lhs.fn < rhs.fn));
}
};

struct InterpreterGT {
bool operator()(const Interpreter& lhs, const Interpreter& rhs) {
	return !InterpreterLT()(lhs, rhs) && !(lhs == rhs);
}
};

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
		printRegions(op, suffix);
	} else {
		auto indent = pushIndent();
    	printRegions(op, suffix);
	}
  }

  void printRegions(Operation *op, const std::optional<std::string> &suffix) {
	for (Region &region : op->getRegions()) 
		printRegion(region);

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

	// op is the module-level op with 
    Operation* op = getOperation();
    resetIndent();
    printOperation(op);

	if (ModuleOp mOp = llvm::dyn_cast<ModuleOp>(op))
		printMainFunction(mOp);

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
	else if (opName.equals("builtin.module")) hp.print(llvm::dyn_cast<ModuleOp>(op)); // not an effect in Haskell 
	else if (opName.equals("func.func")) hp.print(llvm::dyn_cast<func::FuncOp>(op)); // creates function SSA value, but not SSA in haskell
	else if (opName.equals("func.call")) hp.print(llvm::dyn_cast<func::CallOp>(op)); // uses function SSA value
	else if (opName.equals("scf.for")) hp.print(llvm::dyn_cast<scf::ForOp>(op)); // variable binding
	else if (opName.equals("memref.alloc")) hp.print(llvm::dyn_cast<memref::AllocOp>(op)); // dependent return type
	else if (opName.equals("memref.load")) hp.print(llvm::dyn_cast<memref::LoadOp>(op)); // variadic operands
	else if (opName.equals("memref.store")) hp.print(llvm::dyn_cast<memref::StoreOp>(op)); // variadic operands
	else if (opName.equals("arith.cmpi")) hp.print(llvm::dyn_cast<arith::CmpIOp>(op)); // semantics differs according to an enum attribute
	else stream << "UNIMPLEMENTED " << op->getName();

	return hp.getSuffix();
  }

  void printMainFunction(ModuleOp mOp, const std::string entry = "main") {
	// look for the entry point function, return if none found
	Operation* entryFn = mOp.lookupSymbol(entry);
	if (entryFn == nullptr) return; 

	// add interpreters for each of the dialects used
	std::vector<Interpreter> interpreters;
	for (const auto& dialect : getRegionDialects(entryFn)) {
		std::optional<std::string> interpreterFn = getDialectInterpreterFn(dialect);
		if (!interpreterFn) continue;

		interpreters.push_back(Interpreter(*interpreterFn));
		// add necessary lower interpreter too 
		std::vector<Interpreter> relatedInterpreters { HaskellOpPrinter::getRelatedInterpreters(dialect) };
		interpreters.insert(interpreters.begin(), relatedInterpreters.begin(), relatedInterpreters.end());
	}

	// get interpreter stack in printable order
	std::sort(interpreters.begin(), interpreters.end(), InterpreterGT());
	interpreters.erase(
		std::unique(interpreters.begin(), interpreters.end()), 
		interpreters.end());

	// print out function for execution
	stream() << "main :: IO ()\n";
	stream() << "main = do \n";
	auto x = pushIndent();
	printIndent(stream(), indent) << "(outs, res) <- return";
	for (const auto& interp : interpreters)
		stream() << " $ " << interp.fn;
	stream() << " " << entry << FUNCTIONS_SUFFIX << "\n";

	printIndent(stream(), indent) << "putStrLn outs\n";
  }

  std::optional<std::string> getDialectInterpreterFn(const std::string& dialect) {
	if (dialect == "func") return {};
	return "run" + capitalise(dialect);
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
			{"memref.dealloc", "deallocMemref"},
			{"vector.print", "print"}
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
			{"scf.yield", "scfYieldOp"},
			{"affine.yield", "affineYieldOp"}
		};
	};

	void printValue(mlir::Value arg) { arg.printAsOperand(stream(), OpPrintingFlags()); };

	/* Some dialects interpret into lower level dialects and these need to be imported also */
	static std::map<std::string, std::set<std::string>> relatedImports() {
		return {
			{"Scf", {"ControlFlow"}},
			{"Memref", {"Mutable", "Control.Monad.ST"}},
			{"Vector", {"Polysemy.Writer"}}
		};
	}

	static std::map<std::string, std::vector<Interpreter>> relatedInterpreters() {
		return {
			{"scf", {Interpreter("runControlFlow", 1)}},
			{"memref", {Interpreter("runMutable", 1), Interpreter("runM", 2), Interpreter("runST", 3)}},
			{"vector", {Interpreter("runWriterAssocR", 1)}}
		};
	}

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
	where valueToStore is non-empty for operations storing values to memory

	MemRefIndexingOp must have the following functions:
		- getMemref()
		- getIndices()
	*/
	template <typename MemRefIndexingOp>
	void printMemRefIndexingOp(MemRefIndexingOp op, const std::string opName, std::optional<Value> valueToStore = std::nullopt) {
		printResultsWithAssign(op);
		stream() << "Memref." << opName << " ";
		if (valueToStore) { 
			printValue(*valueToStore);
			stream() << " ";
		}
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
	static bool isStandard(llvm::StringRef opName) { return standardOps().contains(opName); };

	/* Checks against predefined list of supported casting operations */
	static bool supportedCasting(llvm::StringRef opName) { return castingOps().contains(opName); };

	/* supported terminator operations */
	static bool supportedTerminator(llvm::StringRef opName) { return terminatorOps().contains(opName); };

	/* related imports for interpretation, empty set if none needed */
	static std::set<std::string> getRelatedImports(const std::string& dialect) {
		auto rImports { relatedImports() }; 
		if (rImports.find(dialect) != rImports.end()) return rImports.at(dialect);
		return {};
	};

	/* related interpreters */
	static std::vector<Interpreter> getRelatedInterpreters(const std::string& dialect) {
		auto rInterpreters { relatedInterpreters() }; 
		if (rInterpreters.find(dialect) != rInterpreters.end()) return rInterpreters.at(dialect);
		return {};
	}

	/* 
	"Standard" case: prints into the form:
		{standardOps[opName]} <operand0> <operand1> ... <operandn>
	*/
	void printStandard(llvm::StringRef opName, Operation *op) {
		assert(isStandard(opName) && "printStandard called on operation not within isStandard() list.");
		printResultsWithAssign(op);
		stream() << capitalise(op->getDialect()->getNamespace().str()) << "." << standardOps().at(opName) << " ";
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

	void print(ModuleOp op) {
		assert(op->hasAttrOfType<ArrayAttr>(DIALECT_ATTR));
		
		stream() << "{-# LANGUAGE TemplateHaskell, LambdaCase, BlockArguments, GADTs, \n"
				 << "FlexibleContexts, TypeOperators, DataKinds, PolyKinds, ScopedTypeVariables #-}\n\n"
				 << "module Main where\n" 
				 << "import Data.Function\n"
				 << "import Polysemy\n";

		// dialects need to be imported, as well as related libraries (for interpreting dialects)
		auto dialects = op->getAttrOfType<ArrayAttr>(DIALECT_ATTR).getAsRange<StringAttr>();
		std::set<std::string> imports;
		for (const auto& dialect : dialects) {
			std::string dialectModule { capitalise(dialect.str()) };
			std::set<std::string> relatedModules { getRelatedImports(dialectModule) };
			imports.insert(dialectModule);
			imports.merge(getRelatedImports(dialectModule));
		}

		for (const auto& mod : imports) {
			stream() << "import " << mod << "\n";
		}
	}

	void print(arith::CmpIOp op) {
		// stringify predicates
		// 	stringifyCmpIPredicate(op.getPredicate())
		// print LHS and RHS
		// 	op.getLhs(); op.getRhs()
	}
	
	void print(memref::AllocOp op) {
		mlir::MemRefType memRefType = op.getType();

		printResults(op); // SSA value
		stream() << " :: " << embedType(memRefType) << " <- Memref.allocND [";
		llvm::interleave( // print memref sizes
			memRefType.getShape(), 
			stream(), 
			[&](int64_t arg) { stream() << arg; }, 
			", "
		); 
		stream() << "]";
	};

	void print(memref::LoadOp op) {	printMemRefIndexingOp(op, "load"); };

	void print(memref::StoreOp op) { printMemRefIndexingOp(op, "store", op.getValueToStore()); };

	void print(scf::ForOp op) {
		printResultsWithAssign(op);
		stream() << "Scf.for ";
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
		assert(op->hasAttrOfType<ArrayAttr>(DIALECT_ATTR));
		stream() << op.getSymName().str() << FUNCTIONS_SUFFIX << " :: ";
		std::vector<std::string> dialectEmbedding;
		for (const auto& dialect : getRegionDialects(op)) {
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
		printIndent(stream(), indent) << op.getSymName().str() << FUNCTIONS_SUFFIX << " ";

		// print arg names, which are arguments to the first block in the attached region
		Block& firstBlock { op.getRegion().front() };
		printValuesInterleave(" ", firstBlock.getArguments());
		
		// print bit before start of region
		stream() << " = do";
		// region will be printed in the parent call
	};

	void print(func::CallOp op) {
		printResultsWithAssign(op);
		stream() << op.getCallee().str() << FUNCTIONS_SUFFIX << " ";
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