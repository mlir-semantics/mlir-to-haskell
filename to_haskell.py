# from build/ run
#   ./bin/mlir-opt <filename> --compute-dialects --to-haskell
# then get outputs between ---- START OF HASKELL ---- and ---- END OF HASKELL ----
# and apply regex replacement:
#   %(\w+) --> _$1
# then output to console
import subprocess
import argparse
import re

def get_args() -> argparse.Namespace:
    args = argparse.ArgumentParser(
        prog="to_haskell",
        description="converts MLIR programs into its Haskell embedding")
    args.add_argument("filename", help=".mlir file path")

    return args.parse_args()

def substr_between(original: str, head: str, tail: str) -> str:
    '''
    gets the substr of `original` between `head` and `tail`, non inclusive.
    '''
    return original.split(tail)[0].split(head)[1].strip()

def get_haskell_variable_names(original: str) -> str:
    return re.sub(r'%(\w+)', r'_\g<1>', original)

def get_haskell(raw_string: str) -> str:
    cut_string = substr_between(raw_string, "---- START OF HASKELL ----", "---- END OF HASKELL ----")
    haskell_file = get_haskell_variable_names(cut_string)
    return haskell_file

def main():
    args = get_args()

    run_result = subprocess.run(
        ["./bin/mlir-opt", args.filename, "--compute-dialects", "--to-haskell"],
        capture_output=True,
        cwd="build")
    raw_string = run_result.stdout.decode()
    
    # obtain haskell file
    haskell_file = get_haskell(raw_string)

    print(haskell_file)

if __name__ == '__main__':
    main()