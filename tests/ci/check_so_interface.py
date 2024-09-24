import ast

# functions and their arguments to check
# specified here: https://www.overleaf.com/project/5e837cac9659910001e5f71e
# could move this to a file
interface = {
    "python/pysmurf/client/tune/smurf_tune.py": {  # file
        "SmurfTuneMixin": {                        # class
            "find_freq": [],                       # function and args
        }
    },
}

def _compare_args(ast_args, spec_args):
    # check the arguments in the code match the specification
    pass

if __name__ == "__main__":
    for fname, spec in interface.items():
        with open(fname, 'r') as fh:
            tree = ast.parse(fh.read())

        for key in spec:
            # check this function/class is in the code
            assert key in tree

            # compare its arguments
            if isinstance(tree[key], ast.FunctionDef):
                assert _compare_args(spec[key], tree[key])
            elif isinstance(tree[key], ast.ClassDef):
                # could make this recursive maybe
                for meth in spec[key]:
                    assert meth in tree[key]
                    assert _compare_args(spec[key][meth], tree[key][meth])
