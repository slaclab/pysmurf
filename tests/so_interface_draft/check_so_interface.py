import ast

# functions and their arguments to check
# specified here: https://www.overleaf.com/project/5e837cac9659910001e5f71e
# could move this to a file

#to complete: work on functions and arguments to check file and spec_args dictionary
interface = {
    "python/pysmurf/client/tune/smurf_tune.py": {  # file
        "SmurfTuneMixin": {                        # class
            "find_freq": [],                       # function and args
        }
    },
}

def _compare_args(ast_args, spec_args) -> bool: #are ast_args and spec_args both dictionaries? are these in the right order? I treated ast_args like a dictionary for this


    # check the arguments in the code match the specification

    if len(spec_args) != len(ast_args):
        return False

    for expected, actual in zip(spec_args, ast_args): #expected, actual == spec, ast (a little more understandable)
        # Check argument name
        if expected['name'] != actual.arg:
            return False
        
        # Check argument type annotation if specified
        if 'type' in expected: #if there is a type annotation
            if actual.annotation is None: #check if annotation of actual -> false
                return False
            if not isinstance(actual.annotation, ast_args.Name) or actual.annotation.id != expected['type']: #if the annotation of actual doesn't match expe
                return False
        
        # Check default value if specified
        if 'default' in expected:
            if actual.default is None:
                return False
            if not isinstance(actual.default, ast.Constant) or actual.default.value != expected['default']:
                return False

    # Check for *args


    # Check for **kwargs


    return True

if __name__ == "__main__":   #if this is the main thing being run
    for fname, spec in interface.items():   #loop over file names "fname" as keys and "spec" as values in dictionary called "interface"
        with open(fname, 'r') as fh:   #opens file "fname", in read mode 'r', to be used in code as "fh"
            tree = ast.parse(fh.read()) #parsing contents of file into abstract syntax tree

        for key in spec:
            # check this function/class is in the code
            assert key in tree

            s # compare its arguments
            if isinstance(tree[key], ast.FunctionDef):
                assert _compare_args(spec[key], tree[key])
            elif isinstance(tree[key], ast.ClassDef):
                # could make this recursive maybe
                for meth in spec[key]: #iterating over methods specified in spec[key]
                    assert meth in tree[key]
                    assert _compare_args(spec[key][meth], tree[key][meth])

pass
