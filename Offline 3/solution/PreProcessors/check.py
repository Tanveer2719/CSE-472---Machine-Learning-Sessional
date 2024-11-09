import sys
if sys.base_prefix != sys.prefix:
    print("The code is running inside a virtual environment.")
else:
    print("The code is not running inside a virtual environment.")