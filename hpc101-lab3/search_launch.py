import tilelang
import tilelang.language as T
import inspect
import os

# Find T.Kernel definition
print('T.Kernel type:', type(T.Kernel))
try:
    print('T.Kernel module:', T.Kernel.__module__)
except:
    pass
try:
    print('T.Kernel file:', inspect.getfile(T.Kernel))
except:
    pass

# Search for launch_bounds in tilelang source
import glob
base = os.path.dirname(tilelang.__file__)
print('tilelang base:', base)

# Search all .py files
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    content = fh.read()
                if 'launch_bound' in content or 'min_block' in content or 'max_threads' in content:
                    for i, line in enumerate(content.split(chr(10)), 1):
                        if 'launch_bound' in line or 'min_block' in line or 'max_threads' in line:
                            print(path + ':' + str(i) + ': ' + line.strip())
            except:
                pass

# Also check T.Kernel signature
try:
    sig = inspect.signature(T.Kernel)
    print('T.Kernel signature:', sig)
except:
    print('Cannot get T.Kernel signature')

# Check for launch_bounds in compiled extensions
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.so'):
            path = os.path.join(root, f)
            os.system('strings ' + path + ' | grep -i launch_bound | head -5')

print('DONE')
