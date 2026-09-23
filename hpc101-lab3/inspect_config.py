import tilelang
import tilelang.language as T
import inspect

print('GemmWarpPolicy values:')
if hasattr(T, 'GemmWarpPolicy'):
    p = T.GemmWarpPolicy
    for attr in dir(p):
        if not attr.startswith('_'):
            try:
                val = getattr(p, attr)
                print('  ' + attr + ' = ' + str(val))
            except:
                pass

print('')
print('PassConfigKey values:')
if hasattr(tilelang, 'PassConfigKey'):
    p = tilelang.PassConfigKey
    for attr in dir(p):
        if not attr.startswith('_'):
            try:
                val = getattr(p, attr)
                print('  ' + attr + ' = ' + str(val))
            except:
                pass

print('')
print('Searching for PTXAS/register configs:')
try:
    src = inspect.getsource(tilelang.PassConfigKey)
    for line in src.split(chr(10)):
        low = line.lower()
        if 'ptxas' in low or 'register' in low:
            print('  ' + line.strip())
except Exception as e:
    print('  Error: ' + str(e))

try:
    from tilelang import config as tlc
    src = inspect.getsource(tlc)
    for line in src.split(chr(10)):
        low = line.lower()
        if 'ptxas' in low or 'register' in low:
            print('  config: ' + line.strip())
except Exception as e:
    print('  config error: ' + str(e))

print('DONE')
