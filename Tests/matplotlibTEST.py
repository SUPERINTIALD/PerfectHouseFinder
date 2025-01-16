# import matplotlib
# print(matplotlib.get_cachedir())

# Set MPLCONFIGDIR environment variable
import os
os.environ['MPLCONFIGDIR'] = "../static/cache/.matplotlib"

# Re-import matplotlib after setting MPLCONFIGDIR
import matplotlib
print(matplotlib.get_cachedir())
