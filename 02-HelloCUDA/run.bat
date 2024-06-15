cls

del HelloCUDA.exe

nvcc.exe HelloCUDA.cu --output-file HelloCUDA.exe

HelloCUDA.exe

del HelloCUDA.lib HelloCUDA.exp
