cls

del MatMul.exe

nvcc.exe MatMul.cu --output-file MatMul.exe

MatMul.exe

del MatMul.lib MatMul.exp
