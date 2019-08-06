release: *.h *.cpp GCNASM/*.asm
	clrxasm GCNASM/randomx_run.asm -o randomx_run.bin
	g++ *.cpp -lOpenCL -lpthread ../RandomX/build/librandomx.a -o opencl_test
