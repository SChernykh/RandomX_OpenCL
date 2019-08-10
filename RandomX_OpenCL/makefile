release: *.h *.cpp GCNASM/*.asm
	clrxasm GCNASM/randomx_run_gfx803.asm -o randomx_run_gfx803.bin
	clrxasm GCNASM/randomx_run_gfx900.asm -o randomx_run_gfx900.bin
	g++ *.cpp -O3 -lOpenCL -lpthread ../RandomX/build/librandomx.a -o opencl_test
