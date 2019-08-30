# RandomX OpenCL implementation

This repository contains full RandomX OpenCL implementation (portable code for all GPUs and optimized code AMD Vega GPUs). The latest version of RandomX (1.1.0 as of August 30th, 2019) is supported.

Note: it's only a benchmark/testing tool, not an actual miner. RandomX hashrate is expected to improve somewhat in the future thanks to further optimizations.

GPUs tested so far:

Model|CryptonightR H/S|RandomX H/S|Relative speed|Comment
-----|---------------|-----------|---------------|-------
AMD Vega 64 (1700/1100 MHz)|2200|1225|55.7%|JIT compiled mode, 285W
AMD Vega 64 (1100/800 MHz)|1023|845|82.6%|JIT compiled mode, 115W
AMD Vega 64 (1700/1100 MHz)|2200|163|7.4%|VM interpreted mode
AMD Vega FE (stock)|2150|980|45.6%|JIT compiled mode (intensity 4096)
AMD Radeon RX 560 4GB (1400/2200 MHz)|495|260|52.5%|JIT compiled mode (intensity 896)
AMD Radeon RX RX470/570 4GB|930-950|400-410|43%|JIT compiled mode, 50W
AMD Radeon RX RX480/580 4GB|960-1000|470|47%|JIT compiled mode, 60W
GeForce GTX 1080 Ti (2037/11800 MHz)|927|601|64.8%|VM interpreted mode

## Building on Windows

- Install Visual Studio 2017 Community and [CLRadeonExtender](https://github.com/CLRX/CLRX-mirror/releases)
- Add CLRadeonExtender's bin directory to PATH environment variable
- Open .sln file in Visual Studio and build it

## Building on Ubuntu

- Install prerequisites `sudo apt install git cmake build-essential`
- If you want to try JIT compiled code for Vega or Polaris GPUs, install amdgpu-pro drivers with OpenCL enabled (run the install script like this `./amdgpu-pro-install --opencl=pal`)
- Download [CLRadeonExtender](https://github.com/CLRX/CLRX-mirror/releases) and copy `clrxasm` to `/usr/local/bin`
- Then run commands:
```
git clone --recursive https://github.com/SChernykh/RandomX_OpenCL
cd RandomX_OpenCL/RandomX
mkdir build && cd build
cmake -DARCH=native ..
make
cd ../../RandomX_OpenCL
make
```

## Donations

If you'd like to support further development/optimization of RandomX miners (both CPU and AMD/NVIDIA), you're welcome to send any amount of XMR to the following address:

```
44MnN1f3Eto8DZYUWuE5XZNUtE3vcRzt2j6PzqWpPau34e6Cf4fAxt6X2MBmrm6F9YMEiMNjN6W4Shn4pLcfNAja621jwyg
```
