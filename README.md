# RandomX OpenCL implementation

This repository contains full RandomX OpenCL implementation (portable code for all GPUs and optimized code AMD Vega GPUs). The latest version of RandomX (1.0.4 as of June 23rd, 2019) is supported.

Note: it's only a benchmark/testing tool, not an actual miner. RandomX hashrate is expected to improve somewhat in the future thanks to further optimizations.

GPUs tested so far:

Model|CryptonightR H/S|RandomX H/S|Relative speed|Comment
-----|---------------|-----------|---------------|-------
AMD Vega 64 (1700/1100 MHz)|2200|1188|54%|JIT compiled mode
AMD Vega FE (stock)|2150|980|45.6%|JIT compiled mode (intensity 4096)
GeForce GTX 1080 Ti (2037/11800 MHz)|927|662|71.4%|VM interpreted mode

## Building on Windows

- Install Visual Studio 2017 Community and [CLRadeonExtender](https://github.com/CLRX/CLRX-mirror/releases)
- Add CLRadeonExtender's bin directory to PATH environment variable
- Open .sln file in Visual Studio and build it

## Donations

If you'd like to support further development/optimization of RandomX miners (both CPU and AMD/NVIDIA), you're welcome to send any amount of XMR to the following address:

```
44MnN1f3Eto8DZYUWuE5XZNUtE3vcRzt2j6PzqWpPau34e6Cf4fAxt6X2MBmrm6F9YMEiMNjN6W4Shn4pLcfNAja621jwyg
```
