# Third-party notice

`llm_converter_cpu_gpu.py` is derived from the Chakra execution-trace converter (`llm_converter.py`) of **LLMServingSim** (https://github.com/casys-kaist/LLMServingSim). It extends that converter with the `TOOLDAG` trace type, and is distributed under the license below.

```
MIT License

Copyright (c) 2024, CASYS Lab at KAIST

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The experiment itself reproduces "Agentic CPU-GPU Scheduling for Heterogeneous AI Workloads" by Tianxi Lu and Sherief Reda (arXiv:2607.22242). The paper is not included in this repository; see https://arxiv.org/abs/2607.22242.

Nothing third-party besides the converter is included. `scripts/setup_simulator.sh` downloads LLMServingSim (MIT) and its ASTRA-Sim and Chakra submodules from their own repositories. `scripts/setup_llm.sh` downloads vLLM (Apache-2.0) from PyPI and the model weights from Hugging Face (`casperhansen/deepseek-r1-distill-qwen-32b-awq`, an AWQ quantization of `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`), under the licenses stated in those repositories.
