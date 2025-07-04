# 🚀 ERNIE 4.5: The Developer's Resource Guide 🤖

Welcome to the developer resource guide for ERNIE 4.5, a powerful family of open-source models from Baidu. This guide provides all the essential information, links, and code examples to help you get started with deploying ERNIE 4.5 models.

## 🔗  Quick Links

| Resource          | URL                                                              |
| ----------------- | ---------------------------------------------------------------- |
| **📝 Blog** | [https://yiyan.baidu.com/blog](https://yiyan.baidu.com/blog)       |
| **📄 Technical Report** | [https://yiyan.baidu.com/blog/publication](https://yiyan.baidu.com/blog/publication/) |
| **🤗 Hugging Face** | [https://huggingface.co/baidu](https://huggingface.co/baidu)       |
| **🔧 ERNIEKit** | [https://github.com/PaddlePaddle/ERNIE](https://github.com/PaddlePaddle/ERNIE) |
| **⚡ FastDeploy** | [https://www.modelscope.cn/studios/PaddlePaddle](https://github.com/PaddlePaddle/FastDeploy) |
| **💡 Baidu AI Studio** | [https://aistudio.baidu.com/](https://aistudio.baidu.com/)         |
| **🔅 ModelScope** | [https://www.modelscope.cn/studios/PaddlePaddle](https://www.modelscope.cn/studios/PaddlePaddle) |

## 📦 Open Source Models

ERNIE 4.5 is available under the **Apache 2.0 License**. The open-source release includes 10 models across 3 series, along with code for pre-training, fine-tuning, and inference deployment.

| Series        | Activated Parameters | Model Name Suffix | Description                                                                                             |
| ------------- | -------------------- | ----------------- | ------------------------------------------------------------------------------------------------------- |
| **0.3B Series** | \~300 Million         | `-0.3B`           | Lightweight models suitable for local and on-device deployment.                                         |
| **A3B Series** | \~3 Billion           | `-A3B`            | Efficient models offering a balance of performance and resource usage.                                  |
| **A47B Series** | \~47 Billion          | `-A47B`           | State-of-the-art models for maximum performance on complex tasks.                                       |

**🏷️  Naming Conventions:**

  * **-Base**: The foundational pre-trained model.
  * *(no suffix)*: The instruction-tuned chat model.
  * **-VL**: The Vision-Language multimodal model.
  * **Hybrid Thinking**: The VL model features a "thinking mode" (controlled by a parameter) that enhances reasoning, alongside a standard non-thinking mode for fast perception.

-----

## 👩‍💻  Getting Started: Running ERNIE 4.5 Locally

You can run the lightweight ERNIE 4.5 models on your local machine. Below are examples using `llama.cpp` for general CPU inference and MNN for optimized on-device deployment.

### 🍎 Example 1: Running with `llama.cpp` (for ERNIE-4.5-0.3B)

The `llama.cpp` project supports the ERNIE 4.5 0.3B models, allowing you to run them efficiently on a CPU.

**Step 1️⃣: Clone and Build `llama.cpp`**
First, get the latest version of `llama.cpp` which includes support for ERNIE 4.5.

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build the project
mkdir build  
cd build
cmake ..
make
```

**Step 2️⃣: Download the ERNIE 4.5 GGUF Model**
download the .gguf file.
```bash
# Install huggingface_hub
pip install -U huggingface_hub
huggingface-cli download --resume-download unsloth/ERNIE-4.5-0.3B-PT-GGUF --local-dir path/to/dir
```
```
# If timeout,use 
export HF_ENDPOINT=https://hf-mirror.com
```
**Step 3️⃣: Run Inference**
Use the `main` executable from `llama.cpp` to run the model.

```bash
# Run the model in interactive mode
cd llama.cpp/build/bin
./llama-cli -m /path/to/dir/ERNIE-4.5-0.3B-PT.gguf --jinja -p "Hello, who are you?" -n 128
```

  * `-m`: Specifies the path to your GGUF model file.
  * `-p`: Provides an initial prompt.
  * `-n`: Sets the number of tokens to generate.

### 🍏 Example 2: Running with MNN (for ERNIE-4.5-0.3B-PT-MNN)
Reference project: https://huggingface.co/taobao-mnn/ERNIE-4.5-0.3B-PT-MNN, welcome to visit the original author link

MNN is a highly efficient deep learning inference engine, perfect for edge and mobile devices. A 4-bit quantized version of ERNIE 4.5 is available specifically for MNN.

**Step 1️⃣: Download the MNN Model**
You can download the model from Hugging Face or ModelScope.

```bash
# Install Hugging Face Hub
pip install -U huggingface_hub
```
```
# Download the model files
# shell download
huggingface-cli download --resume-download taobao-mnn/ERNIE-4.5-0.3B-PT-MNN --local_dir path/to/dir
```
```
# If timeout,use 
export HF_ENDPOINT=https://hf-mirror.com
```
```
# SDK download
from huggingface_hub import snapshot_download
model_dir = snapshot_download('taobao-mnn/ERNIE-4.5-0.3B-PT-MNN')
```
```
# git clone
git clone https://www.modelscope.cn/MNN/ERNIE-4.5-0.3B-PT-MNN
```

**Step 2️⃣: Clone and Compile MNN**
You need to compile the MNN engine from the source with the correct flags to enable LLM support.

```bash
# Clone the MNN repository
git clone https://github.com/alibaba/MNN.git
cd MNN

# Create build directory and compile
mkdir build && cd build
cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true
make -j
```

**Step 3️⃣: Run the Demo**
Use the `llm_demo` application to run the model. 
```bash
# Run the MNN demo
./llm_demo /path/to/ERNIE-4.5-0.3B-PT-MNN/config.json prompt.txt
```
### 🍊 Example 3: Running with mlx (for ERNIE-4.5-0.3B-PT-bf16)
Reference project: https://huggingface.co/mlx-community/ERNIE-4.5-0.3B-PT-bf16, welcome to visit the original author link

MLX LM is a Python package for generating text and fine-tuning large language models on Apple silicon with MLX.

This model mlx-community/ERNIE-4.5-0.3B-PT-bf16 was converted to MLX format from baidu/ERNIE-4.5-0.3B-PT using mlx-lm version 0.25.2.

**Step 1️⃣: Download the mlx Model**
```bash
# Install Hugging Face Hub
pip install -U huggingface_hub
```
```
# Download the model files
# shell download
huggingface-cli download --resume-download mlx-community/ERNIE-4.5-0.3B-PT-bf16 --local_dir path/to/dir
```
```
# If timeout,use 
export HF_ENDPOINT=https://hf-mirror.com
```
**Step 2️⃣: Use with mlx**
```bash
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/ERNIE-4.5-0.3B-PT-bf16")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
-----
## 🌍 Developer Ecosystem and Tools

### 🛠️ Official Toolkits (PaddlePaddle Based)

  * **[ERNIEKit](https://github.com/PaddlePaddle/ERNIE)**: An industrial-grade toolkit for the full development lifecycle of ERNIE models. It supports high-performance pre-training, SFT, DPO, LoRA, and quantization (QAT/PTQ).
  * **[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)**: A production-ready inference and deployment toolkit. It features advanced acceleration (speculative decoding, MTP), comprehensive quantization support, and compatibility with numerous hardware backends (NVIDIA, Kunlunxin, Ascend, etc.).

## **🤝  Friends of OSS Projects (Third-Party Integrations)**

ERNIE 4.5 is being actively integrated into the wider open-source ecosystem. Here is the current status of support in popular projects:

| Project            | Status       |
| ------------------ | ------------ |
| **transformers** | Directly usable. Load models with `trust_remote_code=True`. Native integration is planned. ⚙️ ([Repo](https://github.com/huggingface/transformers))                                                                                    |
| **vLLM** | ✅  **Merged 🎉 \!** Native support for ERNIE 4.5 text models is now available in the main branch. You can use it directly with a standard vLLM installation for high-throughput inference. ([vLLM Repo](https://github.com/vllm-project/vllm))([Merged PR](https://github.com/vllm-project/vllm/pull/20220)) |
| **sglang** | Integration in progress.⏳ ([sglang Repo](https://github.com/sgl-project/sglang))([Tracking PR](https://www.google.com/search?q=https://github.com/sgl-project/sglang/pull/765))                                                                                         |
| **llama.cpp/ollama** | **🎉 0.3B models are already supported in `llama.cpp`\!** This allows for efficient local CPU inference. ([llama.cpp Repo](https://github.com/ggerganov/llama.cpp))([Merged PR](https://www.google.com/search?q=https://github.com/ggerganov/llama.cpp/pull/6926))        |
| **ms-swift** | ✅ **Merged\!** Support is now integrated, enabling streamlined fine-tuning and inference for ERNIE 4.5 within the ModelScope ecosystem. ([ms-swift Repo](https://github.com/modelscope/ms-swift))([Merged PR](https://www.google.com/search?q=https://github.com/modelscope/ms-swift/pull/475))    |