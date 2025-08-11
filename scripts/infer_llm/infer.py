# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# infer.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置模型名称（确保你已登录 Hugging Face）
# model_name = "Qwen/Qwen3-4B-Instruct-2507"  # 或 Qwen3-4B-Instruct 等
# model_name = "./Qwen3-4B-Instruct-2507"
model_name = "./Qwen3-0.6B-Base"

# ================================
# 1. 加载 Tokenizer
# ================================
tokenizer = AutoTokenizer.from_pretrained(model_name)  # , use_fast=False, trust_remote_code=True)

print(f"✅ Tokenizer 类型: {tokenizer.__class__.__name__}")
print(f"_vocab size: {tokenizer.vocab_size}")

# ================================
# 2. 加载模型
# ================================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 推荐使用 bfloat16 或 float16
    device_map="auto",  # 自动分配 GPU（多卡也支持）
    trust_remote_code=True,  # 必须开启，Qwen 使用了自定义代码
)
# 确保模型在 GPU 上
device = next(model.parameters()).device
print(f"✅ 模型加载完成，运行设备: {device}")

# ================================
# 3. 打印模型结构（简化摘要）
# ================================
print("\n" + "=" * 60)
print("🧩 模型结构概览")
print("=" * 60)
print(model)
# 如果你想看更简洁的结构，可以取消注释下面这行（只显示前几层）
# print(list(model.named_children())[0])  # 查看第一层

# ================================
# 4. 打印参数统计
# ================================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 获取典型参数的 dtype
    first_param = next(model.parameters())
    dtype = first_param.dtype

    if dtype == torch.float32:
        bytes_per_param = 4
    elif dtype in [torch.float16, torch.bfloat16]:
        bytes_per_param = 2
    elif dtype == torch.int8:
        bytes_per_param = 1
    else:
        bytes_per_param = 2  # 默认

    total_size_mb = total * bytes_per_param / 1e6
    total_size_gb = total * bytes_per_param / 1e9

    return total, trainable, total_size_mb, total_size_gb


total_params, trainable_params, estimated_size_mb, estimated_size_gb = count_parameters(model)

print("\n" + "=" * 60)
print("📊 参数统计")
print("=" * 60)
print(f"总参数量:     {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"总参数量(M):   {total_params / 1e6:.2f}M")
print(f"粗略模型大小: ~{estimated_size_mb:.2f} MB (~{estimated_size_gb:.2f} GB)")

# ================================
# 5. 打印 GPU 显存占用（推理时）
# ================================
if device.type == "cuda":
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)

    print("\n" + "=" * 60)
    print("💾 GPU 显存占用")
    print("=" * 60)
    print(f"已分配显存 (allocated):  {allocated / 1024**3:.2f} GB")
    print(f"已保留显存 (reserved):    {reserved / 1024**3:.2f} GB")
else:
    print("\n💡 当前运行在 CPU 上，无 GPU 显存信息")

# ================================
# 6. 简单推理测试
# ================================
print("\n" + "=" * 60)
print("💬 推理测试")
print("=" * 60)

# 3. 构建对话输入
prompt = "你好，你是谁？"
messages = [{"role": "user", "content": prompt}]

# 使用 Qwen 的 chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("📝 Prompt:\n", prompt)

# 4. 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 5. 生成输出
outputs = model.generate(
    **inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1
)

# 6. 解码并输出（只显示新生成的部分）
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

print("🤖 回答:\n", response)
