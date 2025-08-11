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

# infer_paddle.py
from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM
import paddle

# 模型路径（必须是 Paddle 格式：model.pdparams + config.json）
model_dir = "./Qwen3-0.6B-Base-Paddle"  # 包含 model.pdparams

# ================================
# 1. 加载 Tokenizer 和 Model
# ================================
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="bfloat16")
model.eval()  # 推理模式
print(f"✅ Model loaded: {model.__class__.__name__}")

# ================================
# 3. 构造输入（支持普通文本 or Chat 模板）
# ================================
prompt = "你好，你是谁？"
messages = [{"role": "user", "content": prompt}]

# 尝试使用 chat template（如果支持）
try:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("📝 使用 Chat Template:\n", text)
except Exception as e:
    print("⚠️ 无法使用 apply_chat_template，回退到原始文本")
    text = prompt
    print("📝 Input:", text)

# ================================
# 4. Tokenize → 返回 Paddle Tensor
# ================================
input_features = tokenizer(
    text, return_tensors="pd", return_attention_mask=True  # ✅ 自动返回 paddle.Tensor  # 确保生成 attention_mask
)

# ================================
# 5. 生成输出
# ================================
with paddle.no_grad():
    outputs = model.generate(
        **input_features,
        max_new_tokens=256,
        decode_strategy="sampling",  # 相当于 do_sample=True
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )

# ================================
# 6. 解码输出
# ================================
generated_ids = outputs[0]

# 解码（跳过输入部分）
input_length = input_features["input_ids"].shape[1]
response_ids = generated_ids[0][input_length:]  # 取第一个样本的新生成部分

response = tokenizer.decode(response_ids, skip_special_tokens=True)

print("\n" + "=" * 50)
print("💬 用户: ", prompt)
print("🤖 模型: ", response)
print("=" * 50)
