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

import torch
import paddle
import os

# 设置路径
pytorch_bin_path = "./Qwen3-0.6B-Base/model.safetensors"  # 或 .safetensors
paddle_save_path = "./Qwen3-0.6B-Base-Paddle/model_state.pdparams"

# PyTorch → Paddle 映射：常见层名替换
pp_to_pt_keymap = {
    "embed_tokens": "embed_tokens",
    "embed_positions": "embed_positions",
    "self_attn.": "self_attn.",
    "k_proj.": "k_proj.",
    "q_proj.": "q_proj.",
    "v_proj.": "v_proj.",
    "o_proj.": "o_proj.",
    "mlp.": "mlp.",
    "norm": "norm",
    "post_attention_layernorm": "post_attention_layernorm",
    # 根据实际结构补充
}

# 加载 PyTorch 权重
print("Loading PyTorch model...")
if pytorch_bin_path.endswith(".safetensors"):
    from safetensors.torch import load_file

    pt_state_dict = load_file(pytorch_bin_path)
else:
    pt_state_dict = torch.load(pytorch_bin_path, map_location="cpu")

# 转换为 Paddle 格式
print("Converting to PaddlePaddle format...")
pd_state_dict = {}

for k, v in pt_state_dict.items():
    # print(f"Converting: {k} | dtype: {v.dtype} | shape: {v.shape}")

    if v.dtype == torch.bfloat16:
        np_array = v.float().numpy()
        pd_tensor = paddle.to_tensor(np_array)
        pd_tensor = paddle.cast(pd_tensor, "bfloat16")  # 直接创建 bfloat16 Tensor
    else:
        np_array = v.numpy()
        pd_tensor = paddle.to_tensor(np_array)

    if k.endswith(".weight") and "embed_tokens" not in k:
        print(f"  🔁 Transposing {k}: {np_array.shape} -> {np_array.T.shape}")
        pd_tensor = pd_tensor.T

    new_k = k
    # new_k = new_k.replace("model.", "qwen.")
    pd_state_dict[new_k] = pd_tensor

# 保存
os.makedirs(os.path.dirname(paddle_save_path) if os.path.dirname(paddle_save_path) else ".", exist_ok=True)
paddle.save(pd_state_dict, paddle_save_path)
print(f"✅ 已保存 Paddle 模型到: {paddle_save_path}")
