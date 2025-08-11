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

from huggingface_hub import snapshot_download
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_name = "Qwen3-0.6B-Base"
# model_name = "Qwen3-4B-Instruct-2507"

# 设置本地目录
local_dir = f"./{model_name}"

# 创建目录
os.makedirs(local_dir, exist_ok=True)

# 下载
snapshot_download(
    repo_id=f"Qwen/{model_name}",
    local_dir=local_dir,
    revision="main",  # 分支
    repo_type="model",
    max_workers=1,  # 只用 1 个线程
    # 🔽 可选：避免使用 aria2（防止自动启用多线程）
    # （如果你之前装了 aria2，它可能被自动启用）
    # 可以设置环境变量禁用：export HF_HUB_ENABLE_HF_TRANSFER=0
    # 🔽 可选：只下载必要的文件（先试跑通）
    # allow_patterns=["*.json", "tokenizer*", "config*"]  # 快速测试
)
