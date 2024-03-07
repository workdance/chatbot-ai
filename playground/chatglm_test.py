import os
import platform
from transformers import AutoTokenizer, AutoModel
MODEL_PATH = "/Users/michael.sl/Code/ai-llm/THUDM/chatglm3-6b"
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
# 在深度学习和 PyTorch 中，表达式 half().to('mps') 是用来将一个 PyTorch 张量（Tensor）转换为半精度浮点（FP16）格式，
# 并将它转移到苹果的 Metal Performance Shaders（MPS）后端。
#
# 解释一下各部分的含义：
#
# half()：这是一个 PyTorch 张量的方法，用于将张量的数据类型转换为半精度浮点（float16 或 FP16）。
# 这通常用于减少模型的内存占用量，有助于在内存受限的设备上运行更大的模型，或者加快模型的计算速度。
#
# to('mps')：这个方法将张量移到指定的设备上。在这种情况下，'mps' 指的是 Metal Performance Shaders，
# 这是苹果公司为其设备提供的一种图形和计算加速技术，类似于 CUDA 在 NVIDIA GPU 上做的事情。
# 使用 MPS，可以在苹果的硬件上（如搭载 M1 芯片的 Mac）加速深度学习模型的训练和推断。


model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").half().to('mps')
model = model.eval()
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=[])
print(response)
