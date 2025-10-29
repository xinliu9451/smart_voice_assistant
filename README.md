# 傻妞语音智能助手

一个基于本地模型的中文语音智能助手，支持关键词唤醒、语音识别、大模型对话、本地知识库问答和语音合成。

## 🎥 演示视频

### 基础功能演示

点击观看：[语音助手演示视频](https://github.com/xinliu9451/smart_voice_assistant/blob/main/assets/%E8%AF%AD%E9%9F%B3%E5%8A%A9%E6%89%8B%E6%BC%94%E7%A4%BA.mp4)

### RAG 功能演示

点击观看：[RAG 功能演示视频](https://github.com/xinliu9451/smart_voice_assistant/blob/main/assets/%E8%AF%AD%E9%9F%B3%E5%8A%A9%E6%89%8B%E6%BC%94%E7%A4%BA_RAG.mp4)

## 📢 最新更新


**新增功能**
- ✨ **RAG 知识库问答**: 新增本地检索增强生成(RAG)功能，支持基于知识库的智能问答。首次运行会自动构建向量库，让傻妞能够基于本地知识进行更精准的回答。

**重大Bug修复**
- 🐛 **VAD断句逻辑修复**: 修复了语音识别中的严重bug，该bug导致即使持续说话也会在3秒后强制断句。问题根源是错误地在检测到语音开始时就初始化了 `last_speech_time` 变量，导致超时判断逻辑失效。现已移除了多余的时间跟踪变量，确保只有在检测到真正的静音间隔后才会断句。现在可以流畅地进行长句对话，不会被错误截断。

## ✨ 主要功能

### 1. 关键词唤醒 (KWS)
- 支持"你好傻妞"或"傻妞"唤醒
- 基于 sherpa-onnx 关键词检测模型
- 低延迟、高准确率

### 2. 语音识别 (ASR)
- 基于 SenseVoice 模型的离线语音识别
- 支持 VAD（语音活动检测）
- 自动检测说话结束（3秒静音）
- 支持逆文本标准化（ITN），自动添加标点符号

### 3. 大模型对话 (LLM)
- 集成 Ollama 本地大模型
- 支持上下文对话（保留最近5轮）
- 智能、简洁的回复风格

### 4. 语音合成 (TTS)
- 基于 Matcha-TTS 的中文语音合成
- 支持中文、数字、日期等场景
- 自然流畅的语音输出

### 5. 连续对话
- 唤醒一次即可进入对话模式
- 支持连续多轮对话，无需重复唤醒
- 30秒无交互自动退出对话模式

### 6. RAG 知识增强（可选）
- 本地检索增强问答，优先依据知识库回答
- 首次运行自动构建向量库（来源：`RAG/realistic_restaurant_reviews.csv`）
- 依赖 Ollama Embeddings（`bge-m3`）与 Chroma

## 🎯 使用场景

- 智能家居语音控制
- 语音问答助手
- 语音笔记助手
- 日常语音交互

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install sherpa-onnx sounddevice soundfile requests numpy

# 如需启用 RAG，请额外安装（建议）：
pip install -r RAG/requirements.txt

# 并确保本地有嵌入模型（用于向量检索）：
ollama pull bge-m3
```

### 2. 下载模型

项目需要以下模型文件（已预置在相应目录）：

- **关键词检测模型**: `kws/kws/`
  - encoder.int8.onnx
  - decoder.onnx
  - joiner.int8.onnx
  - tokens.txt
  - test_wavs/test_keywords.txt

- **语音识别模型**: `StreamingAsr/model/`
  - model.int8.onnx
  - tokens.txt
  - vad.onnx

- **语音合成模型**: `TTS/matcha-icefall-zh-baker/`
  - model-steps-3.onnx
  - vocos-22khz-univ.onnx
  - lexicon.txt
  - tokens.txt
  - dict/（字典目录）
  - phone.fst, date.fst, number.fst

### 3. 启动 Ollama 服务

```bash
# 确保 Ollama 已安装
ollama serve

# 在另一个终端拉取模型（如果还没有）
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

### 4. 运行助手

```bash
python voice_assistant.py
```

> 提示：若 RAG 依赖缺失或初始化失败，程序会自动降级为普通对话模式。

## 💡 使用流程

1. **启动程序**
   ```
   python voice_assistant.py
   ```

2. **唤醒助手**
   - 对着麦克风说："你好傻妞" 或 "傻妞"
   - 听到"主人，有什么可以帮您的？"表示唤醒成功

3. **开始对话**
   - 直接说出你的问题或指令
   - 等待2秒静音后自动识别完成
   - 傻妞会回答你的问题

4. **连续对话**
   - 无需再次唤醒，直接继续说话即可
   - 支持多轮连续对话

5. **退出对话**
   - 30秒内无交互，傻妞会自动退下
   - 听到"傻妞退下了，需要的时候再召唤傻妞哦"
   - 按 Ctrl+C 可完全退出程序

## ⚙️ 配置说明

可在 `voice_assistant.py` 中修改以下配置：

```python
# Ollama 配置
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# 音频参数
SAMPLE_RATE = 16000
VAD_SILENCE_DURATION = 3.0  # 静音判断时长（秒）

# 系统提示词
SYSTEM_PROMPT = """..."""  # 可自定义傻妞的性格和行为
```

## 📁 项目结构

```
smart_voice_assistant/
├── voice_assistant.py          # 主程序
├── kws/                         # 关键词检测模块
│   └── kws/
│       ├── encoder.int8.onnx
│       ├── decoder.onnx
│       ├── joiner.int8.onnx
│       ├── tokens.txt
│       └── test_wavs/
│           └── test_keywords.txt
├── StreamingAsr/                # 语音识别模块
│   ├── infer.py                 # 独立测试脚本
│   └── model/
│       ├── model.int8.onnx
│       ├── tokens.txt
│       └── vad.onnx
├── TTS/                         # 语音合成模块
│   ├── infer.py                 # 独立测试脚本
│   └── matcha-icefall-zh-baker/
│       ├── model-steps-3.onnx
│       ├── vocos-22khz-univ.onnx
│       ├── lexicon.txt
│       ├── tokens.txt
│       ├── dict/
│       └── *.fst
└── ollama/                      # 大模型接口示例
    └── request.py
```

## 🔧 独立测试

各模块可单独测试：

```bash
# 测试关键词检测
cd kws
python infer.py

# 测试语音识别
cd StreamingAsr
python infer.py

# 测试语音合成
cd TTS
python infer.py

# 测试 Ollama 接口
cd ollama
python request.py

# 测试 RAG（可选）
cd ../RAG
python main.py
```

## 🐛 常见问题

### 1. 找不到麦克风设备
- 检查麦克风是否正确连接
- 确认系统麦克风权限已开启

### 2. Ollama 连接失败
- 确认 Ollama 服务已启动：`ollama serve`
- 检查端口是否为 11434
- 确认模型已下载：`ollama list`

### 3. 语音识别不准确
- 确保环境安静
- 麦克风距离适中（20-30cm）
- 说话清晰、语速适中

### 4. 模型加载失败
- 检查模型文件是否完整
- 确认文件路径正确（不含中文）
- 查看错误日志定位问题

### 5. 唤醒词识别不灵敏
- 调整 `keywords_threshold` 参数（降低阈值更容易触发）
- 确保发音清晰完整
- 可在 `test_keywords.txt` 中添加自定义唤醒词

## 🎨 自定义

### 修改唤醒词

编辑 `kws/kws/test_wavs/test_keywords.txt`，添加拼音格式的关键词：

```
sh ǎ n iū @傻妞
n ǐ h ǎo sh ǎ n iū @你好傻妞
```

### 修改语音合成音色

在 `voice_assistant.py` 的 TTS 配置中调整：

```python
audio = self.tts.generate(text, speed=1.0)  # 调整 speed 参数
```

### 修改超时时间

```python
text, timed_out = self.listen_and_recognize_with_timeout(30)  # 修改秒数
```

## 📝 技术栈

- **语音识别**: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (SenseVoice)
- **语音合成**: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (Matcha-TTS)
- **关键词检测**: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (KWS)
- **大模型**: [Ollama](https://ollama.ai/) (Qwen3)
- **音频处理**: sounddevice, soundfile, numpy

## 📄 RAG 使用说明（可选）

- 数据来源：`RAG/realistic_restaurant_reviews.csv`（已含示例数据）
- 首次运行会在 `RAG/chrome_langchain_db` 持久化向量库
- 运行主程序时，若依赖齐全会自动从向量库检索并将结果注入 LLM 提示词
- 无需额外配置，如需替换数据，更新 CSV 并删除向量库目录后重启即可

## 📄 License

本项目仅供学习交流使用。

## 🙏 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - 提供优秀的语音处理工具
- [Ollama](https://ollama.ai/) - 提供便捷的本地大模型部署方案
- 各开源模型的作者和贡献者

## 💬 功能建议与反馈

我们欢迎并期待您的反馈！如果您有以下任何想法，请随时提出 Issue：

### 🌟 功能需求
- 想要添加新功能？告诉我们您的想法
- 需要支持其他语言（英语等）？我们可以考虑添加
- 希望支持更多唤醒词或自定义唤醒词？
- 想要其他语音音色或说话风格？

### 🐛 问题反馈
- 遇到了 Bug？请详细描述问题和重现步骤
- 性能或兼容性问题？让我们知道您的使用环境

### 🔧 改进建议
- 有更好的实现方案？欢迎讨论
- 文档不清楚？告诉我们需要改进的地方

**提交 Issue**: [https://github.com/xinliu9451/smart_voice_assistant/issues](https://github.com/xinliu9451/smart_voice_assistant/issues)

---

**享受与傻妞的智能对话吧！** 🎉

