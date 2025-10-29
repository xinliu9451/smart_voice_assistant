 #!/usr/bin/env python3
"""
语音智能助手 - 傻妞
功能流程：
1. 关键词唤醒（"你好傻妞" 或 "傻妞"）
2. TTS播报"主人，有什么吩咐？"
3. 语音识别（带VAD，3秒静音后结束）
4. 调用大模型生成回复
5. TTS播报回复
6. 返回关键词检测，等待下次唤醒
"""

import os
import sys
import time
import numpy as np
import requests
import soundfile as sf

try:
    import sounddevice as sd
except ImportError:
    print("请先安装 sounddevice: pip install sounddevice")
    sys.exit(-1)

import sherpa_onnx

# ==================== 配置参数 ====================
# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径（相对于脚本目录）
KWS_TOKENS = os.path.join(SCRIPT_DIR, "kws", "kws", "tokens.txt")
KWS_ENCODER = os.path.join(SCRIPT_DIR, "kws", "kws", "encoder.int8.onnx")
KWS_DECODER = os.path.join(SCRIPT_DIR, "kws", "kws", "decoder.onnx")
KWS_JOINER = os.path.join(SCRIPT_DIR, "kws", "kws", "joiner.int8.onnx")
KWS_KEYWORDS_FILE = os.path.join(SCRIPT_DIR, "kws", "kws", "test_wavs", "test_keywords.txt")

ASR_MODEL = os.path.join(SCRIPT_DIR, "StreamingAsr", "model", "model.int8.onnx")
ASR_TOKENS = os.path.join(SCRIPT_DIR, "StreamingAsr", "model", "tokens.txt")
VAD_MODEL = os.path.join(SCRIPT_DIR, "StreamingAsr", "model", "vad.onnx")

TTS_MODEL_DIR = os.path.join(SCRIPT_DIR, "TTS", "matcha-icefall-zh-baker")

# Ollama配置
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# 系统提示词
SYSTEM_PROMPT = """你是傻妞，一个智能语音助手。你的特点：
1. 你叫傻妞，是主人的贴心助手
2. 你聪明、活泼、友善，喜欢帮助主人
3. 回答不要太长，通常控制在10-100字之间，因为要语音播报
4. 语气亲切自然，可以称呼对方为"主人"
5. 如果不知道答案，诚实说不知道，不要编造信息"""

# 音频参数
SAMPLE_RATE = 16000
VAD_SILENCE_DURATION = 2.0  # 静音持续时间（秒）

# ====================================================

# 全局变量
killed = False
 
# RAG 相关（可选）
# 如果依赖未安装或初始化失败，将自动降级为普通对话
RAG_ENABLED = False
rag_retriever = None
try:
    # 加入 RAG 模块路径并尝试加载检索器
    sys.path.append(os.path.join(SCRIPT_DIR, "RAG"))
    from vector import retriever as rag_retriever  # type: ignore
    RAG_ENABLED = True
except Exception as e:
    print(f"RAG 未启用（可选）：{e}")


class VoiceAssistant:
    """语音助手主类"""
    
    def __init__(self):
        self.kws = None
        self.asr = None
        self.vad = None
        self.tts = None
        self.conversation_history = []
    
    def augment_with_rag(self, question: str) -> str:
        """使用本地 RAG 知识库为问题检索相关资料，并返回拼接后的上下文文本。
        依赖 RAG/ 下的向量库，第一次运行会自动构建。
        """
        if not RAG_ENABLED or rag_retriever is None:
            return ""
        try:
            docs = rag_retriever.invoke(question)
            if not docs:
                return ""
            # 将检索到的内容简洁拼接，避免提示词过长
            contents = []
            for d in docs:
                content = getattr(d, "page_content", None)
                if content:
                    contents.append(str(content))
            reviews_text = "\n".join(contents[:5])
            return reviews_text
        except Exception as e:
            print(f"RAG 调用失败：{e}")
            return ""
    
    def check_files(self):
        """检查所有必需的文件是否存在"""
        files_to_check = {
            "KWS tokens": KWS_TOKENS,
            "KWS encoder": KWS_ENCODER,
            "KWS decoder": KWS_DECODER,
            "KWS joiner": KWS_JOINER,
            "KWS keywords": KWS_KEYWORDS_FILE,
            "ASR model": ASR_MODEL,
            "ASR tokens": ASR_TOKENS,
            "VAD model": VAD_MODEL,
            "TTS acoustic": os.path.join(TTS_MODEL_DIR, "model-steps-3.onnx"),
            "TTS vocoder": os.path.join(TTS_MODEL_DIR, "vocos-22khz-univ.onnx"),
        }
        
        missing = []
        for name, path in files_to_check.items():
            if not os.path.exists(path):
                missing.append(f"  ❌ {name}: {path}")
        
        if missing:
            print("以下文件不存在:")
            for m in missing:
                print(m)
            return False
        
        print("✓ 所有文件检查通过")
        return True
        
    def initialize(self):
        """初始化所有模型"""
        print("\n" + "=" * 50)
        print("正在初始化傻妞语音助手...")
        print("=" * 50)
        
        if not self.check_files():
            sys.exit(-1)
        
        # 初始化关键词检测
        print("1. 加载关键词唤醒模型...")
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=KWS_TOKENS,
            encoder=KWS_ENCODER,
            decoder=KWS_DECODER,
            joiner=KWS_JOINER,
            num_threads=2,
            keywords_file=KWS_KEYWORDS_FILE,
            keywords_score=2.0,
            keywords_threshold=0.1,
            max_active_paths=4,
            num_trailing_blanks=1,
            provider="cpu",
        )
        print("   ✓ 关键词检测模型加载完成")
        
        # 初始化语音识别（严格按 StreamingAsr/infer.py 的方式）
        print("2. 加载语音识别模型...")
        self.asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=ASR_MODEL,
            tokens=ASR_TOKENS,
            num_threads=2,
            use_itn=True,  # 与 infer.py 一致
            language="zh",
            debug=False,
            hr_dict_dir="",   # 与 infer.py 默认一致
            hr_rule_fsts="",
            hr_lexicon="",
        )
        print("   ✓ 语音识别模型加载完成")
        
        # 初始化VAD
        print("3. 加载VAD模型...")
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = VAD_MODEL
        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_silence_duration = VAD_SILENCE_DURATION
        vad_config.silero_vad.min_speech_duration = 0.25
        vad_config.silero_vad.max_speech_duration = 15
        vad_config.sample_rate = SAMPLE_RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)
        print("   ✓ VAD模型加载完成")
        
        # 初始化TTS
        print("4. 加载TTS模型...")
        acoustic_model = os.path.join(TTS_MODEL_DIR, "model-steps-3.onnx")
        vocoder = os.path.join(TTS_MODEL_DIR, "vocos-22khz-univ.onnx")
        lexicon = os.path.join(TTS_MODEL_DIR, "lexicon.txt")
        tokens = os.path.join(TTS_MODEL_DIR, "tokens.txt")
        dict_dir = os.path.join(TTS_MODEL_DIR, "dict")
        rule_fsts = f"{os.path.join(TTS_MODEL_DIR, 'phone.fst')},{os.path.join(TTS_MODEL_DIR, 'date.fst')},{os.path.join(TTS_MODEL_DIR, 'number.fst')}"
        
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=acoustic_model,
                    vocoder=vocoder,
                    lexicon=lexicon,
                    tokens=tokens,
                    dict_dir=dict_dir,
                ),
                provider="cpu",
                debug=0,
                num_threads=2,
            ),
            rule_fsts=rule_fsts,
            max_num_sentences=1,
        )
        self.tts = sherpa_onnx.OfflineTts(tts_config)
        print("   ✓ TTS模型加载完成")
        
        print("=" * 50)
        print("傻妞已就绪，等待唤醒...")
        print("唤醒词：'你好傻妞' 或 '傻妞'")
        print("=" * 50 + "\n")
    
    def play_audio(self, audio_samples, sample_rate):
        """播放音频"""
        sd.play(audio_samples, samplerate=sample_rate)
        sd.wait()
    
    def tts_speak(self, text):
        """语音播报"""
        print(f"🔊 傻妞: {text}")
        audio = self.tts.generate(text, speed=1.0)
        if len(audio.samples) > 0:
            self.play_audio(audio.samples, audio.sample_rate)
    
    def wait_for_keyword(self):
        """等待关键词唤醒"""
        print("🎧 正在监听关键词...")
        stream = self.kws.create_stream()
        samples_per_read = int(0.1 * SAMPLE_RATE)
        
        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE) as audio_stream:
            while not killed:
                samples, _ = audio_stream.read(samples_per_read)
                samples = samples.reshape(-1)
                stream.accept_waveform(SAMPLE_RATE, samples)
                
                while self.kws.is_ready(stream):
                    self.kws.decode_stream(stream)
                    result = self.kws.get_result(stream)
                    if result:
                        keyword = result.strip()
                        if "傻妞" in keyword:
                            print(f"✨ 检测到唤醒词: {keyword}")
                            self.kws.reset_stream(stream)
                            return True
                        self.kws.reset_stream(stream)
        return False
    
    def listen_and_recognize(self):
        """监听并识别语音（带VAD）"""
        print("👂 请说话...")
        
        buffer = []
        offset = 0
        window_size = self.vad.config.silero_vad.window_size
        samples_per_read = int(0.1 * SAMPLE_RATE)
        
        started = False
        started_time = None
        
        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE) as audio_stream:
            while not killed:
                samples, _ = audio_stream.read(samples_per_read)
                samples = samples.reshape(-1)
                
                buffer = np.concatenate([buffer, samples])
                
                # VAD处理
                while offset + window_size < len(buffer):
                    self.vad.accept_waveform(buffer[offset : offset + window_size])
                    if not started and self.vad.is_speech_detected():
                        started = True
                        started_time = time.time()
                        print("   🎤 开始说话...")
                    offset += window_size
                
                # 清理缓冲区
                if not started:
                    if len(buffer) > 10 * window_size:
                        offset -= len(buffer) - 10 * window_size
                        buffer = buffer[-10 * window_size :]
                
                # 实时识别（不打印，只用于内部处理）
                if started and time.time() - started_time > 0.2:
                    stream = self.asr.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, buffer)
                    self.asr.decode_stream(stream)
                    started_time = time.time()
                
                # 处理VAD检测到的语音段（检测到静音结束）
                while not self.vad.empty():
                    stream = self.asr.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, self.vad.front.samples)
                    self.vad.pop()
                    self.asr.decode_stream(stream)
                    text = stream.result.text.strip()
                    
                    buffer = []
                    offset = 0
                    started = False
                    started_time = None
                    
                    if text:
                        print(f"   ✓ 识别完成: {text}")
                        return text
        
        return ""

    def listen_and_recognize_with_timeout(self, idle_timeout_seconds: float):
        """在给定的空闲超时时间内等待并识别一段话。
        返回 (text, timed_out)。
        - text: 识别到的文本；如果超时或无内容则为空字符串
        - timed_out: 如果在 idle_timeout_seconds 内没有开始说话则为 True
        """
        print(f"⏳ 等待说话（最多{int(idle_timeout_seconds)}秒）...")
        start_wait_time = time.time()

        buffer = []
        offset = 0
        window_size = self.vad.config.silero_vad.window_size
        samples_per_read = int(0.1 * SAMPLE_RATE)

        started = False
        started_time = None

        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE) as audio_stream:
            while not killed:
                # 空闲超时（尚未开始说话）
                if not started and (time.time() - start_wait_time) > idle_timeout_seconds:
                    print("⌛ 超时未检测到说话")
                    return "", True

                samples, _ = audio_stream.read(samples_per_read)
                samples = samples.reshape(-1)

                buffer = np.concatenate([buffer, samples])

                # VAD处理
                while offset + window_size < len(buffer):
                    self.vad.accept_waveform(buffer[offset : offset + window_size])
                    if not started and self.vad.is_speech_detected():
                        started = True
                        started_time = time.time()
                        print("   🎤 开始说话...")
                    offset += window_size

                # 清理缓冲区
                if not started:
                    if len(buffer) > 10 * window_size:
                        offset -= len(buffer) - 10 * window_size
                        buffer = buffer[-10 * window_size :]

                # 实时识别（不打印，只用于内部处理）
                if started and time.time() - started_time > 0.2:
                    stream = self.asr.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, buffer)
                    self.asr.decode_stream(stream)
                    started_time = time.time()

                # 处理VAD检测到的语音段（检测到静音结束）
                while not self.vad.empty():
                    stream = self.asr.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, self.vad.front.samples)
                    self.vad.pop()
                    self.asr.decode_stream(stream)
                    text = stream.result.text.strip()

                    buffer = []
                    offset = 0
                    started = False
                    started_time = None

                    if text:
                        print(f"   ✓ 识别完成: {text}")
                        return text, False
                    
                    # VAD检测到静音段结束，但没识别到内容，重置等待时间
                    start_wait_time = time.time()
        
        return "", False
    
    def call_llm(self, user_input):
        """调用大模型"""
        # 构建提示词
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # RAG 检索增强：将相关资料注入到系统上下文中
        rag_context = self.augment_with_rag(user_input)
        if rag_context:
            messages.append({
                "role": "system",
                "content": (
                    "以下是与用户问题相关的参考资料，请优先参考作答；若资料未涵盖，再结合常识：\n" 
                    + rag_context
                )
            })
        
        # 添加历史对话（最近3轮）
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_input})
        
        # 构建prompt
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        
        print("🤔 傻妞正在思考...")
        
        # 调用Ollama API
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result['response'].strip()
            
            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": llm_response})
            
            # 只保留最近5轮
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return llm_response
        else:
            return "抱歉主人，我现在有点不舒服，请稍后再试。"
    
    def run(self):
        """主循环"""
        global killed
        
        while not killed:
            # 1) 唤醒一次
            if not self.wait_for_keyword():
                continue

            # 2) 问候
            self.tts_speak("主人，有什么可以帮您的？")

            # 3) 进入对话循环：不再需要再次唤醒
            while not killed:
                text, timed_out = self.listen_and_recognize_with_timeout(30)
                if timed_out:
                    # 30秒无人说话，退下
                    self.tts_speak("傻妞退下了，需要的时候再召唤傻妞哦。")
                    print("—— 退回唤醒等待 ——")
                    break

                if not text:
                    print("⚠️ 没有识别到有效内容")
                    self.tts_speak("主人，我没听清，请再说一遍。")
                    continue

                # 调用LLM
                reply = self.call_llm(text)
                # 播报
                self.tts_speak(reply)
                # 小停顿后继续等待下一句
                time.sleep(0.3)
                print("\n" + "=" * 50 + "\n")


def main():
    print("\n" + "=" * 60)
    print(" " * 20 + "傻妞语音智能助手")
    print("=" * 60)
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本所在目录: {SCRIPT_DIR}")
    
    # 检查麦克风
    devices = sd.query_devices()
    if len(devices) == 0:
        print("❌ 未找到麦克风设备")
        sys.exit(-1)
    
    default_input_device_idx = sd.default.device[0]
    print(f"使用麦克风: {devices[default_input_device_idx]['name']}")
    
    # 创建并运行助手
    assistant = VoiceAssistant()
    assistant.initialize()
    
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\n\n检测到 Ctrl+C，正在退出...")
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("傻妞已退出，再见！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
