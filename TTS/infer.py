#!/usr/bin/env python3
"""
简化的中文语音合成脚本，只支持 matcha-icefall-zh-baker 模型
边生成边播放音频

使用方法:
直接运行 python infer.py，在代码开头修改默认参数
"""

import logging
import queue
import sys
import threading
import time
import os

import numpy as np
import sherpa_onnx
import soundfile as sf

try:
    import sounddevice as sd
except ImportError:
    print("请先安装 sounddevice: pip install sounddevice")
    sys.exit(-1)


# ==================== 默认参数配置 ====================
TEXT = "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。"
OUTPUT_FILENAME = "output.wav"
SPEED = 1.0
MODEL_DIR = "D:/Desktop/smart_voice_assistant/TTS/matcha-icefall-zh-baker"
NUM_THREADS = 2
# ====================================================


# 全局变量
buffer = queue.Queue()  # 音频缓冲区
started = False  # 是否开始播放
stopped = False  # 是否生成完成
killed = False  # 是否被中断
sample_rate = None
event = threading.Event()
first_message_time = None


def generated_audio_callback(samples: np.ndarray, progress: float):
    """音频生成回调函数，每生成一段音频就会被调用"""
    global first_message_time, started
    
    if first_message_time is None:
        first_message_time = time.time()
    
    buffer.put(samples)
    
    if started is False:
        logging.info("开始播放...")
    started = True
    
    # 返回1继续生成，返回0停止生成
    if killed:
        return 0
    return 1


def play_audio_callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
    """音频播放回调函数"""
    if killed or (started and buffer.empty() and stopped):
        event.set()
    
    if buffer.empty():
        outdata.fill(0)
        return
    
    n = 0
    while n < frames and not buffer.empty():
        remaining = frames - n
        k = buffer.queue[0].shape[0]
        
        if remaining <= k:
            outdata[n:, 0] = buffer.queue[0][:remaining]
            buffer.queue[0] = buffer.queue[0][remaining:]
            n = frames
            if buffer.queue[0].shape[0] == 0:
                buffer.get()
            break
        
        outdata[n : n + k, 0] = buffer.get()
        n += k
    
    if n < frames:
        outdata[n:, 0] = 0


def play_audio():
    """播放音频线程"""
    with sd.OutputStream(
        channels=1,
        callback=play_audio_callback,
        dtype="float32",
        samplerate=sample_rate,
        blocksize=1024,
    ):
        event.wait()
    
    logging.info("播放结束")


def main():
    global sample_rate, stopped, killed
    
    # 构建模型文件路径
    acoustic_model = os.path.join(MODEL_DIR, "model-steps-3.onnx")
    vocoder = os.path.join(MODEL_DIR, "vocos-22khz-univ.onnx")
    lexicon = os.path.join(MODEL_DIR, "lexicon.txt")
    tokens = os.path.join(MODEL_DIR, "tokens.txt")
    dict_dir = os.path.join(MODEL_DIR, "dict")
    rule_fsts = f"{os.path.join(MODEL_DIR, 'phone.fst')},{os.path.join(MODEL_DIR, 'date.fst')},{os.path.join(MODEL_DIR, 'number.fst')}"
    
    # 配置TTS模型
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
            num_threads=NUM_THREADS,
        ),
        rule_fsts=rule_fsts,
        max_num_sentences=1,
    )
    
    if not tts_config.validate():
        raise ValueError("模型配置验证失败，请检查模型文件路径")

    # 加载模型
    logging.info("正在加载模型...")
    tts = sherpa_onnx.OfflineTts(tts_config)
    logging.info("模型加载完成")
    
    sample_rate = tts.sample_rate
    
    # 启动音频播放线程
    play_back_thread = threading.Thread(target=play_audio)
    play_back_thread.start()
    
    # 生成语音
    logging.info(f"正在生成语音: {TEXT}")
    start_time = time.time()
    audio = tts.generate(
        TEXT,
        speed=SPEED,
        callback=generated_audio_callback,
    )
    end_time = time.time()
    logging.info("语音生成完成!")
    stopped = True

    if len(audio.samples) == 0:
        print("生成音频失败，请检查输入文字")
        killed = True
        play_back_thread.join()
        return

    # 保存音频文件
    sf.write(
        OUTPUT_FILENAME,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )
    
    # 输出统计信息
    elapsed_seconds = end_time - start_time
    audio_duration = len(audio.samples) / audio.sample_rate
    rtf = elapsed_seconds / audio_duration
    first_audio_time = first_message_time - start_time if first_message_time else 0
    
    logging.info(f"文本内容: {TEXT}")
    logging.info(f"首次响应时间: {first_audio_time:.3f}秒")
    logging.info(f"总处理时间: {elapsed_seconds:.3f}秒")
    logging.info(f"音频时长: {audio_duration:.3f}秒")
    logging.info(f"实时率(RTF): {rtf:.3f}x")
    logging.info(f"*** 音频已保存到: {OUTPUT_FILENAME} ***")
    
    print("\n   >>>>>>>>> 可以按 Ctrl+C 停止播放 <<<<<<<<<<\n")
    
    play_back_thread.join()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在退出...")
        killed = True
        sys.exit(0)
