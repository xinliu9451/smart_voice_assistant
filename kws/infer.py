import argparse
import sys
import time
import wave
from pathlib import Path
from typing import List, Tuple
import psutil
import os

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_onnx

# 全局变量
killed = False
sample_rate = 16000  # 采样率


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def create_keyword_spotter():
    kws = sherpa_onnx.KeywordSpotter(
        tokens="./kws/tokens.txt",
        encoder="./kws/encoder.int8.onnx",
        decoder="./kws/decoder.onnx",
        joiner="./kws/joiner.int8.onnx",
        num_threads=2,
        keywords_file="./kws/test_wavs/test_keywords.txt",
        keywords_score=2.0,          # 关键词得分，越大越容易检测
        keywords_threshold=0.1,      # 检测阈值，越小越容易触发（降低阈值）
        max_active_paths=4,          # 活跃路径数
        num_trailing_blanks=1,       # 关键词后面的空白数
        provider="cpu",
    )

    return kws


def get_memory_usage():
    """获取当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # 转换为MB


def main():
    global killed
    
    # 检测可用的音频设备
    devices = sd.query_devices()
    if len(devices) == 0:
        print("未找到麦克风设备")
        sys.exit(0)

    # print(devices)

    # 可以通过 sd.default.device[0] = xxx 来选择不同的输入设备
    # 其中 xxx 是设备编号
    default_input_device_idx = sd.default.device[0]
    print(f'使用默认设备: {devices[default_input_device_idx]["name"]}')

    # 记录初始内存使用量
    initial_memory = get_memory_usage()
    print(f"初始内存使用量: {initial_memory:.2f} MB")
    
    # 创建关键词检测器并记录内存使用量
    print("正在加载模型...")
    kws = create_keyword_spotter()
    model_loaded_memory = get_memory_usage()
    print(f"模型加载后内存使用量: {model_loaded_memory:.2f} MB")
    print(f"模型加载占用内存: {model_loaded_memory - initial_memory:.2f} MB")

    print("\n----------开始实时关键词检测----------")
    print("按 Ctrl+C 停止检测")
    
    print("开始监听... 请说话")
    print("提示：请完整说出关键词，如'打开空调'、'声音大一点'等")
    
    # 创建关键词检测流 - 使用官方推荐的连续流式处理
    stream = kws.create_stream()
    
    idx = 0
    samples_per_read = int(0.1 * sample_rate)  # 0.1秒 = 100毫秒
    audio_chunks_count = 0
    
    print(f"每次读取 {samples_per_read} 个采样点（{samples_per_read/sample_rate:.1f}秒）")
    print("音频输入状态监控开始...")
    
    try:
        # 使用sounddevice直接流式处理，不用队列和线程
        with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
            while not killed:
                samples, _ = s.read(samples_per_read)  # 阻塞读取
                samples = samples.reshape(-1)
                
                # 监控音频输入
                audio_chunks_count += 1
                audio_level = np.abs(samples).mean()
                
                # 每50个chunk（5秒）显示一次状态
                if audio_chunks_count % 50 == 0:
                    print(f"[状态] 已处理 {audio_chunks_count} 个音频块，音频强度: {audio_level:.4f}")
                
                # 如果音频强度太低，可能是静音
                if audio_level > 0.001:  # 有声音输入
                    if audio_chunks_count % 50 == 1:  # 只在第一次检测到声音时提示
                        print(f"[音频] 检测到音频输入，强度: {audio_level:.4f}")
                
                # 直接输入到检测流
                stream.accept_waveform(sample_rate, samples)
                
                # 检测关键词
                while kws.is_ready(stream):
                    kws.decode_stream(stream)
                    result = kws.get_result(stream)
                    if result:
                        print(f"\n🎯 [{idx}] 检测到关键词: {result}")
                        print("继续监听...")
                        idx += 1
                        # 检测到关键词后立即重置流
                        kws.reset_stream(stream)
                
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在停止...")
    
    # 清理资源
    killed = True
    print("程序退出")


if __name__ == "__main__":
    main()