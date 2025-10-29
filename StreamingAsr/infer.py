import argparse
import queue
import sys
import threading
import time
from pathlib import Path

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

killed = False
recording_thread = None
sample_rate = 16000  # Please don't change it

# buffer saves audio samples to be played
samples_queue = queue.Queue()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--vad-model",
        type=str,
        default="./model/vad.onnx",
        help="Path to vad.onnx",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="./model/tokens.txt",
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--model",
        default="./model/model.int8.onnx",
        type=str,
        help="Path to the model.onnx",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--hr-dict-dir",
        type=str,
        default="",
        help="If not empty, it is the jieba dict directory for homophone replacer",
    )

    parser.add_argument(
        "--hr-lexicon",
        type=str,
        default="",
        help="If not empty, it is the lexicon.txt for homophone replacer",
    )

    parser.add_argument(
        "--hr-rule-fsts",
        type=str,
        default="",
        help="If not empty, it is the replace.fst for homophone replacer",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
    )


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    assert_file_exists(args.model)
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=args.model,
        tokens=args.tokens,
        num_threads=args.num_threads,
        use_itn=True,  # 启用逆文本标准化以添加标点符号
        debug=False,
        hr_dict_dir=args.hr_dict_dir,
        hr_rule_fsts=args.hr_rule_fsts,
        hr_lexicon=args.hr_lexicon,
    )

    return recognizer


def start_recording():
    # You can use any value you like for samples_per_read
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while not killed:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            samples = np.copy(samples)
            samples_queue.put(samples)


def main():
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)

    # If you want to select a different input device, please use
    # sd.default.device[0] = xxx
    # where xxx is the device number

    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    args = get_args()
    

    
    try:
        assert_file_exists(args.tokens)
    except AssertionError as e:
        print(f"  ✗ tokens file not found: {e}")
        sys.exit(-1)
    
    try:
        assert_file_exists(args.vad_model)
    except AssertionError as e:
        print(f"  ✗ vad_model file not found: {e}")
        sys.exit(-1)
    
    try:
        assert_file_exists(args.model)
    except AssertionError as e:
        print(f"  ✗ model file not found: {e}")
        sys.exit(-1)

    assert args.num_threads > 0, args.num_threads

    print("\nCreating recognizer. Please wait...")
    
    try:
        recognizer = create_recognizer(args)
        print("Recognizer created successfully!")
    except Exception as e:
        print(f"Failed to create recognizer: {e}")
        sys.exit(-1)

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = args.vad_model
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.1  # seconds
    config.silero_vad.min_speech_duration = 0.25  # seconds
    # If the current segment is larger than this value, then it increases
    # the threshold to 0.9 internally. After detecting this segment,
    # it resets the threshold to its original value.
    config.silero_vad.max_speech_duration = 8  # seconds
    config.sample_rate = sample_rate

    window_size = config.silero_vad.window_size

    try:
        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
        print("VAD initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize VAD: {e}")
        sys.exit(-1)

    print("Started! Please speak")

    buffer = []

    global recording_thread
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()

    display = sherpa_onnx.Display()

    started = False
    started_time = None

    offset = 0
    while not killed:
        samples = samples_queue.get()  # a blocking read

        buffer = np.concatenate([buffer, samples])
        while offset + window_size < len(buffer):
            vad.accept_waveform(buffer[offset : offset + window_size])
            if not started and vad.is_speech_detected():
                started = True
                started_time = time.time()
            offset += window_size

        if not started:
            if len(buffer) > 10 * window_size:
                offset -= len(buffer) - 10 * window_size
                buffer = buffer[-10 * window_size :]

        if started and time.time() - started_time > 0.2:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, buffer)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                display.update_text(text)
                display.display()

            started_time = time.time()

        while not vad.empty():
            # In general, this while loop is executed only once
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, vad.front.samples)

            vad.pop()
            recognizer.decode_stream(stream)

            text = stream.result.text.strip()

            display.update_text(text)

            buffer = []
            offset = 0
            started = False
            started_time = None

            display.finalize_current_sentence()
            display.display()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        killed = True
        if recording_thread:
            recording_thread.join()
        print("\nCaught Ctrl + C. Exiting")
