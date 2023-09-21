import sys

try:
    import sounddevice as sd
except ImportError as e:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_ncnn
from moviepy.editor import AudioFileClip
import ffmpeg
from pathlib import Path
import wave
import numpy as np

def create_recognizer():
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt",
        encoder_param="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )
    return recognizer


def real_time_speak_translation():
    devices = sd.query_devices()
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    print("Started! Please speak")
    recognizer = create_recognizer()
    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    with sd.InputStream(
        channels=1, dtype="float32", samplerate=sample_rate
    ) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            recognizer.accept_waveform(sample_rate, samples)
            result = recognizer.text
            if last_result != result:
                last_result = result
                print(result)

def wave_file_translation(filename:str):
    recognizer = create_recognizer()
    with wave.open(filename) as f:
        assert f.getframerate() == recognizer.sample_rate, (
            f.getframerate(),
            recognizer.sample_rate,
        )
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        num_samples = f.getnframes()
        samples_per_read = int(1 * f.getframerate())  # 1 second = 1000 ms
        this_sample = 0
        result = ""
        while(this_sample < num_samples):
            samples = f.readframes(samples_per_read)
            this_sample += samples_per_read
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32)
            samples_float32 = samples_float32 /32768
            recognizer.accept_waveform(recognizer.sample_rate, samples_float32)
            tail_paddings = np.zeros(
                int(recognizer.sample_rate * 0.5), dtype=np.float32
            )
            recognizer.accept_waveform(recognizer.sample_rate, tail_paddings)
            if result != recognizer.text:
                sys.stdout.write(recognizer.text[len(result):])
                sys.stdout.flush()
                result = recognizer.text

        recognizer.input_finished()
        with open(filename[:-3]+'txt', 'w') as f:
            f.write(recognizer.text)
        # print(recognizer.text)

def Extract_video_audio(video_path, audio_path):
    if Path(audio_path).is_file():
        print("The audio file already exists. Please check.")
    else:
        my_audio_clip = AudioFileClip(video_path)
        my_audio_clip.write_audiofile(audio_path)

def format_audio(input_file, output_file):
    if Path(output_file).is_file():
        print("The formated audio already exists. Please check.")
    else:
        desired_sample_rate = 16000
        desired_bit_depth = 16
        desired_channels = 1
        ffmpeg.input(input_file).output(output_file, ac=desired_channels,ar=desired_sample_rate,bits_per_raw_sample=desired_bit_depth).run()

if __name__ == "__main__":

    video_file_path = "test.mp4"
    audio_file_path = "test.wav"
    format_audio_path = "test_format.wav"
    Extract_video_audio(video_file_path, audio_file_path)
    with wave.open(audio_file_path) as f:
        if f.getframerate != 16000:
            format_audio(audio_file_path, format_audio_path)
    # wave_file_translation(format_audio_path)
    try:
        # real_time_speak_translation()
        wave_file_translation(format_audio_path)
    except:
        print("LOL, I'm break down.")