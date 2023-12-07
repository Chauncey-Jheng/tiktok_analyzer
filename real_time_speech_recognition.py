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
import wave
import numpy as np
import subprocess

def create_recognizer():
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-09-30/tokens.txt",
        encoder_param="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-09-30/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-09-30/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-09-30/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-09-30/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-09-30/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-09-30/joiner_jit_trace-pnnx.ncnn.bin",
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
        with open(filename[:-4]+'_asr.txt', 'w') as f:
            f.write(recognizer.text)
        # print(recognizer.text)

def Extract_video_audio(video_path, audio_path):
    desired_sample_rate = 16000
    desired_bit_depth = 16
    desired_channels = 1
    ffmpeg_cmd = f"ffmpeg -y {video_path} -ac {desired_channels} -ar {desired_sample_rate} -sample_fmt s{desired_bit_depth} {audio_path}"
    subprocess.run(ffmpeg_cmd,stdout=subprocess.DEVNULL)

def video_speech_recognition(video_file_path):
    audio_file_path = video_file_path[:-3] + "wav"
    Extract_video_audio(video_file_path, audio_file_path)
    try:
        wave_file_translation(audio_file_path)
    except:
        print("LOL, I'm break down.")

if __name__ == "__main__":

    video_file_path = "live_test_dir\\test0.flv"
    audio_file_path = "live_test_dir\\test0.wav"
    Extract_video_audio(video_file_path, audio_file_path)
    try:
        wave_file_translation(audio_file_path)
    except:
        print("LOL, I'm break down.")