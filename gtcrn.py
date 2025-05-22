"""
gtcrn.py gtcrn ans.
"""

import os
import wave

import ncnn
import numpy as np


class GTCRNNCNN:
    def __init__(self, hop_length=256, nfft=512, config=None):
        cwd_dir = os.path.split(os.path.realpath(__file__))[0]
        net = ncnn.Net()
        net.load_param(f"{cwd_dir}/gtcrn.ncnn.param")
        net.load_model(f"{cwd_dir}/gtcrn.ncnn.bin")
        self.net = net

        self.conv_cache = np.zeros([16, 32, 33], dtype=np.float32)
        self.tra_cache = np.zeros([6, 16], dtype=np.float32)
        self.inter_cache = np.zeros([1, 33, 32], dtype=np.float32)

        self.in_buffer = np.zeros([1, nfft - hop_length], dtype=np.float32)
        self.out_buffer = np.zeros([1, nfft - hop_length], dtype=np.float32)
        self.hop_length = hop_length
        self.nfft = nfft
        self.win = np.hanning(512) ** 0.5
        self.mix = np.zeros([2, 257], dtype=np.float32)

    def __call__(self, frame):
        x = np.concatenate((self.in_buffer, frame), axis=1)
        self.in_buffer = x[:, self.hop_length :]
        x0 = x * self.win
        spec = np.fft.rfft(x0).astype("complex64")

        self.mix[0, :] = spec.real.flatten()
        self.mix[1, :] = spec.imag.flatten()

        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(self.mix).clone())
        ex.input("in1", ncnn.Mat(self.conv_cache).clone())
        ex.input("in2", ncnn.Mat(self.tra_cache).clone())
        ex.input("in3", ncnn.Mat(self.inter_cache).clone())
        _, out0 = ex.extract("out0")
        _, out1 = ex.extract("out1")
        _, out2 = ex.extract("out2")
        _, out3 = ex.extract("out3")

        out = np.array(out0)
        self.conv_cache = np.array(out1)
        self.tra_cache = np.array(out2)
        self.inter_cache = np.array(out3)

        spec = out[0, :] + 1j * out[1, :]
        enhanced = np.fft.irfft(spec.reshape(1, -1)) * self.win

        y = self.out_buffer + enhanced[0, : self.hop_length]
        self.out_buffer = enhanced[0, self.hop_length :]
        return y


def audioread(filename):
    with wave.open(filename) as f:
        sample_width = f.getsampwidth()
        sample_rate = f.getframerate()
        channels = f.getnchannels()
        if sample_width == 2:
            sample_max = 32768
        elif sample_width == 4:
            sample_max = 2147483648
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_int16 = samples_int16.reshape(-1, channels).T
        samples_float32 = samples_int16.astype(np.float32)
        samples_float32 = samples_float32 / sample_max

    return samples_float32, sample_rate


def audiowrite(filename, y, fs, nbits=16):
    if y.ndim == 1:
        y = y.reshape(1, -1)

    channels = y.shape[0]
    y = (y.T.reshape(-1) * 32767).clip(-32768, 32767).astype(np.int16)

    with wave.open(filename, "w") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes(y.tobytes())


if __name__ == "__main__":
    wav_in = "mix.wav"
    pin = 0
    pout = 0
    framesize = 256
    x, fs = audioread(wav_in)
    num_sampels = x.shape[-1]
    y = np.zeros_like(x)
    gtcrn = GTCRNNCNN()

    while pin + framesize < num_sampels:
        frame = x[:, pin : pin + framesize]
        frame = frame.reshape(1, -1)
        out = gtcrn(frame)
        y[:, pout : pout + framesize] = out

        pin += framesize
        pout += framesize

    audiowrite("out.wav", y, fs)
