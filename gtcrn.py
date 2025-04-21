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

        self.en_conv_cache1 = np.zeros([16, 2, 33], dtype=np.float32)
        self.en_conv_cache2 = np.zeros([16, 4, 33], dtype=np.float32)
        self.en_conv_cache3 = np.zeros([16, 10, 33], dtype=np.float32)
        self.de_conv_cache1 = np.zeros([16, 10, 33], dtype=np.float32)
        self.de_conv_cache2 = np.zeros([16, 4, 33], dtype=np.float32)
        self.de_conv_cache3 = np.zeros([16, 2, 33], dtype=np.float32)
        self.en_tra_cache1 = np.zeros([1, 16], dtype=np.float32)
        self.en_tra_cache2 = np.zeros([1, 16], dtype=np.float32)
        self.en_tra_cache3 = np.zeros([1, 16], dtype=np.float32)
        self.de_tra_cache1 = np.zeros([1, 16], dtype=np.float32)
        self.de_tra_cache2 = np.zeros([1, 16], dtype=np.float32)
        self.de_tra_cache3 = np.zeros([1, 16], dtype=np.float32)
        self.inter_cache1 = np.zeros([1, 33, 16], dtype=np.float32)
        self.inter_cache2 = np.zeros([1, 33, 16], dtype=np.float32)

        self.in_buffer = np.zeros([1, nfft - hop_length], dtype=np.float32)
        self.out_buffer = np.zeros([1, nfft - hop_length], dtype=np.float32)
        self.hop_length = hop_length
        self.nfft = nfft
        self.win = np.hanning(512) ** 0.5
        self.mix = np.zeros([257, 1, 2], dtype=np.float32)

    def __call__(self, frame):
        x = np.concatenate((self.in_buffer, frame), axis=1)
        self.in_buffer = x[:, self.hop_length :]
        x0 = x * self.win
        spec = np.fft.rfft(x0).astype("complex64")

        self.mix[:, 0, 0] = spec.real.flatten()
        self.mix[:, 0, 1] = spec.imag.flatten()

        ex = self.net.create_extractor()
        ex.input("in0", ncnn.Mat(self.mix).clone())
        ex.input("in1", ncnn.Mat(self.en_conv_cache1).clone())
        ex.input("in2", ncnn.Mat(self.en_conv_cache2).clone())
        ex.input("in3", ncnn.Mat(self.en_conv_cache3).clone())

        ex.input("in4", ncnn.Mat(self.de_conv_cache1).clone())
        ex.input("in5", ncnn.Mat(self.de_conv_cache2).clone())
        ex.input("in6", ncnn.Mat(self.de_conv_cache3).clone())

        ex.input("in7", ncnn.Mat(self.en_tra_cache1).clone())
        ex.input("in8", ncnn.Mat(self.en_tra_cache2).clone())
        ex.input("in9", ncnn.Mat(self.en_tra_cache3).clone())

        ex.input("in10", ncnn.Mat(self.de_tra_cache1).clone())
        ex.input("in11", ncnn.Mat(self.de_tra_cache2).clone())
        ex.input("in12", ncnn.Mat(self.de_tra_cache3).clone())

        ex.input("in13", ncnn.Mat(self.inter_cache1).clone())
        ex.input("in14", ncnn.Mat(self.inter_cache2).clone())

        _, out0 = ex.extract("out0")
        _, out1 = ex.extract("out1")
        _, out2 = ex.extract("out2")
        _, out3 = ex.extract("out3")
        _, out4 = ex.extract("out4")
        _, out5 = ex.extract("out5")
        _, out6 = ex.extract("out6")
        _, out7 = ex.extract("out7")
        _, out8 = ex.extract("out8")
        _, out9 = ex.extract("out9")
        _, out10 = ex.extract("out10")
        _, out11 = ex.extract("out11")
        _, out12 = ex.extract("out12")
        _, out13 = ex.extract("out13")
        _, out14 = ex.extract("out14")

        out = np.array(out0)
        self.en_conv_cache1 = np.array(out1)
        self.en_conv_cache2 = np.array(out2)
        self.en_conv_cache3 = np.array(out3)
        self.de_conv_cache1 = np.array(out4)
        self.de_conv_cache2 = np.array(out5)
        self.de_conv_cache3 = np.array(out6)
        self.en_tra_cache1 = np.array(out7)
        self.en_tra_cache2 = np.array(out8)
        self.en_tra_cache3 = np.array(out9)
        self.de_tra_cache1 = np.array(out10)
        self.de_tra_cache2 = np.array(out11)
        self.de_tra_cache3 = np.array(out12)
        self.inter_cache1 = np.array(out13)
        self.inter_cache2 = np.array(out14)

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
