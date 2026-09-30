"""Synthetic stage sounds for tests: audience, CO2 jets, fireworks, confetti cannons."""
import numpy as np
import scipy.signal as ss

SR = 44100


def applause(seconds, level, rng):
    n = int(seconds * SR)
    out = np.zeros(n)
    for start in rng.integers(0, n - 800, int(seconds * 250)):  # 250 claps a second
        out[start:start + 800] += rng.standard_normal(800) * np.exp(-np.arange(800) / 120)
    out = ss.sosfilt(ss.butter(2, [400, 8000], "bandpass", fs=SR, output="sos"), out)
    return out / (np.sqrt(np.mean(out**2)) + 1e-12) * level


def cheering(seconds, level, rng):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    voices = sum(np.sin(2 * np.pi * (f + 20 * np.sin(2 * np.pi * rng.uniform(0.2, 2) * t)) * t) for f in rng.uniform(200, 900, 40))
    noise = ss.sosfilt(ss.butter(2, [300, 4000], "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
    out = voices / 40 + noise * 0.5
    return out / (np.sqrt(np.mean(out**2)) + 1e-12) * level


def co2_jet(seconds, level, rng):
    n = int(seconds * SR)
    env = np.minimum(1, np.arange(n) / (0.05 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.3 * SR))
    out = ss.sosfilt(ss.butter(2, [300, 12000], "bandpass", fs=SR, output="sos"), rng.standard_normal(n)) * env
    return out / (np.sqrt(np.mean(out**2)) + 1e-12) * level


def firework(seconds, level, rng):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    boom = ss.sosfilt(ss.butter(4, 120, "lowpass", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-t / 0.6) * 8
    crackle = np.zeros(n)
    for s in rng.integers(int(0.3 * SR), n - 200, 60):
        crackle[s:s + 200] += rng.standard_normal(200) * np.exp(-np.arange(200) / 30)
    out = boom + ss.sosfilt(ss.butter(2, 2000, "highpass", fs=SR, output="sos"), crackle) * 0.5
    return out / (np.sqrt(np.mean(out**2)) + 1e-12) * level


def confetti(seconds, level, rng):
    n = int(seconds * SR)
    out = np.zeros(n)
    out[:60] = rng.standard_normal(60) * 30
    rustle = ss.sosfilt(ss.butter(2, 3000, "highpass", fs=SR, output="sos"), rng.standard_normal(n)) * np.exp(-np.arange(n) / (0.5 * SR))
    out += rustle
    return out / (np.sqrt(np.mean(out**2)) + 1e-12) * level


def place(total_seconds, pieces):
    """Mix (start, signal) pieces into a stereo track of total_seconds."""
    out = np.zeros(int(total_seconds * SR))
    for start, sig in pieces:
        a = int(start * SR)
        out[a:a + sig.size] += sig[: out.size - a]
    return np.stack([out, out]).astype(np.float32)
