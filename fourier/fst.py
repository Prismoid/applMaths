import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# サンプル関数の定義
def user_function(t):
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)

# 普通のフーリエ変換を実行する関数
def fourier_series_expansion(func, freq, t_min=0, t_max=1):
    # フーリエ級数の実部と虚部を計算するための積分関数
    real_part = lambda t: func(t) * np.cos(2 * np.pi * freq * t)
    imag_part = lambda t: func(t) * np.sin(2 * np.pi * freq * t)

    # 数値積分を使って実部と虚部を計算
    real_integral = quad(real_part, t_min, t_max)[0]
    imag_integral = quad(imag_part, t_min, t_max)[0]

    # フーリエ級数値を返す
    return real_integral - 1j * imag_integral

# 周波数範囲と分割数の設定
freqs = np.arange(-10, 11) # [-10 ~ 10]までの周波数成分を計算

# フーリエ変換の結果を格納するリスト
fourier_results = []

# 各周波数に対してフーリエ級数を計算
for freq in freqs:
    result = fourier_series_expansion(user_function, freq)
    fourier_results.append(result)

# フーリエ変換結果をプロット
plt.figure(figsize=(12, 6))

# 元の関数をプロット
t_vals = np.linspace(0, 1, 1000)
plt.subplot(1, 2, 1)
plt.plot(t_vals, user_function(t_vals))
plt.title("Original Function")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# フーリエ級数展開の結果をプロット
plt.subplot(1, 2, 2)
for freq, fourier_result in zip(freqs, fourier_results):
    if np.abs(fourier_result) > 0.0001: 
        plt.arrow(freq, 0, 0, np.abs(fourier_result), head_width=0.2, head_length=0.05, fc='blue', ec='blue', length_includes_head=True)

plt.plot()
plt.xlim(-12, 12)  # x 軸の範囲を設定
plt.ylim(-1, 1)    # y 軸の範囲を設定（y 軸方向の矢印は 0 なので小さい範囲で設定）
plt.axhline(0, color='black', lw=1)  # x 軸に線を引く
plt.title("Fourier Series Expansion (Magnitude)")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
