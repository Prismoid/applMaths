import numpy as np
import matplotlib.pyplot as plt

# サンプル関数の定義
def user_function(t):
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)

# サンプル関数のフーリエ変換を実行する関数
def perform_fourier_transform(func, t_min=0, t_max=1, num_points=1000):
    # 時間軸を定義
    t = np.linspace(t_min, t_max, num_points)
    # 関数の値を取得
    y = func(t)

    # フーリエ変換を実行
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(num_points, (t_max - t_min) / num_points)

    # グラフをプロット
    plt.figure(figsize=(12, 6))

    # 元の関数をプロット
    plt.subplot(1, 2, 1)
    plt.plot(t, y)
    plt.title("Original Function")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # フーリエ変換結果をプロット
    plt.subplot(1, 2, 2)
    plt.xlim(-20, 20)
    plt.plot(xf, np.abs(yf))
    plt.title("Fourier Transform")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

# 実行例：サンプル関数に対してフーリエ変換を実行
perform_fourier_transform(user_function)
