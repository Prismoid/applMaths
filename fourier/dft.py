import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# サンプル関数の定義
def user_function(t):
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)

# 普通のフーリエ変換を実行する関数
def dft(func, T, N):
    # f(kT)をベクトルとして与える
    input_vec = np.arange(0, N) * T + 0j
    sample_vec = func(input_vec)
    # print(sample_vec)
    # 複素行列の作成
    bs_vec = np.arange(0, N)
    indices = np.outer(bs_vec, bs_vec)
    comp_exp_matrix = np.exp(-1j * (2 * np.pi) / N * indices)
    return np.dot(comp_exp_matrix, sample_vec)

# 周波数1/T, 1/2T, ..., 1/NTまでの成分に関する情報の足し合わせ
def idft(freq_vec, N):
    # 複素行列の生成
    bs_vec = np.arange(0, N)
    indices = np.outer(bs_vec, bs_vec)
    comp_exp_matrix = np.exp(+1j * (2 * np.pi) / N * indices)
    return 1.0/N * np.dot(comp_exp_matrix, freq_vec)


# サンプリング周期は1/8、サンプリング数は8、(本プログラムのDFT対象の関数の周期はNT = 1)
T, N = 1/8, 8

# DFTを実行
freq_vec = dft(user_function, T, N)
# print(freq_vec)

# IDFTを実行
# print( idft(freq_vec, N) )

# フーリエ変換結果をプロット
plt.figure(figsize=(12, 6))

# 元の関数をプロット
t_vals = np.linspace(0, 1, 1000)
plt.subplot(1, 2, 1)
plt.plot(t_vals, user_function(t_vals))
plt.title("Original Function")
plt.xlabel("Time")
plt.ylabel("Amplitude")

# DFTの結果をプロット(共役対称性に注意)
plt.subplot(1, 2, 2)
for i, freq in enumerate(freq_vec): 
    if np.abs(freq) > 0.0001: 
        plt.arrow(i, 0, 0, np.abs(freq), fc='blue', ec='blue', length_includes_head=True) # head_width=0.5, head_length=10.0

plt.plot()
plt.xlim(-0.5, N)  # x 軸の範囲を設定
# plt.ylim(-1, 1)    # y 軸の範囲を設定（y 軸方向の矢印は 0 なので小さい範囲で設定）
plt.axhline(0, color='black', lw=1)  # x 軸に線を引く
plt.title("DFT (Magnitude)")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
