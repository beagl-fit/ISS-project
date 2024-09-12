import cmath
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import tf2zpk, spectrogram, iirnotch, freqz, filtfilt
from scipy.io import wavfile

plt.style.use('bmh')

#################### Příklad 1 ####################
print("1)")
# Načtení wavky -- jupyter ntb
fs, data = wavfile.read('xnovak2r.wav')
data = data / 2 ** 15

# Vypsání údajů
print("Délka signálu:")
print("\tve vzorcích:", data.size)
print("\tv sekundách:", data.size / fs, "sekund")
print("Minimální hodnota:", data.min())
print("Maximální hodnota:", data.max())

# Tvorba grafu
t = np.arange(data.size) / fs
plt.plot(t, data)
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Zvukový signál')
plt.tight_layout()
plt.savefig('plot_1.pdf')
plt.show()

print()
#################### Příklad 2 ####################
# Ustřednění signálu
data = data - np.mean(data)
data = data / data.max()

# Rozdělení na rámce o velikosti 1024 vzorků s překrytím 512 vzorků
# 1. řádek matice
data_matrix = np.array([data[0:1024]])
i = 512

while i < data.size:
    data_segment = np.array(data[i:i + 1024])
    # Doplnění nulami do 1024 vzorků
    missing = 1024 - len(data_segment)
    if missing != 0:
        data_segment = np.append(data_segment, np.zeros(missing))
    # Přidání řádku do matice
    data_matrix = np.append(data_matrix, [data_segment], axis=0)
    i = i + 512

# Transpozice matice -- sloupce <=> řádky
data_matrix = np.transpose(data_matrix)

# Tvorba grafu
t = np.arange(1024)
# Vybrán 14. sloupec matice
plt.plot(t, data_matrix[:, 13])  # 3-5 ?
plt.gca().set_xlabel('$t[s]$')
plt.gca().set_title('Rámec s periodickým charakterem')
plt.tight_layout()

plt.xticks(np.arange(0, 1025, 128))
plt.yticks(np.arange(-0.5, 0.8, 0.1))
plt.savefig('plot_2.pdf')
plt.show()

#################### Příklad 3 ####################
f_max = fs / 2
print("DFT bude chvíli trvat\n")
# Moje DFT
i = 1
base_matrix = [np.ones((1024,), dtype=complex)]
while i < f_max:
    n = 0
    base_matrix_line = []
    while n < 1024:
        num = -2j * np.pi / 1024 * n * i
        if num == 0 + 0j:
            num = complex(1, 0)
            base_matrix_line.append(num)
            n += 1
            continue
        num = cmath.exp(num)
        base_matrix_line.append(num)
        n += 1
    base_matrix = np.append(base_matrix, [base_matrix_line], axis=0)
    i += 1

DFT = np.dot(data_matrix[:, 13], base_matrix.T)

# Numpy FFT
numpy_FFT = np.fft.fft(data_matrix[:, 13])
y = np.arange(0, f_max / 2, f_max / 1024)

# Tvorba grafů
plt.plot(y, abs(numpy_FFT[0:512]), label='numpy FFT')
plt.plot(y, abs(DFT[0:512]), label='Moje DFT')
plt.gca().set_ylabel('Magnitude')
plt.gca().set_xlabel('$f[Hz]$')
plt.gca().set_title('DFT & FFT')
plt.tight_layout()
plt.legend()
plt.savefig('plot_3.pdf')
plt.show()

#################### Příklad 4 ####################
# Tvorba spektrogramu -- jupyter ntb
f, t, sgr = spectrogram(data, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr + 1e-20)

plt.figure(figsize=(9, 3))
plt.pcolormesh(t, f, sgr_log, shading='auto')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.yticks(np.arange(0, 9000, step=1000))
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()

plt.savefig('plot_4.pdf')
plt.show()

#################### Příklad 5 ####################
# Ručně nalezené frekvence
# 2600
f1 = "4*650 Hz"
f2 = "2*" + f1
f3 = "3*" + f1
f4 = "4*" + f1

print("5)")
print("f1 = ", f1)
print("f2 = ", f2)
print("f3 = ", f3)
print("f4 = ", f4)
print()

f1 = 4 * 650
f2 = 2 * f1
f3 = 3 * f1
f4 = 4 * f1

#################### Příklad 6 ####################
# Generování kosinusovek pro nalezené freqence
arr = []
for i in range(data.size):
    arr.append(i / data.size)

cos1 = np.cos(2 * np.pi * f1 * np.array(arr))
cos2 = np.cos(2 * np.pi * f2 * np.array(arr))
cos3 = np.cos(2 * np.pi * f3 * np.array(arr))
cos4 = np.cos(2 * np.pi * f4 * np.array(arr))
cos5 = cos1 + cos2 + cos3 + cos4

wavfile.write("audio/4cos.wav", fs, cos5.astype(np.float32))

# Tvorba spektogramu pro vygenerováné kosinusovky -- jupyter ntb
f, t, sgr = spectrogram(cos5, fs, nperseg=1024, noverlap=512)
sgr_log = 10 * np.log10(sgr + 1e-20)

plt.pcolormesh(t, f, sgr_log, shading='auto')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
plt.yticks(np.arange(0, 9000, step=1000))
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
plt.tight_layout()

plt.savefig('plot_6.pdf')
plt.show()

#################### Příklad 7 ####################
# Výroba filtrů
b0, a0 = iirnotch(650, 60, fs)
b1, a1 = iirnotch(2 * 650, 60, fs)
b2, a2 = iirnotch(3 * 650, 60, fs)
b3, a3 = iirnotch(4 * 650, 60, fs)

freq0, h0 = freqz(b0, a0, fs=fs)
freq1, h1 = freqz(b1, a1, fs=fs)
freq2, h2 = freqz(b2, a2, fs=fs)
freq3, h3 = freqz(b3, a3, fs=fs)

# Impulsní odezvy filtrů
response0 = filtfilt(b0, a0, np.append([1], np.zeros(400, dtype=int)))
response1 = filtfilt(b1, a1, np.append([1], np.zeros(400, dtype=int)))
response2 = filtfilt(b2, a2, np.append([1], np.zeros(400, dtype=int)))
response3 = filtfilt(b3, a3, np.append([1], np.zeros(400, dtype=int)))

plt.plot(np.linspace(0, 1, 401), response0)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Amplituda')
plt.savefig('plot_7.1.pdf')
plt.show()

plt.plot(np.linspace(0, 1, 401), response1)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Amplituda')
plt.savefig('plot_7.2.pdf')
plt.show()

plt.plot(np.linspace(0, 1, 401), response2)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Amplituda')
plt.savefig('plot_7.3.pdf')
plt.show()

plt.plot(np.linspace(0, 1, 401), response3)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Amplituda')
plt.savefig('plot_7.4.pdf')
plt.show()

#################### Příklad 8 ####################
angle = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(angle), np.sin(angle))

# nuly, poly -- jupyter ntb
z0, p0, k0 = tf2zpk(b0, a0)
z1, p1, k1 = tf2zpk(b1, a1)
z2, p2, k2 = tf2zpk(b2, a2)
z3, p3, k3 = tf2zpk(b3, a3)

plt.scatter(np.real(z0), np.imag(z0), marker='o', facecolors='none', edgecolors='r', label='nuly')
plt.scatter(np.real(p0), np.imag(p0), marker='x', color='g', label='póly')
plt.scatter(np.real(z1), np.imag(z1), marker='o', facecolors='none', edgecolors='r')
plt.scatter(np.real(p1), np.imag(p1), marker='x', color='g')
plt.scatter(np.real(z2), np.imag(z2), marker='o', facecolors='none', edgecolors='r')
plt.scatter(np.real(p2), np.imag(p2), marker='x', color='g')
plt.scatter(np.real(z3), np.imag(z3), marker='o', facecolors='none', edgecolors='r')
plt.scatter(np.real(p3), np.imag(p3), marker='x', color='g')

plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig('plot_8.pdf')
plt.show()

#################### Příklad 9 ####################
# Tvorba grafů frekvenční charakteristiky
### Filtr 0
plt.plot(freq0, 20 * np.log10(abs(h0)), color='green')
plt.xlim(0, 5300)
plt.xticks(np.arange(0, 5300, step=650))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_ylabel('Amplituda [dB]')
plt.grid(True)
plt.title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
plt.savefig('plot_9.1.pdf')
plt.show()

plt.plot(freq0, np.unwrap(np.angle(h0)) * 180 / np.pi, color='green')
plt.gca().set_xlabel("Frekvence [Hz]")
plt.xlim(0, 2000)
plt.yticks(np.arange(-90, 91, step=30))
plt.ylim(-90, 90)
plt.grid(True)
plt.title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')
plt.savefig('plot_9.2.pdf')
plt.show()

### Filtr 1
plt.plot(freq1, 20 * np.log10(abs(h1)), color='red')
plt.xlim(0, 5300)
plt.xticks(np.arange(0, 5300, step=650))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_ylabel('Amplituda [dB]')
plt.grid(True)
plt.title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
plt.savefig('plot_9.3.pdf')
plt.show()

plt.plot(freq1, np.unwrap(np.angle(h1)) * 180 / np.pi, color='red')
plt.gca().set_xlabel("Frekvence [Hz]")
plt.xlim(0, 2750)
plt.yticks(np.arange(-90, 91, step=30))
plt.ylim(-90, 90)
plt.grid(True)
plt.title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')
plt.savefig('plot_9.4.pdf')
plt.show()

### Filtr 2
plt.plot(freq2, 20 * np.log10(abs(h2)), color='gold')
plt.xlim(0, 5300)
plt.xticks(np.arange(0, 5300, step=650))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_ylabel('Amplituda [dB]')
plt.grid(True)
plt.title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
plt.savefig('plot_9.5.pdf')
plt.show()

plt.plot(freq2, np.unwrap(np.angle(h2)) * 180 / np.pi, color='gold')
plt.gca().set_xlabel("Frekvence [Hz]")
plt.xlim(0, 4000)
plt.yticks(np.arange(-90, 91, step=30))
plt.ylim(-90, 90)
plt.grid(True)
plt.title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')
plt.savefig('plot_9.6.pdf')
plt.show()

### Filtr 3
plt.plot(freq3, 20 * np.log10(abs(h3)))
plt.xlim(0, 5300)
plt.xticks(np.arange(0, 5300, step=650))
plt.gca().set_xlabel('Frekvence [Hz]')
plt.gca().set_ylabel('Amplituda [dB]')
plt.grid(True)
plt.title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')
plt.savefig('plot_9.7.pdf')
plt.show()

plt.plot(freq3, np.unwrap(np.angle(h3)) * 180 / np.pi)
plt.gca().set_xlabel("Frekvence [Hz]")
plt.xlim(0, 6000)
plt.yticks(np.arange(-90, 91, step=30))
plt.ylim(-90, 90)
plt.grid(True)
plt.title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')
plt.savefig('plot_9.8.pdf')
plt.show()

#################### Příklad 10 ####################
outputSignal0 = filtfilt(b0, a0, cos5)
outputSignal1 = filtfilt(b1, a1, outputSignal0)
outputSignal2 = filtfilt(b2, a2, outputSignal1)
outputSignal3 = filtfilt(b3, a3, outputSignal2)

wavfile.write("audio/clean_bandstop.wav", fs, outputSignal3.astype(np.float32))

quit()
