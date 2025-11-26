import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import freqz, firwin, kaiserord, firwin2, remez

# 1. Definicja parametrów zadania
Fs = 5000  # Częstotliwość próbkowania [Hz]
F_pass = 1000  # Częstotliwość graniczna pasma przepustowego [Hz]
F_stop = 1500  # Częstotliwość graniczna pasma zaporowego [Hz]
Rp = 1  # Tętnienie w pasmie przepustowym [dB]
Rs = 30  # Tłumienie w pasmie zaporowym [dB]

# Normalizacja częstotliwości (do pi rad/próbkę)
wp = F_pass / (Fs / 2)  # 0.4
ws = F_stop / (Fs / 2)  # 0.6

# 2. Obliczenia pomocnicze dla porównania
# Wymagane krotności (amplitudy na skali liniowej)
A_pass = 10**(-Rp / 20)  # Min. amplituda w pasmie przepustowym (~0.891)
A_stop = 10**(-Rs / 20)  # Max. amplituda w pasmie zaporowym (~0.0316)

# b) Projektowanie filtru metodą Okna (Hanning)
# Przyjmujemy długość M=47 (zgodnie z oszacowaniem Kaisera dla Rs=30dB)
M_window = 47
# Częstotliwość odcięcia (pomiędzy wp i ws)
Fc_window = (F_pass + F_stop) / 2 / (Fs / 2) # 0.5
h_window = firwin(M_window, Fc_window, window='hann')

# c) Projektowanie filtru Metodą Okna Kaisera
# Różnica w znormalizowanych pulsacjach (szerokość pasma przejściowego)
delta_w = ws - wp

# Obliczenie długości M i parametru beta dla okna Kaisera
# Tłumienie Rs musi być w skali liniowej (nie ma potrzeby konwersji,
# ponieważ kaiserord operuje na wartościach dB i znormalizowanych częstotliwościach)
M_kaiser, beta_kaiser = kaiserord(Rs, delta_w)
M_kaiser += 1  # Długość M musi być nieparzysta dla idealnego FIR

# Częstotliwość odcięcia
Fc_kaiser = (F_pass + F_stop) / 2 / (Fs / 2) # 0.5

# Projektowanie filtru
h_kaiser = firwin(M_kaiser, Fc_kaiser, window=('kaiser', beta_kaiser))

# d) Projektowanie filtru Algorytmem Remez (Parks-McClellan)
# Przyjmujemy długość M oszacowaną w punkcie c (Kaiser)
M_remez = M_kaiser

# Definicja pasm i pożądanej amplitudy
# freqs: [0, wp, ws, 1] - granice pasm (0 do Fs/2)
# amps: [1, 1, 0, 0] - pożądane wzmocnienie (1 w pass, 0 w stop)
# Przy Remez, bierzemy tylko [0, wp] oraz [ws, 1]
bands_remez = [0, wp, ws, 1]
desired_amps_remez = [1, 0] # 1 dla [0, wp], 0 dla [ws, 1]

# Błąd (Ripple) (nie jest wymagane wprost, ale można użyć weights)
# Remez minimalizuje błąd, więc wystarczy podać M i pasma.
h_remez = remez(M_remez, bands_remez, desired_amps_remez, fs=Fs/2)

# e) Projektowanie filtru metodą firls (Least-Squares)
# Przyjmujemy długość M oszacowaną w punkcie c (Kaiser)
M_firls = M_kaiser

# Definicja pasm i pożądanej amplitudy
bands_firls = [0, F_pass, F_stop, Fs/2]
desired_amps_firls = [1, 1, 0, 0] # Wartości pożądane na granicy pasm

# Projektowanie filtru
# Uwaga: firwin2 to odpowiednik firls z Matlaba
h_firls = firwin2(M_firls, bands_firls, desired_amps_firls, fs=Fs)

# 3. Obliczenie charakterystyk częstotliwościowych (magnitude response)
# Generujemy punkty dla wykresu (N=2048)
N_fft = 2048
w, H_window = freqz(h_window, 1, N_fft, fs=Fs)
w, H_kaiser = freqz(h_kaiser, 1, N_fft, fs=Fs)
w, H_remez = freqz(h_remez, 1, N_fft, fs=Fs)
w, H_firls = freqz(h_firls, 1, N_fft, fs=Fs)

# Konwersja do dB
H_window_dB = 20 * np.log10(np.abs(H_window))
H_kaiser_dB = 20 * np.log10(np.abs(H_kaiser))
H_remez_dB = 20 * np.log10(np.abs(H_remez))
H_firls_dB = 20 * np.log10(np.abs(H_firls))

# 4. Generowanie wykresu

plt.figure(figsize=(12, 6))
plt.plot(w, H_window_dB, label=f'Window (Hanning, M={M_window})')
plt.plot(w, H_kaiser_dB, label=f'Kaiser (M={M_kaiser}, β={beta_kaiser:.2f})')
plt.plot(w, H_remez_dB, label=f'Remez (Parks-McClellan, M={M_remez})')
plt.plot(w, H_firls_dB, label=f'FIRLS (Least Squares, M={M_firls})')

# Dodanie linii wymagań (Requirement lines)
# Pasmo przepustowe
plt.plot([0, F_pass], [Rp, Rp], 'k--')
plt.plot([0, F_pass], [-Rp, -Rp], 'k--', label='_nolegend_')
# Pasmo zaporowe
plt.plot([F_stop, Fs/2], [-Rs, -Rs], 'r--', linewidth=2, label='Wymagane Rs')
plt.plot([F_pass, F_stop], [0, 0], 'g--', label='_nolegend_') # Pasmo przejściowe

# Ustawienia wykresu
plt.title(f'Charakterystyka Amplitudowa Filtrów FIR (Fs={Fs/1000} kHz)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.xlim(0, Fs/2)
plt.ylim(-60, 5)
plt.grid(which='both', axis='both', linestyle='--')
plt.legend()
plt.show()

# Wypisanie długości filtrów
print("\n### Podsumowanie Długości Filtrów (M) ###")
print(f"Długość M (Window Hanning): {M_window}")
print(f"Długość M (Kaiser): {M_kaiser}")
print(f"Długość M (Remez): {M_remez} (Wybór projektanta)")
print(f"Długość M (FIRLS): {M_firls} (Wybór projektanta)")