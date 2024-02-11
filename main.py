import math
import cmath
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Uklad:
    def __init__(self, szybka_transformata):
        self.Czytaj()

        if self.wymiarowosc == 1:
            self.C = [(0j) for _ in range(self.wymiar_N)]
            self.C_modulo = [0 for _ in range(self.wymiar_N)]
            
            self.kroki_N = list(range(self.wymiar_N))

        if self.wymiarowosc == 2:
            self.C = np.zeros((self.wymiar_N, self.wymiar_M), dtype=complex)
            self.C_modulo = np.zeros((self.wymiar_N, self.wymiar_M), dtype=complex)

            self.kroki_N = list(range(self.wymiar_N))
            self.kroki_M = list(range(self.wymiar_M))

        self.Main(szybka_transformata)

    ############################# MAIN #############################

    def Main(self, szybka_transformata):
        if self.wymiarowosc == 1:
            if szybka_transformata == False: # 1D DFT
                self.DFT_1D()
                self.IDFT_1D()
                self.Wykresy("DFT")
            if szybka_transformata == True: # 1D FFT
                self.FFT_1D()
                self.IFFT_1D()
                self.Wykresy("FFT")
        if self.wymiarowosc == 2:
            if szybka_transformata == True: # 2D FFT
                self.FFT_2D()
                self.Wykresy("FFT")
            if szybka_transformata == False: # 2D DFT
                print("Nie ma DFT/IDFT w bibliotece")


    ####################### DFT DLA 1 WYMIARU #######################
    def DFT_1D(self):
        N = self.wymiar_N
        operacje_dominujace = 0

        for n in range(N):
            self.C[n] = 0
            for i in range(N):
                self.C[n] += self.Fi[i] * cmath.exp(-1j * math.pi * 2 * i * n / N)
                operacje_dominujace += 1

        self.C = [round(c.real, 4) + round(c.imag, 4) * 1j for c in self.C]
        self.C_modulo = [(math.sqrt(c.real**2 + c.imag**2)) for c in self.C]
        self.UsunPeaki()
        print("\033[91mPo DFT: {} operacji dominujących\033[0m".format(operacje_dominujace))
        print()

    ###################### IDFT DLA 1 WYMIARU #######################
    def IDFT_1D(self):
        N = self.wymiar_N
        operacje_dominujace = 0

        for n in range(N):
            for i in range(N):
                self.Fi_after[n] += self.C[i] * cmath.exp(1j * math.pi * 2 * i * n / N)
                operacje_dominujace += 1
            self.Fi_after[n] /= N
            self.Fi_after[n] = round(self.Fi_after[n].real, 4) + round(self.Fi_after[n].imag, 4) * 1j

        print("\033[91mPo IDFT: {} operacji dominujących\033[0m".format(operacje_dominujace))
        print()

    ####################### FFT DLA 1 WYMIARU #######################
    def FFT_1D(self):
        operacje_dominujace = 0

        def fft_in(x):
            nonlocal operacje_dominujace
            N = len(x)
            if N <= 1:
                return x
            even = fft_in(x[0::2])
            odd = fft_in(x[1::2])
            T = [0 for _ in range(N // 2)]
            for i in range(N // 2):
                T[i] = cmath.exp(-2j * math.pi * i / N) * odd[i]
                operacje_dominujace += 2

            return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]
        
        self.C = fft_in(self.Fi)
        self.C = [round(c.real, 4) + round(c.imag, 4) * 1j for c in self.C]
        self.C_modulo = [(math.sqrt(c.real**2 + c.imag**2)) for c in self.C]
        self.UsunPeaki()
        print("\033[91mPo FFT: {} operacji dominujących\033[0m".format(operacje_dominujace))
        print()
    
    ###################### IFFT DLA 1 WYMIARU #######################
    def IFFT_1D(self):
        operacje_dominujace = 0

        def ifft_in(x):
            nonlocal operacje_dominujace
            N = len(x)
            if N <= 1:
                return x
            even = ifft_in(x[0::2])
            odd = ifft_in(x[1::2])

            T = [0 for _ in range(N // 2)]
            for i in range(N // 2):
                T[i] = cmath.exp(2j * math.pi * i / N) * odd[i]
                operacje_dominujace += 2
            
            return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]
        
        self.Fi_after = ifft_in(self.C)
        self.Fi_after = [c / len(self.C) for c in self.Fi_after]
        self.Fi_after = [round(c.real, 4) + round(c.imag, 4) * 1j for c in self.Fi_after]

        print("\033[91mPo IFFT: {} operacji dominujących\033[0m".format(operacje_dominujace))
        print()

    ############################# 1D USUWANIE PEAKÓW #############################

    def UsunPeaki(self):
        self.C_bezpeakow = []
        self.kroki_bezpeakow = []

        for i in range(0, len(self.C_modulo)):
            if self.C_modulo[i] < 10:
                if self.C_modulo[i] == 0:
                    self.C_modulo[i] = 0.000000000001
                self.kroki_bezpeakow.append(self.kroki_N[i])
                self.C_bezpeakow.append(math.log(self.C_modulo[i]))
        

    ##################### FFT DLA 2 WYMIARÓW #######################
    def FFT_2D(self):
        self.C = np.fft.fft2(self.Fi)
        self.C_modulo = np.abs(self.C)

    ##################### IFFT DLA 2 WYMIARÓW #######################
    def IFFT_2D(self):
        self.Fi = np.fft.ifft2(self.C)
        self.C = np.zeros((self.wymiar_N, self.wymiar_M), dtype=complex)
        self.C_modulo = np.abs(self.C)
    

    # pokazuje liczby f, C oraz mod(C)
    def PokaZLiczby(self):
        print("Fi = \n {} \n ".format(self.Fi))
        print("C = \n {} \n ".format(self.C))
        print("Modulo C = \n {} \n ".format(self.C_modulo))
        print()

    # Pokazuje wymiarowość, wymiary, oraz liczby (Arg1 czy liczb pokazac)
    def PokazDaneWejsciowe(self, czy_liczby = True):
        print("Wymiarowość = \n{}{}".format(self.wymiarowosc))
        print("N = {}".format(self.wymiar_N))
        print("M = {}".format(self.wymiar_M))
        if czy_liczby:
            print()
            print(self.Fi)

    # odczytuje dane wejściem standardowym
    def Czytaj(self):

        print("\033[91mPodaj dane wejsciowe:\033[0m")

        self.wymiarowosc = int(input().strip())
    
        if self.wymiarowosc == 1:
            self.wymiar_N = int(input().strip())
            self.wymiar_M = None
        elif self.wymiarowosc == 2:
            self.wymiar_N, self.wymiar_M = map(int, input().strip().split())
        else:
            raise ValueError("Niewłaściwa wartość wymiarowości")

        self.Fi = [] 
        self.Fi_after = []
        if self.wymiarowosc == 1:
            for _ in range(self.wymiar_N):
                line = input().strip()
                row = complex(line)
                self.Fi.append(row)
                self.Fi_after.append(0)
        elif self.wymiarowosc == 2:
            self.Fi = np.zeros((self.wymiar_N, self.wymiar_M), dtype=complex)
            self.Fi_after = np.zeros((self.wymiar_N, self.wymiar_M), dtype=complex)
            for i in range(self.wymiar_N):
                line = input().strip().split()
                row = [complex(num) for num in line]
                self.Fi[i, :] = row

    ############################# WYKRESY #############################

    def Wykresy(self, transformata):
        if self.wymiarowosc == 1:
            plt.figure(figsize=(12, 9))

            # Wykres Fi przed transformacją
            plt.subplot(4, 1, 1)
            plt.scatter(self.kroki_N, self.Fi, marker='o', s=10, linestyle='-', color='b', label='Liczby F')
            plt.title("Sygnał wejściowy przed transformatą {}".format(transformata))
            plt.xlabel('Numer harmonicznej')
            plt.ylabel('Dane Fi')
            plt.grid(False)

            # Wykres C
            plt.subplot(4, 1, 2)
            plt.scatter(self.kroki_N, self.C_modulo, marker='o', s=10, linestyle='-', color='r', label='Liczby C')
            plt.title("Sygnał wyjściowy po transformacie {}".format(transformata))
            plt.xlabel('Numer harmonicznej')
            plt.ylabel('Dane C')
            plt.grid(False)

            # Wykres C bez peaków
            plt.subplot(4, 1, 3)
            plt.scatter(self.kroki_bezpeakow, self.C_bezpeakow, marker='o', s=10, label='C_bezpeakow')
            plt.xlabel('Numer harmonicznej')
            plt.ylabel('Dane log(C)')
            plt.grid(False)

            # Linia trendu
            trend_line_coefficients = np.polyfit(self.kroki_bezpeakow, self.C_bezpeakow, 1)
            trend_line = np.polyval(trend_line_coefficients, self.kroki_bezpeakow)
            plt.plot(self.kroki_bezpeakow, trend_line, color='orange', label='Linia trendu')

            # Wzór linii trendu
            wzor_linii_trendu = f'{trend_line_coefficients[0]:.4f}x + {trend_line_coefficients[1]:.4f}'
            
            plt.title("Bezpeakowy Zlinearyzowany Sygnał Wyjściowy {} - Zmodulowany. \n Wzór linii trendu = {}".format(transformata, wzor_linii_trendu))


            # Wykres Fi po transformacji
            plt.subplot(4, 1, 4)
            plt.scatter(self.kroki_N, self.Fi_after, marker='o', s=10, linestyle='-', color='g', label='Liczby F')
            plt.title("Sygnał wyjściowy po transformacie {}".format(str("I" + transformata)))
            plt.xlabel('Numer harmonicznej')
            plt.ylabel('Dane Fi')
            plt.grid(False)

            plt.tight_layout()
            plt.show()

        if self.wymiarowosc == 2:
            plt.figure(figsize=(12, 10))

            # Wykres 2D Fi przed transformacją
            plt.subplot(2, 2, 1)
            plt.imshow(np.abs(self.Fi), cmap='viridis', origin='lower', extent=(0, self.wymiar_N, 0, self.wymiar_M), aspect='auto')
            plt.xlabel('Numer harmonicznej N')
            plt.ylabel('Numer harmonicznej M')
            plt.colorbar(label='Dane F')
            plt.title('Sygnał wejściowy przed transformatą {}'.format(transformata))

            # Wykres 2D C
            plt.subplot(2, 2, 2)
            plt.imshow(np.abs(self.C_modulo), cmap='viridis', origin='lower', extent=(0, self.wymiar_N, 0, self.wymiar_M), aspect='auto')
            plt.xlabel('Numer harmonicznej N')
            plt.ylabel('Numer harmonicznej M')
            plt.colorbar(label='Dane C')
            plt.title('Sygnał wyjściowy po transformacie {}'.format(transformata))

            # Dodanie wykresu 3D Liczb F
            ax = plt.subplot(2, 2, 3, projection='3d')
            ax.plot_surface(self.kroki_N, self.kroki_M, np.abs(self.Fi), cmap='viridis')
            ax.set_xlabel('Numer harmonicznej N')
            ax.set_ylabel('Numer harmonicznej M')
            ax.set_zlabel('Dane Fi')
            plt.title('Sygnał wejściowy przed transformatą {}'.format(transformata))

            # Dodanie wykresu 3D Modułów Liczb C
            ax = plt.subplot(2, 2, 4, projection='3d')
            ax.plot_surface(self.kroki_N, self.kroki_M, np.abs(self.C_modulo), cmap='viridis')
            ax.set_xlabel('Numer harmonicznej N')
            ax.set_ylabel('Numer harmonicznej M')
            ax.set_zlabel('Dane C')
            plt.title('Sygnał wyjściowy po transformacie {}'.format(transformata))

            plt.tight_layout()
            plt.show()







def main():

    # Uklad(Arg1) 
    #   Arg1: True = FFT, False = DFT    

    #uklad1_Slow = Uklad(False)
    uklad1_Fast = Uklad(True)

    #uklad2_Slow = Uklad(False)
    uklad2_Fast = Uklad(True)

    uklad3_Fast = Uklad(True)


if __name__ == "__main__":
    main()
