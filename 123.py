import matplotlib.pyplot as plt

# Przykładowe dane
x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 12, 9]

# Tworzenie wykresu liniowego
plt.plot(x, y)

# Dodawanie etykiet osi
plt.xlabel('Oś X')
plt.ylabel('Oś Y')

# Dodawanie tytułu wykresu
plt.title('Przykładowy wykres')

# Wyświetlanie legendy (opcjonalnie)
plt.legend(['Dane'])

# Wyświetlanie wykresu
plt.show()
