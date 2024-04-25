import numpy as np
import matplotlib.pyplot as plt


def generate_validation_indices(data_length, num_of_val_samples, sequence_length):
    min_distance = sequence_length + 1  # Minimalna odległość pomiędzy indeksami
    available_indices = set(range(data_length))  # Tworzymy zbiór dostępnych indeksów

    val_indices = []
    for _ in range(num_of_val_samples):
        if len(available_indices) == 0:
            raise ValueError(
                "Nie można wygenerować więcej próbek z uwzględnieniem minimalnego dystansu"
            )

        chosen_index = int(np.random.choice(
            list(available_indices))
        )  # Losujemy z dostępnych indeksów
        val_indices.append(chosen_index)

        # Usuwamy indeksy w zakresie `sequence_length` w obie strony od wybranego indeksu
        indices_to_remove = set(
            range(
                max(0, chosen_index - (2 * sequence_length) - 3),
                min(data_length, chosen_index + (2 * sequence_length) + 3),
            )
        )
        available_indices.difference_update(
            indices_to_remove
        )  # Aktualizujemy zbiór dostępnych indeksów

    return val_indices


def generate_mask(data_length, val_i, sequence_length):
    # Ustal minimalną odległość pomiędzy indeksami
    min_distance = sequence_length + 1
    # Utwórz maskę początkową ze wszystkimi wartościami ustawionymi na True
    mask = np.ones(data_length, dtype=bool)

    # Iteruj przez każdy wybrany indeks walidacyjny
    for index in val_i:
        # Ustal zakres indeksów, które należy ustawić na False
        start = max(0, index - min_distance)
        end = min(data_length, index + min_distance + 1)

        # Ustaw odpowiednie wartości w masce na False
        mask[start:end] = False

    # Zwróć indeksy, gdzie maska jest True, czyli indeksy zbioru treningowego
    training_indices = list(np.where(mask)[
        0
    ])  # np.where(mask) zwraca tuple, [0] wyciąga array z indeksami
    return training_indices


def plot_val_indices(data_length, val_indices, sequence_length):
    # Ustal minimalną odległość pomiędzy indeksami
    min_distance = sequence_length + 1

    # Inicjalizacja figury
    plt.figure(figsize=(10, 2))
    plt.title("Rozkład indeksów walidacyjnych i ich zakresy")
    plt.xlabel("Indeksy danych")
    plt.ylabel("Wartość (dla wizualizacji)")

    # Rysowanie linii dla całej długości danych
    plt.plot([0, data_length - 1], [1, 1], label="Dane", color="blue")

    # Rysowanie punktów dla walidacyjnych indeksów
    for index in val_indices:
        plt.scatter([index], [1], color="red")  # punkt walidacyjny
        start = max(0, index - min_distance)
        end = min(data_length, index + min_distance)
        plt.axvspan(start, end, color="red", alpha=0.3)  # zakres wokół punktu)

    plt.legend(["Dane", "Indeksy walidacyjne i zakres"])
    plt.grid(True)
    plt.show()


def plot_train_val_indices(data_length, train_indices, val_indices, sequence_length):
    # Ustal minimalną odległość pomiędzy indeksami
    min_distance = sequence_length + 1

    # Inicjalizacja figury
    plt.figure(figsize=(10, 2))
    plt.title("Rozkład indeksów walidacyjnych i ich zakresy")
    plt.xlabel("Indeksy danych")
    plt.ylabel("Wartość (dla wizualizacji)")

    # Rysowanie linii dla całej długości danych
    plt.plot([0, data_length - 1], [1, 1], label="Dane", color="yellow")

    # Rysowanie punktów dla walidacyjnych indeksów
    for index in val_indices:
        plt.scatter([index], [1], color="red")  # punkt walidacyjny
        start = max(0, index - min_distance)
        end = min(data_length, index + min_distance)
        plt.axvspan(start, end, color="red", alpha=0.3)  # zakres wokół punktu)
    # if train_indices is not empty:
    for index in train_indices:
        plt.scatter([index], [1], color="blue")  # punkt walidacyjny
        start = index
        end = min(data_length, index + min_distance)
        plt.axvspan(start, end, color="blue", alpha=0.1)  # zakres wokół punktu)

    plt.legend(["Dane", "Indeksy walidacyjne i zakres"])
    plt.grid(True)
    plt.show()


# Przykładowe wywołanie funkcji
data_length = 5000
num_of_val_samples = 100
sequence_length = 10

val_indices = generate_validation_indices(
    data_length, num_of_val_samples, sequence_length
)
train_indices = generate_mask(data_length, val_indices, sequence_length)
plot_val_indices(data_length, val_indices, sequence_length)
plot_train_val_indices(data_length, train_indices, val_indices, sequence_length)

