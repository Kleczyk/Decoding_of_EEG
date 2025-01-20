import timesfm
import pandas as pd

# Inicjalizacja modelu TimesFM
    context_len=512,
    horizon_len=128,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="cpu"  # lub "gpu" jeśli dostępne
)

# Załadowanie wstępnie wytrenowanych wag
model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m-pytorch")

# Przygotowanie danych wejściowych jako DataFrame
# Zakładając, że masz dane w formacie DataFrame z kolumnami 'unique_id', 'ds' (daty) i 'y' (wartości)
df = pd.read_csv("path_to_your_time_series_data.csv")

# Prognozowanie
forecast_df = model.forecast_on_df(
    inputs=df,
    freq="D",  # Częstotliwość danych: 'D' dla dziennych, 'H' dla godzinnych itp.
    value_name="y",
    num_jobs=-1
)

# Wyświetlenie prognoz
print(forecast_df)
