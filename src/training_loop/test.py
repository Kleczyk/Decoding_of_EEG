import ray
import torch
import os

# Inicjalizacja Ray
ray.init()
print("Ray został zainicjalizowany.")

# Wyświetlenie dostępnych zasobów
print("Dostępne zasoby:", ray.available_resources())


# Definicja zadania testowego na CPU
@ray.remote
def test_cpu_task():
    pid = os.getpid()
    return f"Zadanie CPU wykonane w procesie {pid}"


# Definicja zadania testowego na GPU
@ray.remote(num_gpus=1)
def test_gpu_task():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.rand(1000, 1000, device=device)
        return f"Zadanie GPU wykonane na urządzeniu: {torch.cuda.get_device_name(0)}"
    else:
        return "CUDA nie jest dostępna w PyTorch"


# Sprawdzenie dostępności GPU
num_gpus = int(ray.available_resources().get("GPU", 0))

# Uruchomienie wielu zadań na CPU
cpu_tasks = [test_cpu_task.remote() for _ in range(10)]
cpu_results = ray.get(cpu_tasks)
print("Wyniki zadań CPU:", cpu_results)

# Uruchomienie wielu zadań na GPU (jeśli dostępne)
if num_gpus > 0:
    gpu_tasks = [test_gpu_task.remote() for _ in range(num_gpus)]
    gpu_results = ray.get(gpu_tasks)
    print("Wyniki zadań GPU:", gpu_results)
else:
    print("GPU nie jest dostępne dla Ray.")

# Zakończenie działania Ray
ray.shutdown()
