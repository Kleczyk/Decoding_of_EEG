from ray.tune.schedulers import FIFOScheduler
import time


class DynamicFIFOScheduler(FIFOScheduler):
    def __init__(self, max_concurrent_default=2, memory_threshold=80, check_interval=5):
        """
        Dynamiczny scheduler oparty na FIFO.

        Args:
            max_concurrent_default (int): Domyślna maksymalna liczba równoległych eksperymentów.
            memory_threshold (int): Próg pamięci w procentach.
            check_interval (int): Czas w sekundach między sprawdzaniem pamięci.
        """
        super().__init__(max_concurrent=max_concurrent_default)
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval

    def _adjust_max_concurrent(self):
        memory_usage = get_memory_usage()
        if memory_usage > self.memory_threshold:
            self.max_concurrent = max(1, self.max_concurrent - 1)
        elif memory_usage < self.memory_threshold - 10:  # Dodaj trochę marginesu
            self.max_concurrent += 1

    def on_trial_result(self, trial_runner, trial, result):
        self._adjust_max_concurrent()
        super().on_trial_result(trial_runner, trial, result)

    def run(self):
        while True:
            self._adjust_max_concurrent()
            time.sleep(self.check_interval)
