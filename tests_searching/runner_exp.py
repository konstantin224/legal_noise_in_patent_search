import subprocess
import sys
from pathlib import Path

main_file = "main_bge.py"

PYTHON = sys.executable  # гарантированно берёт текущий интерпретатор
SCRIPTS = [
    [PYTHON, f"tests_searching\{main_file}", "autoencoder_relu"],
    [PYTHON, f"tests_searching\{main_file}", "autoencoder_silu"],
    [PYTHON, f"tests_searching\{main_file}", "autoencoder_tanh"],
  
]

def run_sequential():
    for i, cmd in enumerate(SCRIPTS, 1):
        print(f"\n🚀 [{i}/{len(SCRIPTS)}] Запуск: {' '.join(cmd[1:])}")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Успешно завершён.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка в скрипте {cmd[-1]}. Код выхода: {e.returncode}")
            print("💡 Добавьте '--continue-on-error' в аргументы, если нужно идти дальше")
            sys.exit(1)
    print("\n🎉 Все скрипты выполнены успешно!")

if __name__ == "__main__":
    run_sequential()