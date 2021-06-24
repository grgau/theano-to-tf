import sys
import subprocess
from contextlib import redirect_stdout

for r in range(5):
  # cell [391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[391] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [391, 391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[391,391] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391,391]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[391, 391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [391, 391, 391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[391,391,391] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391,391,391]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[391, 391, 391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[783] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [783, 783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[783,783] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783,783]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[783, 783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [783, 783, 783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[783,783,783] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783,783,783]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[783, 783, 783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[1566] --maxConsecutiveNonImprovements=5',shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1566, 1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[1566,1566] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566,1566]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1566, 1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1566, 1566, 1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[1566,1566,1566] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566,1566,1566]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1566, 1566, 1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=5',shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[3133]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [3133, 3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[3133,3133] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133,3133]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[3133, 3133]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [3133, 3133, 3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-lstm-model --state=cell --hiddenDimSize=[3133,3133,3133] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133,3133,3133]', shell=True)

  with open(str(r) + 'lstm-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[3133, 3133, 3133]")
      print(stdout.decode().split("\n"))
      print("\n")