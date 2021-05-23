import sys
import subprocess
from contextlib import redirect_stdout

for r in range(5):
  # cell [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[271] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[271,271] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271,271]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[271,271,271] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271,271,271]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[542] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[542,542] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542,542]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[542,542,542] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542,542,542]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[1084] --maxConsecutiveNonImprovements=5',shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[1084,1084] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084,1084]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[1084,1084,1084] --maxConsecutiveNonImprovements=5', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084,1084,1084]', shell=True)

  with open(str(r) + 'encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084, 1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")