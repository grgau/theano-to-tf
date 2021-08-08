import sys
import subprocess
from contextlib import redirect_stdout

for r in range(5):
  # cell [391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[391] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [391, 391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[391,391] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391,391] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[391, 391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [391, 391, 391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[391,391,391] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391,391,391] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[391, 391, 391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[783] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [783, 783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[783,783] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783,783] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[783, 783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [783, 783, 783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[783,783,783] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783,783,783] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[783, 783, 783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[1566] --maxConsecutiveNonImprovements=10',shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1566, 1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[1566,1566] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566,1566] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1566, 1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1566, 1566, 1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[1566,1566,1566] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566,1566,1566] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1566, 1566, 1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10',shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[3133]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [3133, 3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133,3133] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133,3133] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[3133, 3133]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [3133, 3133, 3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133,3133,3133] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133,3133,3133] --runName=A_c-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[3133, 3133, 3133]")
      print(stdout.decode().split("\n"))
      print("\n")


  # hidden [391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[391] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [391, 391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[391,391] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391,391] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[391, 391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [391, 391, 391]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[391,391,391] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[391,391,391] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[391, 391, 391]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[783] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [783, 783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[783,783] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783,783] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[783, 783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [783, 783, 783]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[783,783,783] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[783,783,783] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[783, 783, 783]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[1566] --maxConsecutiveNonImprovements=10',shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [1566, 1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[1566,1566] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566,1566] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[1566, 1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [1566, 1566, 1566]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[1566,1566,1566] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1566,1566,1566] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[1566, 1566, 1566]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10',shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[3133]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [3133, 3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[3133,3133] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133,3133] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[3133, 3133]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [3133, 3133, 3133]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-incor/incor_90-10_3133" compiled_models/incor-encdec-model --state=hidden --hiddenDimSize=[3133,3133,3133] --maxConsecutiveNonImprovements=10', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-incor/incor_90-10_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133,3133,3133] --runName=A_h-INCOR_', shell=True)

  with open(str(r) + '_encdec-incor-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[3133, 3133, 3133]")
      print(stdout.decode().split("\n"))
      print("\n")