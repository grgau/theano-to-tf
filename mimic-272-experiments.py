import sys
import subprocess
from contextlib import redirect_stdout

for r in range(5):
  # cell [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271,271] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271,271,271] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542,542] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542,542,542] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084,1084] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=cell --hiddenDimSize=[1084,1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084,1084,1084] --runName=A_c-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084, 1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")




  # hidden [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271,271] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[271,271,271] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542,542] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[542,542,542] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084,1084] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # hidden [1084, 1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "src/data-apehr-preprocessed/mimic_90-10_272" compiled_models/272-lstm-model --state=hidden --hiddenDimSize=[1084,1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "src/data-apehr-preprocessed/mimic_90-10_272"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[1084,1084,1084] --runName=A_h-MIMIC-272_', shell=True)

  with open(str(r) + '_lstm-mimic-272-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=hidden --hiddenDimSize=[1084, 1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")