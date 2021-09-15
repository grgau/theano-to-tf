import sys
import subprocess
from contextlib import redirect_stdout

for r in range(3):
  # cell [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + 'gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271,271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271,271,271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542,542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542,542,542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

# for r in range(3):
#   # cell [1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")

# for r in range(3):
#   # cell [1084, 1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084,1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084,1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084, 1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")

# for r in range(3):
#   # cell [1084, 1084, 1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084,1084,1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084,1084,1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084, 1084, 1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")


for r in range(3):
  # cell [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + 'gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271,271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271,271,271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542,542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542,542,542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

# for r in range(3):
#   # cell [1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")

# for r in range(3):
#   # cell [1084, 1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084,1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084,1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084, 1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")

# for r in range(3):
#   # cell [1084, 1084, 1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084,1084,1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084,1084,1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084, 1084, 1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")


for r in range(3):
  # cell [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + 'gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271,271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[271,271,271] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542,542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

for r in range(3):
  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[542,542,542] --runName=Gru-MIMIC-9722_', shell=True)

  with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

# for r in range(3):
#   # cell [1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")

# for r in range(3):
#   # cell [1084, 1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084,1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084,1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084, 1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")

# for r in range(3):
#   # cell [1084, 1084, 1084]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/9722-gru-model  --hiddenDimSize=[1084,1084,1084]', shell=True)
#   best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
#   stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722"' + " \"" + best_model + "\"" + '  --hiddenDimSize=[1084,1084,1084] --runName=Gru-MIMIC-9722_', shell=True)

#   with open(str(r) + '_gru-mimic-9722-experiments.txt', 'a') as f:
#     with redirect_stdout(f):
#       print(" --hiddenDimSize=[1084, 1084, 1084]")
#       print(stdout.decode().split("\n"))
#       print("\n")