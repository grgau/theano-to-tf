import sys
import subprocess
from contextlib import redirect_stdout

for r in range(5):
  # cell [271]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + 'encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084, 1084]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=cell --hiddenDimSize=[1084,1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--state=cell --hiddenDimSize=[1084, 1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")





# # hidden [271]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[271]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[271]")
#     print(stdout.decode().split("\n"))
#     print("\n")
#
# # hidden [271, 271]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[271,271]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[271, 271]")
#     print(stdout.decode().split("\n"))
#     print("\n")
#
# # hidden [271, 271, 271]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[271,271,271]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[271, 271, 271]")
#     print(stdout.decode().split("\n"))
#     print("\n")
#
# # hidden [542]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[542]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[542]")
#     print(stdout.decode().split("\n"))
#     print("\n")
#
# # hidden [542, 542]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[542,542]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[542, 542]")
#     print(stdout.decode().split("\n"))
#     print("\n")
#
# # hidden [542, 542, 542]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[542,542,542]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[542, 542, 542]")
#     print(stdout.decode().split("\n"))
#     print("\n")
#
# # hidden [1084]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[1084]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[1084]")
#     print(stdout.decode().split("\n"))
#     print("\n")
#
# # hidden [1084, 1084]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[1084,1084]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[1084, 1084]")
#     print(stdout.decode().split("\n"))
#     print("\n")
#
# # hidden [1084, 1084, 1084]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google.py "src/data-without-bidirectional/mimic-90-10_855" compiled_models/855-encdec-model --state=hidden --hiddenDimSize=[1084,1084,1084]', shell=True)
# best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
# stdout = subprocess.check_output('python3.5 src/tf/LSTM-tf-Google-test.py "src/data-without-bidirectional/mimic-90-10_855"' + " \"" + best_model + "\"", shell=True)
#
# with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
#   with redirect_stdout(f):
#     print("--state=hidden --hiddenDimSize=[1084, 1084, 1084]")
#     print(stdout.decode().split("\n"))
#     print("\n")
