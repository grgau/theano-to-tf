import sys
import subprocess
from contextlib import redirect_stdout

# cell [3133]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133]', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[3133]")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [3133] - LR 1.0
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10 --learningRate=1.0', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133] --learningRate=1.0', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[3133] --learningRate=1.0")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [3133] - Dropout 0.30
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10 --dropoutRate=0.3', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133] --dropoutRate=0.3', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[3133] --dropoutRate=0.3")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [3133] - Dropout 0.55
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10 --dropoutRate=0.55', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133] --dropoutRate=0.55', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[3133] --dropoutRate=0.55")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [3133] - LR 0.25
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10 --learningRate=0.25', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133] --learningRate=0.25', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[3133] --learningRate=0.25")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [3133] - LR 0.25 - Dropout 0.30
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10 --learningRate=0.25 --dropoutRate=0.3', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133] --learningRate=0.25 --dropoutRate=0.3', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[3133] --learningRate=0.25 --dropoutRate=0.3")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [3133] - LR 1.0 - Dropout 0.30
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[3133] --maxConsecutiveNonImprovements=10 --learningRate=1.0 --dropoutRate=0.3', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[3133] --learningRate=1.0 --dropoutRate=0.3', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[3133] --learningRate=1.0 --dropoutRate=0.3")
    print(stdout.decode().split("\n"))
    print("\n")





# cell [6266]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[6266] --maxConsecutiveNonImprovements=10', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[6266]', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[6266]")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [6266] - LR 1.0
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[6266] --maxConsecutiveNonImprovements=10 --learningRate=1.0', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[6266] --learningRate=1.0', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[6266] --learningRate=1.0")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [6266] - Dropout 0.30
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[6266] --maxConsecutiveNonImprovements=10 --dropoutRate=0.3', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[6266] --dropoutRate=0.3', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[6266] --dropoutRate=0.3")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [6266] - Dropout 0.55
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[6266] --maxConsecutiveNonImprovements=10 --dropoutRate=0.55', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[6266] --dropoutRate=0.55', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[6266] --dropoutRate=0.55")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [6266] - LR 0.25
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[6266] --maxConsecutiveNonImprovements=10 --learningRate=0.25', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[6266] --learningRate=0.25', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[6266] --learningRate=0.25")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [6266] - LR 0.25- Dropout 0.30
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[6266] --maxConsecutiveNonImprovements=10 --learningRate=0.25 --dropoutRate=0.3', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[6266] --learningRate=0.25 --dropoutRate=0.3', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[6266] --learningRate=0.25 --dropoutRate=0.3")
    print(stdout.decode().split("\n"))
    print("\n")


# cell [6266] - LR 0.25- Dropout 0.30
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "data/incor/_3133" compiled_models/incor-encdec-model --state=cell --hiddenDimSize=[6266] --maxConsecutiveNonImprovements=10 --learningRate=1.0 --dropoutRate=0.3', shell=True)
best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "data/incor/_3133"' + " \"" + best_model + "\"" + ' --hiddenDimSize=[6266] --learningRate=1.0 --dropoutRate=0.3', shell=True)

with open('encdec-new-incor-experiments.txt', 'a') as f:
  with redirect_stdout(f):
    print("--state=cell --hiddenDimSize=[6266] --learningRate=1.0 --dropoutRate=0.3")
    print(stdout.decode().split("\n"))
    print("\n")
