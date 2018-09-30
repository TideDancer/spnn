import sys
import os
import numpy as np

filename = sys.argv[1]
with open(filename, 'r') as f:
  results = f.read().splitlines()
f.close()

results = filter(lambda e: 'layer' in e and 'reverse' not in e and 'skip' not in e and 'optimize' not in e, results)

for i in range(14):
  s = filter(lambda e: 'layer  '+str(i)+'  ' in e, results)
  s = map(lambda e: e.split(' : ')[1].split(' ==> ')[0], s)
  s = list(map(float, s))
  print(s)
