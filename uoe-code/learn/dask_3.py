import dask
from time import sleep
import time

start_time = time.time()

data = [1, 2, 3, 4, 5, 6, 7, 8]

def inc(x):
    sleep(1)
    return x + 1


results = []
for x in data:
    y = inc(x)
    results.append(y)

total = sum(results)

print("--- %s seconds ---" % (time.time() - start_time))