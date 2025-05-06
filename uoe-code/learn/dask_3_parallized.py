import dask
from time import sleep
import time

from dask.distributed import Client

client = Client(n_workers=4)

start_time = time.time()

data = [1, 2, 3, 4, 5, 6, 7, 8]

@dask.delayed
def inc(x):
    sleep(1)
    return x + 1


results = []
for x in data:
    y = inc(x)
    results.append(y)

total = sum(results)
print("Before computing:", total)  # Let's see what type of thing total is
result = total.compute()
print("After computing :", result)  # After it's computed

print("--- %s seconds ---" % (time.time() - start_time))