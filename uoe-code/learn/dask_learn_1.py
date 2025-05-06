import dask
from time import sleep
import time

start_time = time.time()

@dask.delayed
def inc(x):
    sleep(1)
    return x + 1


@dask.delayed
def add(x, y):
    sleep(1)
    return x + y

x = inc(1)
y = inc(2)
z = add(x, y)

z.compute()

print("--- %s seconds ---" % (time.time() - start_time))