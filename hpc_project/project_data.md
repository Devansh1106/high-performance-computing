## Comparison based on varying threads
This is for 1 GPU
---
```
M = N =   16384  
Serial Time:    820.290000 ms  
Multi-GPU Time: 111.282532 ms  
Verification: y[0](gpu) = 4090.139648 and (cpu) 4090.139648  
threads_per_block=64  
```

```
M = N =   16384  
Serial Time:    821.446000 ms  
Multi-GPU Time: 113.311295 ms  
Verification: y[0](gpu) = 4090.139648 and (cpu) 4090.139648  
threads_per_block=128  
```

```
M = N =   16384  
Multi-GPU Time: 116.815742 ms  
Verification: y[0](gpu) = 4090.139648  
threads_per_block=256  
```

```
M = N =   16384  
Multi-GPU Time: 116.791328 ms  
Verification: y[0](gpu) = 4090.139648  
theards_per_block=512  
```

```
M = N =   16384  
Multi-GPU Time: 121.635300 ms  
Verification: y[0](gpu) = 4090.13964  
threads_per_block=1024  
```

---

This is for 2 GPUs
---
```
M = N =   16384  
Multi-GPU Time: 120.644508 ms  
Verification: y[0](gpu) = 4090.139648  
threads_per_block = 1024  
```

```
M = N =   16384  
Multi-GPU Time: 102.326996 ms  
Verification: y[0](gpu) = 4090.139648  
threads_per_block = 512  
```

```
M = N =   16384  
Multi-GPU Time: 101.792160 ms  
Verification: y[0](gpu) = 4090.139648  
threads_per_block = 256  
```

```
M = N =   16384  
Multi-GPU Time: 101.303200 ms  
Verification: y[0](gpu) = 4090.139648  
threads_per_block = 128  
```

```
M = N =   16384  
Multi-GPU Time: 100.049759 ms  
Verification: y[0](gpu) = 4090.139648  
threads_per_block = 64  
```

---
## Comparison based on varying GPUs vs CPU

Timings for numGPUs = 1
---
```
M = N =   8192  
Serial Time:    204.096000 ms  
Multi-GPU Time: 29.296608 ms  
Verification: y[0](gpu) = 2064.640137 and (cpu) 2064.640137  
threads_per_block = 256  
```

```
M = N =   16384  
Serial Time:    818.345000 ms  
Multi-GPU Time: 116.717758 ms  
Verification: y[0](gpu) = 4090.139648 and (cpu) 4090.139648  
threads_per_block = 256  
```

```
M = N =   32768  
Serial Time:    3266.392000 ms  
Multi-GPU Time: 401.356812 ms  
Verification: y[0](gpu) = 8177.334961 and (cpu) 8177.334961  
threads_per_block = 256  
```

---

Now is the timing for 2 GPUs:
---
```
M = N =   32768  
Serial Time:    3243.403000 ms  
Multi-GPU Time: 371.139435 ms  
Verification: y[0](gpu) = 8177.334961 and (cpu) 8177.334961  
threads_per_block = 256  
```

```
M = N =   16384  
Serial Time:    814.523000 ms  
Multi-GPU Time: 101.856094 ms  
Verification: y[0](gpu) = 4090.139648 and (cpu) 4090.139648  
threads_per_block = 256  
```

```
M = N =   8192  
Serial Time:    220.527000 ms  
Multi-GPU Time: 25.780191 ms  
Verification: y[0](gpu) = 2064.640137 and (cpu) 2064.640137  
threads_per_block = 256  
```
