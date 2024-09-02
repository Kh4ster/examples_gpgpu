import numpy as np
import cupy as cp
import nvtx

@nvtx.annotate("Host computation with NumPy")
def host_computation():
    a_host = np.random.rand(1000000)
    b_host = np.random.rand(1000000)
    c_host = a_host + b_host
    return c_host

def device_computation():
    nvtx.range_push("Device computation with CuPy")
    a_device = cp.random.rand(1000000)
    b_device = cp.random.rand(1000000)
    c_device = a_device + b_device
    nvtx.range_pop()
    return c_device

# Host computation
c_host = host_computation()

# Device computation
c_device = device_computation()

# Copy result back to host
c_device_host = cp.asnumpy(c_device)

# Validate results
print("Host computation result (first 5 elements):", c_host[:5])
print("Device computation result (first 5 elements):", c_device_host[:5])
