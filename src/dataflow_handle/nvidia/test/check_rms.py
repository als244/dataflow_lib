import numpy as np

M = 16384
N = 8192

fwd_dtype = np.float16
bwd_dtype = np.float16
sq_sum_dtype = np.float32

orig = np.fromfile("test_rms/orig_matrix.dat", dtype=fwd_dtype).reshape(M, N)
weights = np.fromfile("test_rms/weights.dat", dtype=fwd_dtype).reshape(N)
out = np.fromfile("test_rms/fwd_out_matrix.dat", dtype=fwd_dtype).reshape(M, N)
sq_sums = np.fromfile("test_rms/sq_sums.dat", dtype=sq_sum_dtype).reshape(M)

upstream_dX = np.fromfile("test_rms/upstream_dX.dat", dtype=bwd_dtype).reshape(M, N)
dX = np.fromfile("test_rms/dX_matrix.dat", dtype=bwd_dtype).reshape(M, N)
dW = np.fromfile("test_rms/dWeights.dat", dtype=bwd_dtype).reshape(N)

