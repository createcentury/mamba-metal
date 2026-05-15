// copy_scalar — measure raw global-memory bandwidth (1 float per thread)
//
// Inputs:  device const float* in_,  device const uint& n
// Output:  device float* out
// Dispatch: grid = (n,)

uint i = thread_position_in_grid.x;
if (i >= n) return;
out[i] = in_[i];
