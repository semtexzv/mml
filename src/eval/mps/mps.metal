#include <metal_stdlib>

kernel void _add(
    device const float* a,
    device const float* b,
    device float* out,

    uint index [[thread_position_in_grid]])
{
    out[index] = a[index] + b[index];
}

kernel void _mul(
    device const float* a,
    device const float* b,
    device float* out,

    uint index [[thread_position_in_grid]])
{
    out[index] = a[index] + b[index];
}