// Step 1 — Deterministic ray-triangle intersection on CUDA (Möller–Trumbore).
// 1 ray per thread. No stochastic masking yet (that lands in step 3).
// Output convention matches step0_cpu_reference.py:
//   bary = (w, u, v) with P = w*V0 + u*V1 + v*V2, w = 1 - u - v
//   miss -> t = NaN, bary = NaN, hit = false

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void intersect_kernel(
    const scalar_t* __restrict__ origins,     // (N, 3) row-major
    const scalar_t* __restrict__ directions,  // (N, 3) row-major
    const scalar_t* __restrict__ verts,       // 9 floats: v0xyz, v1xyz, v2xyz
    bool*           __restrict__ hit,         // (N,)
    scalar_t*       __restrict__ t_out,       // (N,)
    scalar_t*       __restrict__ bary,        // (N, 3)
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const scalar_t EPS = (scalar_t)1e-8;
    const scalar_t NaN = (scalar_t)NAN;

    const scalar_t v0x = verts[0], v0y = verts[1], v0z = verts[2];
    const scalar_t v1x = verts[3], v1y = verts[4], v1z = verts[5];
    const scalar_t v2x = verts[6], v2y = verts[7], v2z = verts[8];

    const scalar_t ox = origins[3*i+0], oy = origins[3*i+1], oz = origins[3*i+2];
    const scalar_t dx = directions[3*i+0], dy = directions[3*i+1], dz = directions[3*i+2];

    const scalar_t e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const scalar_t e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    // pvec = direction × e2
    const scalar_t px = dy*e2z - dz*e2y;
    const scalar_t py = dz*e2x - dx*e2z;
    const scalar_t pz = dx*e2y - dy*e2x;

    const scalar_t det = px*e1x + py*e1y + pz*e1z;
    const bool parallel = (det > -EPS) && (det < EPS);
    const scalar_t inv_det = parallel ? (scalar_t)0 : (scalar_t)1 / det;

    // tvec = origin - v0
    const scalar_t tx = ox - v0x, ty = oy - v0y, tz = oz - v0z;
    const scalar_t u = (tx*px + ty*py + tz*pz) * inv_det;

    // qvec = tvec × e1
    const scalar_t qx = ty*e1z - tz*e1y;
    const scalar_t qy = tz*e1x - tx*e1z;
    const scalar_t qz = tx*e1y - ty*e1x;

    const scalar_t v = (dx*qx + dy*qy + dz*qz) * inv_det;
    const scalar_t t = (qx*e2x + qy*e2y + qz*e2z) * inv_det;
    const scalar_t w = (scalar_t)1 - u - v;

    const bool h = (!parallel) && (u >= 0) && (v >= 0) && (w >= 0) && (t > EPS);

    hit[i] = h;
    t_out[i]      = h ? t : NaN;
    bary[3*i+0]   = h ? w : NaN;
    bary[3*i+1]   = h ? u : NaN;
    bary[3*i+2]   = h ? v : NaN;
}

std::vector<torch::Tensor> intersect_forward(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor v0,
    torch::Tensor v1,
    torch::Tensor v2)
{
    TORCH_CHECK(origins.is_cuda(),    "origins must be CUDA");
    TORCH_CHECK(directions.is_cuda(), "directions must be CUDA");
    TORCH_CHECK(origins.dim() == 2 && origins.size(1) == 3,    "origins must be (N, 3)");
    TORCH_CHECK(directions.sizes() == origins.sizes(),         "directions must match origins");
    TORCH_CHECK(origins.scalar_type() == directions.scalar_type(), "dtype mismatch");
    TORCH_CHECK(v0.numel() == 3 && v1.numel() == 3 && v2.numel() == 3, "vertices must have 3 elements");

    origins    = origins.contiguous();
    directions = directions.contiguous();

    const auto N = origins.size(0);
    const auto opts = origins.options();

    // Pack the 3 vertices into a single 9-element device buffer in the same dtype.
    auto verts = torch::empty({9}, opts);
    verts.narrow(0, 0, 3).copy_(v0.to(opts.device()).to(opts.dtype()).view({3}));
    verts.narrow(0, 3, 3).copy_(v1.to(opts.device()).to(opts.dtype()).view({3}));
    verts.narrow(0, 6, 3).copy_(v2.to(opts.device()).to(opts.dtype()).view({3}));

    auto hit  = torch::empty({N},    opts.dtype(torch::kBool));
    auto t    = torch::empty({N},    opts);
    auto bary = torch::empty({N, 3}, opts);

    const int threads = 256;
    const int blocks  = (int)((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(origins.scalar_type(), "intersect_kernel", [&] {
        intersect_kernel<scalar_t><<<blocks, threads>>>(
            origins.data_ptr<scalar_t>(),
            directions.data_ptr<scalar_t>(),
            verts.data_ptr<scalar_t>(),
            hit.data_ptr<bool>(),
            t.data_ptr<scalar_t>(),
            bary.data_ptr<scalar_t>(),
            (int)N);
    });

    return {hit, t, bary};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("intersect_forward", &intersect_forward,
          "Deterministic ray-triangle intersection (Möller–Trumbore), 1 ray per thread");
}
