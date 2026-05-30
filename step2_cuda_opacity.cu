// Step 2 — Smooth opacity probability α via a window function I(p). Still deterministic
// (no stochastic decision yet — that lands in step 3). 1 ray per thread.
//
// Idea: convert the ray's in-plane hit point into a SIGNED PERPENDICULAR DISTANCE d to the
// nearest triangle edge (world units, +inside / -outside), then
//     α = I(p) = sigmoid(d / σ).
// On the edge d=0 -> α=0.5; deep inside d≫σ -> α→1; outside d≪-σ -> α→0. The transition
// band has width ~σ, so σ=0.01 gives a thin soft edge.
//
// Crucially we compute α even when the ray geometrically MISSES the triangle (as long as it
// hits the plane in front): rays straddling the edge are exactly where the gradient lives in
// later steps, so we must not NaN them out here.
//
// Output convention (matches step0/step1 for bary):
//   bary  = (w, u, v) with P = w*V0 + u*V1 + v*V2,  w = 1 - u - v
//   sdist = signed perpendicular distance to nearest edge (world units)
//   alpha = sigmoid(sdist / sigma)  in [0, 1]
//   ray misses the plane (parallel) or hits behind origin -> alpha = 0, sdist/bary = NaN

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Numerically stable logistic sigmoid.
template <typename scalar_t>
__device__ __forceinline__ scalar_t stable_sigmoid(scalar_t x) {
    if (x >= (scalar_t)0) {
        const scalar_t z = exp(-x);
        return (scalar_t)1 / ((scalar_t)1 + z);
    } else {
        const scalar_t z = exp(x);
        return z / ((scalar_t)1 + z);
    }
}

template <typename scalar_t>
__global__ void opacity_kernel(
    const scalar_t* __restrict__ origins,     // (N, 3) row-major
    const scalar_t* __restrict__ directions,  // (N, 3) row-major
    const scalar_t* __restrict__ verts,       // 9 floats: v0xyz, v1xyz, v2xyz
    scalar_t        sigma,
    scalar_t*       __restrict__ alpha,        // (N,)
    scalar_t*       __restrict__ sdist,        // (N,)
    scalar_t*       __restrict__ bary,         // (N, 3)
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

    // --- Möller–Trumbore: barycentric (u, v, w) and parametric t ---
    const scalar_t px = dy*e2z - dz*e2y;
    const scalar_t py = dz*e2x - dx*e2z;
    const scalar_t pz = dx*e2y - dy*e2x;

    const scalar_t det = px*e1x + py*e1y + pz*e1z;
    const bool parallel = (det > -EPS) && (det < EPS);
    const scalar_t inv_det = parallel ? (scalar_t)0 : (scalar_t)1 / det;

    const scalar_t tx = ox - v0x, ty = oy - v0y, tz = oz - v0z;
    const scalar_t u = (tx*px + ty*py + tz*pz) * inv_det;

    const scalar_t qx = ty*e1z - tz*e1y;
    const scalar_t qy = tz*e1x - tx*e1z;
    const scalar_t qz = tx*e1y - ty*e1x;

    const scalar_t v = (dx*qx + dy*qy + dz*qz) * inv_det;
    const scalar_t t = (qx*e2x + qy*e2y + qz*e2z) * inv_det;
    const scalar_t w = (scalar_t)1 - u - v;

    // Only require an in-front plane intersection. We do NOT require u,v,w >= 0:
    // points just outside the triangle still get a (small) opacity so the edge is smooth.
    const bool valid = (!parallel) && (t > EPS);

    if (!valid) {
        alpha[i]    = (scalar_t)0;
        sdist[i]    = NaN;
        bary[3*i+0] = NaN; bary[3*i+1] = NaN; bary[3*i+2] = NaN;
        return;
    }

    // --- barycentric -> signed perpendicular distance to each edge (world units) ---
    // |n| = 2 * Area, where n = e1 x e2. Altitude from vertex i = |n| / L_i, with L_i the
    // length of the edge opposite vertex i. Perp distance to that edge = bary_i * altitude_i.
    const scalar_t nx = e1y*e2z - e1z*e2y;
    const scalar_t ny = e1z*e2x - e1x*e2z;
    const scalar_t nz = e1x*e2y - e1y*e2x;
    const scalar_t two_area = sqrt(nx*nx + ny*ny + nz*nz);   // |n|

    // Edge opposite V0 is V1V2; opposite V1 is V0V2 (=e2); opposite V2 is V0V1 (=e1).
    const scalar_t l0x = v2x - v1x, l0y = v2y - v1y, l0z = v2z - v1z;
    const scalar_t L0 = sqrt(l0x*l0x + l0y*l0y + l0z*l0z);
    const scalar_t L1 = sqrt(e2x*e2x + e2y*e2y + e2z*e2z);
    const scalar_t L2 = sqrt(e1x*e1x + e1y*e1y + e1z*e1z);

    const scalar_t d0 = w * (two_area / L0);   // distance to edge opposite V0
    const scalar_t d1 = u * (two_area / L1);   // distance to edge opposite V1
    const scalar_t d2 = v * (two_area / L2);   // distance to edge opposite V2

    scalar_t d = d0 < d1 ? d0 : d1;
    d = d < d2 ? d : d2;                        // signed distance to nearest edge

    alpha[i]    = stable_sigmoid(d / sigma);
    sdist[i]    = d;
    bary[3*i+0] = w;
    bary[3*i+1] = u;
    bary[3*i+2] = v;
}

std::vector<torch::Tensor> opacity_forward(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor v0,
    torch::Tensor v1,
    torch::Tensor v2,
    double sigma)
{
    TORCH_CHECK(origins.is_cuda(),    "origins must be CUDA");
    TORCH_CHECK(directions.is_cuda(), "directions must be CUDA");
    TORCH_CHECK(origins.dim() == 2 && origins.size(1) == 3,    "origins must be (N, 3)");
    TORCH_CHECK(directions.sizes() == origins.sizes(),         "directions must match origins");
    TORCH_CHECK(origins.scalar_type() == directions.scalar_type(), "dtype mismatch");
    TORCH_CHECK(v0.numel() == 3 && v1.numel() == 3 && v2.numel() == 3, "vertices must have 3 elements");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    origins    = origins.contiguous();
    directions = directions.contiguous();

    const auto N = origins.size(0);
    const auto opts = origins.options();

    auto verts = torch::empty({9}, opts);
    verts.narrow(0, 0, 3).copy_(v0.to(opts.device()).to(opts.dtype()).view({3}));
    verts.narrow(0, 3, 3).copy_(v1.to(opts.device()).to(opts.dtype()).view({3}));
    verts.narrow(0, 6, 3).copy_(v2.to(opts.device()).to(opts.dtype()).view({3}));

    auto alpha = torch::empty({N},    opts);
    auto sdist = torch::empty({N},    opts);
    auto bary  = torch::empty({N, 3}, opts);

    const int threads = 256;
    const int blocks  = (int)((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(origins.scalar_type(), "opacity_kernel", [&] {
        opacity_kernel<scalar_t><<<blocks, threads>>>(
            origins.data_ptr<scalar_t>(),
            directions.data_ptr<scalar_t>(),
            verts.data_ptr<scalar_t>(),
            (scalar_t)sigma,
            alpha.data_ptr<scalar_t>(),
            sdist.data_ptr<scalar_t>(),
            bary.data_ptr<scalar_t>(),
            (int)N);
    });

    return {alpha, sdist, bary};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("opacity_forward", &opacity_forward,
          "Smooth opacity α = sigmoid(signed_edge_distance / sigma), 1 ray per thread");
}