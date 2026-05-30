// Step 3 — Stochastic opacity decision. Builds on step 2's smooth α, then turns it into a
// Bernoulli sample: draw ξ ~ U(0,1] per ray and decide opaque ⇔ (ξ < α). 1 ray per thread.
//
// Why stochastic? A hard inside/outside test is a step function — ∂/∂V = 0 almost everywhere,
// so the edge carries no gradient. Replacing it with a coin flip whose bias is α makes the
// *expected* contribution equal to α, which is smooth in V. Averaging M samples at a fixed
// point converges to α (Monte Carlo); the gradient of that expectation is the score-function
// (REINFORCE) estimator we wire up in step 4. This step just produces the samples + α.
//
// RNG: cuRAND Philox (counter-based). Seed is shared; the per-thread *sequence* = ray index,
// so the 10^6 streams are independent and the whole launch is reproducible for a given seed.
//
// Output:
//   alpha  (N,)  smooth opacity probability in [0,1]   (same as step 2)
//   sample (N,)  Bernoulli draw in {0.0, 1.0}: 1 = opaque/blocked, 0 = transmitted
//   ray misses the plane (parallel / behind) -> alpha = 0, sample = 0

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

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
__global__ void stochastic_kernel(
    const scalar_t* __restrict__ origins,     // (N, 3) row-major
    const scalar_t* __restrict__ directions,  // (N, 3) row-major
    const scalar_t* __restrict__ verts,       // 9 floats: v0xyz, v1xyz, v2xyz
    scalar_t          sigma,
    unsigned long long seed,
    scalar_t*       __restrict__ alpha,        // (N,)
    scalar_t*       __restrict__ sample,       // (N,)  {0,1}
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const scalar_t EPS = (scalar_t)1e-8;

    const scalar_t v0x = verts[0], v0y = verts[1], v0z = verts[2];
    const scalar_t v1x = verts[3], v1y = verts[4], v1z = verts[5];
    const scalar_t v2x = verts[6], v2y = verts[7], v2z = verts[8];

    const scalar_t ox = origins[3*i+0], oy = origins[3*i+1], oz = origins[3*i+2];
    const scalar_t dx = directions[3*i+0], dy = directions[3*i+1], dz = directions[3*i+2];

    const scalar_t e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    const scalar_t e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

    // --- Möller–Trumbore: barycentric (u, v, w), parametric t ---
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

    const bool valid = (!parallel) && (t > EPS);

    scalar_t a = (scalar_t)0;
    if (valid) {
        // barycentric -> signed perpendicular distance to nearest edge (world units)
        const scalar_t nx = e1y*e2z - e1z*e2y;
        const scalar_t ny = e1z*e2x - e1x*e2z;
        const scalar_t nz = e1x*e2y - e1y*e2x;
        const scalar_t two_area = sqrt(nx*nx + ny*ny + nz*nz);

        const scalar_t l0x = v2x - v1x, l0y = v2y - v1y, l0z = v2z - v1z;
        const scalar_t L0 = sqrt(l0x*l0x + l0y*l0y + l0z*l0z);
        const scalar_t L1 = sqrt(e2x*e2x + e2y*e2y + e2z*e2z);
        const scalar_t L2 = sqrt(e1x*e1x + e1y*e1y + e1z*e1z);

        const scalar_t d0 = w * (two_area / L0);
        const scalar_t d1 = u * (two_area / L1);
        const scalar_t d2 = v * (two_area / L2);
        scalar_t d = d0 < d1 ? d0 : d1;
        d = d < d2 ? d : d2;

        a = stable_sigmoid(d / sigma);
    }

    // --- stochastic decision: ξ ~ U(0,1], opaque ⇔ ξ < α -------------------
    // Philox: shared seed, per-ray sequence = i -> independent reproducible streams.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)i, 0ULL, &state);
    const float xi = curand_uniform(&state);          // (0, 1]
    const scalar_t s = (xi < (float)a) ? (scalar_t)1 : (scalar_t)0;

    alpha[i]  = a;
    sample[i] = valid ? s : (scalar_t)0;
}

std::vector<torch::Tensor> stochastic_forward(
    torch::Tensor origins,
    torch::Tensor directions,
    torch::Tensor v0,
    torch::Tensor v1,
    torch::Tensor v2,
    double sigma,
    int64_t seed)
{
    TORCH_CHECK(origins.is_cuda(),    "origins must be CUDA");
    TORCH_CHECK(directions.is_cuda(), "directions must be CUDA");
    TORCH_CHECK(origins.dim() == 2 && origins.size(1) == 3, "origins must be (N, 3)");
    TORCH_CHECK(directions.sizes() == origins.sizes(),      "directions must match origins");
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

    auto alpha  = torch::empty({N}, opts);
    auto sample = torch::empty({N}, opts);

    const int threads = 256;
    const int blocks  = (int)((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(origins.scalar_type(), "stochastic_kernel", [&] {
        stochastic_kernel<scalar_t><<<blocks, threads>>>(
            origins.data_ptr<scalar_t>(),
            directions.data_ptr<scalar_t>(),
            verts.data_ptr<scalar_t>(),
            (scalar_t)sigma,
            (unsigned long long)seed,
            alpha.data_ptr<scalar_t>(),
            sample.data_ptr<scalar_t>(),
            (int)N);
    });

    return {alpha, sample};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stochastic_forward", &stochastic_forward,
          "Stochastic opacity: Bernoulli(α) per ray via Philox RNG, 1 ray per thread");
}
