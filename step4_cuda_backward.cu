// Step 4 — Backward pass with atomics.
//
// Gradient of the smooth opacity α w.r.t. the (in-plane) vertex positions, accumulated into a
// shared 9-float buffer with atomicAdd. This is the DiffSoup trick: the FORWARD render is a
// stochastic Bernoulli(α) sample (step 3), but the GRADIENT flows through the smooth, exact
// expectation E[sample] = α(V), which is differentiable. So
//     dL/dV = Σ_i (dL/dα_i) · (dα_i/dV)            (pathwise, low-variance)
// is what we accumulate. (The score-function / REINFORCE estimator would target the same
// expected gradient with higher variance; we use the smooth-α path here.)
//
// Signed distance to the edge opposite vertex k, with the hit point P fixed in the plane:
//     d_k = ((Va - P) × (Vb - Va)) / |Vb - Va|        (× = 2D scalar cross, CCW triangle)
// where (Va,Vb) is that edge oriented CCW. This equals λ_k·|n|/L_k from steps 2–3 but depends
// only on the edge's two endpoints, so dα_i/dV touches exactly the two nearest-edge vertices —
// the opposite vertex gets zero. α = sigmoid(min_k d_k / σ),  dα/dd = α(1-α)/σ.
//
// All N rays write the SAME 3 vertices ⇒ heavy contention on 6 floats. atomicAdd serializes the
// writes so there is no race (the sum is correct); FP add-ordering still varies run-to-run, which
// is NOT a race (step 6 distinguishes the two).
//
// Scope: vertices lie in z = 0 and rays shoot along ±z (the challenge test case). Gradients are
// computed for the x,y of each vertex; the z column stays 0 (in-plane perturbations keep P fixed).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t stable_sigmoid(scalar_t x) {
    if (x >= (scalar_t)0) { const scalar_t z = exp(-x); return (scalar_t)1 / ((scalar_t)1 + z); }
    else                  { const scalar_t z = exp(x);  return z / ((scalar_t)1 + z); }
}

// 2D signed distance from P to the directed line A->B. Positive on the left (CCW) side.
template <typename scalar_t>
__device__ __forceinline__ scalar_t sdist2d(
    scalar_t px, scalar_t py, scalar_t ax, scalar_t ay, scalar_t bx, scalar_t by)
{
    const scalar_t ex = bx - ax, ey = by - ay;
    const scalar_t L  = sqrt(ex*ex + ey*ey);
    const scalar_t cr = (ax - px) * ey - (ay - py) * ex;
    return cr / L;
}

// Möller–Trumbore -> (valid, hit point P). Shared by forward and backward so both agree exactly.
template <typename scalar_t>
__device__ __forceinline__ bool hit_point(
    const scalar_t* o, const scalar_t* d, const scalar_t* V,
    scalar_t& px, scalar_t& py, scalar_t& pz)
{
    const scalar_t EPS = (scalar_t)1e-8;
    const scalar_t e1x = V[3]-V[0], e1y = V[4]-V[1], e1z = V[5]-V[2];
    const scalar_t e2x = V[6]-V[0], e2y = V[7]-V[1], e2z = V[8]-V[2];
    const scalar_t hx = d[1]*e2z - d[2]*e2y;
    const scalar_t hy = d[2]*e2x - d[0]*e2z;
    const scalar_t hz = d[0]*e2y - d[1]*e2x;
    const scalar_t det = hx*e1x + hy*e1y + hz*e1z;
    if (det > -EPS && det < EPS) return false;
    const scalar_t inv = (scalar_t)1 / det;
    const scalar_t tx = o[0]-V[0], ty = o[1]-V[1], tz = o[2]-V[2];
    const scalar_t qx = ty*e1z - tz*e1y, qy = tz*e1x - tx*e1z, qz = tx*e1y - ty*e1x;
    const scalar_t t  = (qx*e2x + qy*e2y + qz*e2z) * inv;
    if (t <= EPS) return false;
    px = o[0] + t*d[0]; py = o[1] + t*d[1]; pz = o[2] + t*d[2];
    return true;
}

// Nearest edge: returns min signed distance and the edge index (0,1,2 opposite V0,V1,V2).
template <typename scalar_t>
__device__ __forceinline__ scalar_t nearest_edge(
    scalar_t px, scalar_t py, const scalar_t* V, int& edge)
{
    // edge opposite V0 -> (V1,V2); opposite V1 -> (V2,V0); opposite V2 -> (V0,V1)
    const scalar_t d0 = sdist2d(px,py, V[3],V[4], V[6],V[7]);
    const scalar_t d1 = sdist2d(px,py, V[6],V[7], V[0],V[1]);
    const scalar_t d2 = sdist2d(px,py, V[0],V[1], V[3],V[4]);
    scalar_t d = d0; edge = 0;
    if (d1 < d) { d = d1; edge = 1; }
    if (d2 < d) { d = d2; edge = 2; }
    return d;
}

// ---------------------------------------------------------------------------
// Forward: α = sigmoid(min edge distance / σ).  Matches steps 2–3.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ origins, const scalar_t* __restrict__ directions,
    const scalar_t* __restrict__ verts, scalar_t sigma,
    scalar_t* __restrict__ alpha, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    scalar_t px, py, pz;
    if (!hit_point(origins + 3*i, directions + 3*i, verts, px, py, pz)) { alpha[i] = 0; return; }
    int edge; const scalar_t d = nearest_edge(px, py, verts, edge);
    alpha[i] = stable_sigmoid(d / sigma);
}

// ---------------------------------------------------------------------------
// Backward: accumulate dL/dV = Σ_i grad_alpha_i · dα_i/dV  via atomicAdd.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ origins, const scalar_t* __restrict__ directions,
    const scalar_t* __restrict__ verts, scalar_t sigma,
    const scalar_t* __restrict__ grad_alpha,   // (N,) upstream dL/dα
    scalar_t* __restrict__ grad_verts,         // (9,) accumulated
    int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    scalar_t px, py, pz;
    if (!hit_point(origins + 3*i, directions + 3*i, verts, px, py, pz)) return;

    int edge;
    const scalar_t d = nearest_edge(px, py, verts, edge);
    const scalar_t a = stable_sigmoid(d / sigma);
    const scalar_t dadd = a * ((scalar_t)1 - a) / sigma;   // dα/dd
    const scalar_t factor = grad_alpha[i] * dadd;          // dL/dα · dα/dd = dL/dd

    // Endpoints (A,B) of the active edge, oriented CCW, and their vertex slots.
    int ia, ib;
    if      (edge == 0) { ia = 1; ib = 2; }   // (V1,V2)
    else if (edge == 1) { ia = 2; ib = 0; }   // (V2,V0)
    else                { ia = 0; ib = 1; }   // (V0,V1)

    const scalar_t ax = verts[3*ia+0], ay = verts[3*ia+1];
    const scalar_t bx = verts[3*ib+0], by = verts[3*ib+1];
    const scalar_t ex = bx - ax, ey = by - ay;
    const scalar_t L2 = ex*ex + ey*ey;
    const scalar_t L  = sqrt(L2);
    const scalar_t cr = (ax - px) * ey - (ay - py) * ex;   // = d * L

    // ∂d/∂(endpoint coord) = ∂cross/∂·/L − cross·∂L/∂·/L²
    const scalar_t dd_ax = (ey + (ay - py)) / L - cr * (-ex / L) / L2;
    const scalar_t dd_ay = (-(ax - px) - ex) / L - cr * (-ey / L) / L2;
    const scalar_t dd_bx = (-(ay - py))     / L - cr * ( ex / L) / L2;
    const scalar_t dd_by = ( (ax - px))     / L - cr * ( ey / L) / L2;

    atomicAdd(&grad_verts[3*ia+0], factor * dd_ax);
    atomicAdd(&grad_verts[3*ia+1], factor * dd_ay);
    atomicAdd(&grad_verts[3*ib+0], factor * dd_bx);
    atomicAdd(&grad_verts[3*ib+1], factor * dd_by);
    // z components (slots 2,5,8) are untouched: in-plane gradient only.
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
static torch::Tensor pack_verts(torch::Tensor v0, torch::Tensor v1, torch::Tensor v2,
                                const torch::TensorOptions& opts) {
    auto verts = torch::empty({9}, opts);
    verts.narrow(0,0,3).copy_(v0.to(opts.device()).to(opts.dtype()).view({3}));
    verts.narrow(0,3,3).copy_(v1.to(opts.device()).to(opts.dtype()).view({3}));
    verts.narrow(0,6,3).copy_(v2.to(opts.device()).to(opts.dtype()).view({3}));
    return verts;
}

torch::Tensor opacity_forward(torch::Tensor origins, torch::Tensor directions,
                              torch::Tensor v0, torch::Tensor v1, torch::Tensor v2, double sigma) {
    TORCH_CHECK(origins.is_cuda() && directions.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(origins.dim()==2 && origins.size(1)==3, "origins must be (N,3)");
    origins = origins.contiguous(); directions = directions.contiguous();
    const auto N = origins.size(0); const auto opts = origins.options();
    auto verts = pack_verts(v0,v1,v2,opts);
    auto alpha = torch::empty({N}, opts);
    const int threads = 256, blocks = (int)((N+threads-1)/threads);
    AT_DISPATCH_FLOATING_TYPES(origins.scalar_type(), "forward_kernel", [&]{
        forward_kernel<scalar_t><<<blocks,threads>>>(
            origins.data_ptr<scalar_t>(), directions.data_ptr<scalar_t>(),
            verts.data_ptr<scalar_t>(), (scalar_t)sigma, alpha.data_ptr<scalar_t>(), (int)N);
    });
    return alpha;
}

torch::Tensor opacity_backward(torch::Tensor origins, torch::Tensor directions,
                               torch::Tensor v0, torch::Tensor v1, torch::Tensor v2,
                               double sigma, torch::Tensor grad_alpha) {
    TORCH_CHECK(origins.is_cuda() && directions.is_cuda() && grad_alpha.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(grad_alpha.numel()==origins.size(0), "grad_alpha must be (N,)");
    origins = origins.contiguous(); directions = directions.contiguous();
    grad_alpha = grad_alpha.contiguous();
    const auto N = origins.size(0); const auto opts = origins.options();
    auto verts = pack_verts(v0,v1,v2,opts);
    auto grad_verts = torch::zeros({9}, opts);          // <- zero-init; kernel accumulates
    const int threads = 256, blocks = (int)((N+threads-1)/threads);
    AT_DISPATCH_FLOATING_TYPES(origins.scalar_type(), "backward_kernel", [&]{
        backward_kernel<scalar_t><<<blocks,threads>>>(
            origins.data_ptr<scalar_t>(), directions.data_ptr<scalar_t>(),
            verts.data_ptr<scalar_t>(), (scalar_t)sigma,
            grad_alpha.data_ptr<scalar_t>(), grad_verts.data_ptr<scalar_t>(), (int)N);
    });
    return grad_verts.view({3,3});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("opacity_forward",  &opacity_forward,  "Smooth opacity α (2D signed-distance form)");
    m.def("opacity_backward", &opacity_backward, "Accumulate dL/dV via atomicAdd (in-plane)");
}
