# Challenge 1: Stochastic Differentiable Triangle Soup — แผนงาน

> โจทย์จาก Umetani Lab อ้างอิงงาน **DiffSoup** (Tojo, Bickel, Umetani — CVPR 2026)
> หมายเหตุ: ชื่อ "UTrice" ในโจทย์หาเป็น paper สาธารณะไม่พบ น่าจะเป็นชื่อภายในแล็บ/ชื่อสมมุติ — ไม่กระทบการทำงาน เพราะแก่นเทคนิคทั้งหมดมาจาก stochastic opacity masking ของ DiffSoup

---

## 1. สรุปโจทย์ว่าจริงๆ ต้องทำอะไร

เขียน **CUDA kernel** ที่ทำ pipeline สามขั้น สำหรับสามเหลี่ยมหนึ่งรูปที่ differentiate ได้:

1. **Forward** — ยิงรังสี N=10⁶ เส้นเข้าหาสามเหลี่ยม คำนวณจุดตัด (ray intersection) และ barycentric coordinates
2. **Stochastic opacity masking** — แปลงระยะห่างจากขอบเป็นความน่าจะเป็นทึบแสง α ผ่าน window function `I(p)` แล้วสุ่มตัดสินว่ารังสีทะลุผ่านหรือไม่ สะสมค่า transmittance
3. **Backward** — พ่น gradient กลับไปอัปเดตตำแหน่ง vertex V₀, V₁, V₂ ด้วย atomic operations โดยไม่เกิด race condition

### หมายเหตุสำคัญเรื่องทิศทาง
paper จริงทำ **rasterization** (ฉายสามเหลี่ยมลงจอ) แต่โจทย์นี้ทำ **ray intersection** (ยิงรังสีเข้าหาสามเหลี่ยม) — เป็นมุมกลับกัน แต่ใช้ trick เดียวกันคือ stochastic masking

### Test Case
- Triangle: `V₀(0,0,0)`, `V₁(1,0,0)`, `V₂(0,1,0)`
- `σ = 0.01` (smoothness factor)
- ยิงรังสี `N = 10⁶` เส้น โดยเจาะจงบริเวณ **ขอบ** สามเหลี่ยม (barycentric coordinate เข้าใกล้ 0)

### Expected Result
- contribution กระจายตัวเหมือนขอบสามเหลี่ยมมีความโปร่งแสงแบบ gradient (gradient response)
- พ่นเวกเตอร์ gradient กลับไปอัปเดต V₀, V₁, V₂ ได้ถูกต้องโดยไม่เกิด race condition

---

## 2. หัวใจของโจทย์ — ทำไมต้อง stochastic

| ปัญหา | สาเหตุ | ทางแก้ |
|---|---|---|
| Binary opacity ไม่ differentiable | ฟังก์ชันขั้นบันได → ∂/∂V = 0 ทุกที่ → เรียนรู้ไม่ได้ | แปลงระยะขอบเป็นความน่าจะเป็น α แล้วสุ่ม → contribution มี gradient ที่ขอบ |
| 10⁶ threads เขียน gradient ของ V เดียวกัน | vertex 3 ตัวถูกแชร์โดยทุกรังสี | `atomicAdd` ลง gradient buffer |

แนวคิดเชิงสถิติเบื้องหลัง: เป็นหลักการเดียวกับ **score-function estimator / REINFORCE** — แปลงปัญหา discrete ให้กลายเป็น *expectation* ที่ derivative ผ่านเข้าไปได้

---

## 3. สิ่งที่ต้องเรียนรู้ก่อน

จัดลำดับความเสี่ยง: คนส่วนใหญ่**ไม่ได้พลาดที่ Möller–Trumbore** (หาโค้ดได้) แต่พลาดที่ *"ไม่เข้าใจว่าทำไม stochastic ให้ gradient"* กับ *"race condition ใน backward"* — ลงแรงสองอันนี้มากที่สุด

### กลุ่มที่ 1 — คณิตศาสตร์ของ ray-triangle
- [ ] **Möller–Trumbore intersection** — เข้าใจถึงระดับสมการ ไม่ใช่แค่ copy โค้ด เพราะ backward pass บังคับให้รู้ว่า barycentric (u, v, w) มาจากสมการไหน
- [ ] barycentric coordinate เข้าใกล้ 0 = จุดตัดอยู่ที่ "ขอบ" สามเหลี่ยม (บริเวณที่ test case เจาะจง)

### กลุ่มที่ 2 — แก่นงานวิจัย (สำคัญที่สุด)
- [ ] **stochastic opacity masking** — ทำไมการสุ่มถึงทำให้ค่าคาดหวังของ gradient ไม่เป็นศูนย์
- [ ] เชื่อมโยงกับ score-function estimator / REINFORCE
- [ ] ทำไม binary opacity ที่ดูเหมือน non-differentiable ถึง differentiate ได้ผ่านการสุ่ม

### กลุ่มที่ 3 — Window function I(p) และ σ
- [ ] σ = ความกว้างของบริเวณรอยต่อที่ขอบ; σ เล็ก (0.01) → ขอบคม → gradient กระจุกในแถบแคบ
- [ ] ออกแบบ I(p) เอง (เช่น sigmoid ของ signed distance / σ) และเข้าใจว่าทำไมรูปแบบนั้นให้ gradient ดี

### กลุ่มที่ 4 — CUDA
- [ ] **atomic operations** — `atomicAdd` แก้ race ได้อย่างไร และทำไมช้าถ้า contention สูง
- [ ] **per-thread RNG** — cuRAND หรือ counter-based (Philox); 10⁶ threads ต้องการ random ที่ไม่ซ้ำและ reproducible
- [ ] **thread/memory layout** — grid/block, map รังสี 1 เส้น → 1 thread

---

## 4. สิ่งที่ต้องทำ — milestones ที่ทดสอบได้ทีละขั้น

> **กฎเหล็ก:** สร้างแบบ incremental ห้ามเขียนรวดเดียวจบ — debug CUDA + gradient พร้อมกันคือฝันร้าย แต่ละขั้นต้องตรวจสอบได้ก่อนไปขั้นถัดไป

### ขั้น 0 — Environment + CPU reference
เขียน ray-triangle intersection บน CPU (Python/NumPy) สำหรับ test case ตรวจด้วยมือว่ารังสีที่ควรโดน/ไม่โดนให้ผลถูก → ใช้เป็น ground truth เทียบ CUDA ทีหลัง

### ขั้น 1 — Forward pass แบบ deterministic
CUDA kernel ยิง 1 รังสี/thread ตอบแค่ว่าโดนไหม + คืน barycentric ยังไม่สุ่ม
**ตรวจ:** ผลตรงกับ CPU reference

### ขั้น 2 — เพิ่ม opacity probability
ใส่ window function I(p) คำนวณ α จากระยะขอบและ σ แต่ยังไม่สุ่ม คืน α ออกมาดู
**ตรวจ:** ยิงรังสีไล่จากกลางสามเหลี่ยมไปขอบ → α ลดจาก ~1 ไป ~0 อย่างนุ่มนวล ความกว้างแถบสอดคล้องกับ σ=0.01

### ขั้น 3 — ใส่ stochastic decision
เพิ่ม RNG ต่อ thread สุ่ม ξ เทียบ α สะสม transmittance ยิงครบ N=10⁶
**ตรวจ:** ค่าเฉลี่ย contribution ที่ขอบนุ่มนวล (gradient response); ค่าเฉลี่ยจาก 10⁶ samples ลู่เข้าหา α

### ขั้น 4 — Backward pass + atomics
หา ∂contribution/∂V₀,V₁,V₂ ด้วย chain rule แล้ว `atomicAdd` ลง gradient buffer
**ระวัง:** vertex 3 ตัวถูกแชร์โดยทุกรังสี → ทุก thread atomicAdd ลงที่เดียวกัน

### ขั้น 5 — ตรวจความถูกต้องของ gradient (สำคัญที่สุด)
**Finite differences:** ขยับ V₀ ทีละ ε เล็กๆ ดู contribution เปลี่ยนเท่าไหร่ เทียบกับ gradient ที่ kernel คำนวณ ตรง (ในขอบเขต Monte Carlo noise) = ใช้ได้

### ขั้น 6 — ตรวจ race condition
รัน kernel เดิมหลายครั้งด้วย input เดียวกัน → ผล gradient ต้องเสถียร (ภายใน FP tolerance)
ถ้าผลแกว่งแบบสุ่มทุกรอบ = ยังมี race

---

## 5. กับดักที่ต้องระวัง

**`atomicAdd` กับ `float` มี non-determinism โดยธรรมชาติ** — ลำดับการบวก floating-point ไม่คงที่ระหว่างรอบ นี่**ไม่ใช่** race condition (ผลถูกต้องเสมอ) แต่ทำให้ bit-exact reproducibility ทำไม่ได้

ตอนทดสอบขั้น 6 ต้องแยกให้ออกระหว่าง:
- ผลต่างจาก **floating-point ordering** → ยอมรับได้
- ผลต่างจาก **race จริง** → bug

---

## 6. แหล่งอ้างอิง

- DiffSoup project page: https://kenji-tojo.github.io/publications/diffsoup/
- GitHub (official code): https://github.com/kenji-tojo/diffsoup
- arXiv: https://arxiv.org/abs/2603.27151
