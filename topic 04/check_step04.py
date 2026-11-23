"""
ai-econ-lab · Step 04 — checks for your TODOs (no internet, no model download).

    python step04/check_step04.py              # checks your attention.py
    python step04/check_step04.py --solution   # checks the reference solution
"""
import sys
import numpy as np

if "--solution" in sys.argv:
    import solution_attention as A
else:
    import attention as A

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except NotImplementedError as e:
        results.append((name, False, f"not done yet ({e})"))
    except AssertionError as e:
        results.append((name, False, str(e) or "assertion failed"))


# --- TODO 1: softmax -------------------------------------------------------
def t_softmax():
    w = A.softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(w.sum(), 1.0), "weights must sum to 1"
    assert np.allclose(w, [0.09003057, 0.24472847, 0.66524096]), f"got {w}"
    big = A.softmax(np.array([1000.0, 1001.0]))          # naive exp overflows here
    assert np.all(np.isfinite(big)), "overflow: subtract the max before exp"
    assert np.allclose(big, [0.26894142, 0.73105858]), f"got {big}"
    m = A.softmax(np.array([5.0, 1.0, 1.0]), mask=np.array([False, True, True]))
    assert m[0] == 0.0 and np.allclose(m[1:], [0.5, 0.5]), f"mask ignored: {m}"


# --- TODO 2: attention -----------------------------------------------------
def t_attention():
    # Worked example: q=[1,0], keys e1,e2, values 10,20, d=2, tau=1
    # scores = [1/sqrt(2), 0] = [0.7071, 0]
    # w1 = e^0.7071 / (e^0.7071 + 1) = 2.0281 / 3.0281 = 0.6698
    # out = 0.6698*10 + 0.3302*20 = 13.30
    q = np.array([1.0, 0.0])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([10.0, 20.0])
    out, w = A.attention(q, K, V, tau=1.0)
    assert np.allclose(w, [0.66976155, 0.33023845]), f"weights {w}"
    assert np.isclose(out, 13.3023845), f"output {out} (expected 13.30)"
    out0, _ = A.attention(q, K, V, tau=1e-3)             # tau -> 0 : nearest neighbour
    assert np.isclose(out0, 10.0), f"tau->0 should pick the closest key, got {out0}"
    outinf, _ = A.attention(q, K, V, tau=1e6)            # tau -> inf : simple average
    assert np.isclose(outinf, 15.0, atol=1e-4), f"tau->inf should average, got {outinf}"


# --- TODO 3: top-k sparse attention -----------------------------------------
def t_topk():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    K = rng.normal(size=(50, 4))
    V = rng.normal(size=50)
    dense, wd = A.attention(q, K, V, tau=0.5)
    full, wf = A.topk_attention(q, K, V, k=50, tau=0.5)
    assert np.isclose(dense, full), "k = n must equal dense attention"
    out, w = A.topk_attention(q, K, V, k=5, tau=0.5)
    assert (w > 0).sum() == 5, f"exactly 5 non-zero weights expected, got {(w > 0).sum()}"
    best5 = np.argsort(K @ q)[-5:]
    assert set(np.flatnonzero(w)) == set(best5), "kept keys must be the 5 highest-scoring"
    one, _ = A.topk_attention(q, K, V, k=1)
    assert np.isclose(one, V[np.argmax(K @ q)]), "k=1 must return the best key's value"


# --- TODO 4: compress_kv ----------------------------------------------------
def t_compress():
    K = np.arange(14, dtype=float).reshape(7, 2)       # rows 0..6
    V = np.arange(7, dtype=float)
    Kc, Vc = A.compress_kv(K, V, block=2)               # drop row 0, pool (1,2),(3,4),(5,6)
    assert Kc.shape == (3, 2) and Vc.shape == (3,), f"shapes {Kc.shape}, {Vc.shape}"
    assert np.allclose(Vc, [1.5, 3.5, 5.5]), f"Vc {Vc}: drop the OLDEST rows, keep recent"
    assert np.allclose(Kc[-1], [11.0, 12.0]), f"Kc last block {Kc[-1]}"


# --- given scaffolding: no look-ahead ---------------------------------------
def t_causal():
    y = np.arange(100, dtype=float) ** 1.1
    T, L, h = 60, 12, 3
    K, V, ends = A.build_memory(y, T, L, h)
    assert ends.max() + h <= T, "a value uses data after the forecast origin (leak!)"
    assert np.isclose(V[-1], y[T] - y[T - h]), "last value should be y[T] - y[T-h]"


check("TODO 1 softmax", t_softmax)
check("TODO 2 attention", t_attention)
check("TODO 3 topk_attention", t_topk)
check("TODO 4 compress_kv", t_compress)
check("scaffold: causal memory", t_causal)

for name, ok, msg in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  ->  {msg}"))
n_ok = sum(ok for _, ok, _ in results)
print(f"\n{n_ok}/{len(results)} passed")
sys.exit(0 if n_ok == len(results) else 1)
