"""Forward + backward time per training step of each layer in the language skeleton (MPS if present)."""
import sys, time, torch
from layers import Cfg
from net import Net, Model
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
B, T, D, H = int(sys.argv[1]) if len(sys.argv) > 1 else 64, 128, 256, 4
for name in ("SM", "GATED", "SIGMOID", "PG", "KM", "KMe"):
    cfg = Cfg(model="KM" if name.startswith("KM") else name, max_len=T, km_explicit=2 if name == "KMe" else 0)
    m = Model(Net(vocab=8004, n_out=8004, d_model=D, n_heads=H, n_layers=2, d_ff=2 * D, pe="sinusoidal" if name == "PG" else "none"), cfg).to(dev)
    tok = torch.randint(1, 8004, (B, T), device=dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    def step():
        loss = torch.nn.functional.cross_entropy(m(tok).flatten(0, 1), tok.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(3): step()
    if dev.type == "mps": torch.mps.synchronize()
    t0 = time.time(); n = 10
    for _ in range(n): step()
    if dev.type == "mps": torch.mps.synchronize()
    print(f"{name:8s} {1000 * (time.time() - t0) / n:7.1f} ms/step   params {m.n_params():,} (attention {m.attn_params():,})", flush=True)
