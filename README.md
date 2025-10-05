# 🧩 JAX DQN on Jumanji's Game2048-v1

This repository demonstrates a **Deep Q-Network (DQN)** agent implemented entirely in **JAX**, trained on the [`Game2048-v1`](https://github.com/instadeepai/jumanji/tree/main/jumanji/environments/logic/game_2048) environment from **Jumanji**.

The project highlights how to combine:
- 🧠 **Jumanji** for clean, JIT-compatible environments  
- ⚡ **JAX** for high-performance functional ML  
- 💾 **Flashbax** for efficient, device-friendly replay buffers  

All together, this forms a lightweight and elegant reinforcement learning pipeline.

---

## 🚀 Highlights

- 🧠 **DQN implementation in JAX** — fully differentiable and JIT-compiled  
- 🎮 **Jumanji’s Game2048-v1** — a clean, RL-compatible 2048 environment  
- ⚡ **Fast & reproducible** — powered by JAX transformations (`jit`, `vmap`)  

---

---

## 🧩 Environment Details

- **Environment ID:** `Game2048-v1`  
- **Provided by:** [Jumanji](https://github.com/instadeepai/jumanji)  
- **Action Space:**  
  `0 = up`, `1 = right`, `2 = down`, `3 = left`  
- **Observation:**  
  - `board`: `int32[4, 4]` (stores exponents of 2, 0 = empty)  
  - `action_mask`: `bool[4]` (valid moves)  
  - `step_count`: `int32`  
- **Reward:** Sum of merged tile values after each valid move.

---

