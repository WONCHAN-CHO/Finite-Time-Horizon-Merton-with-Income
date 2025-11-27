# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:22:29 2025

@author: WONCHAN
"""

import os, math, numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 메인 루틴: 하나의 T_max 에 대해 PG-DPO + PMP 수행
# ------------------------------------------------------------
def run_pgdpo_backward(T_max_target, k_anal, pi_anal):
    """
    입력:
        T_max_target: 최종 유한 수평선 (예: 10.0)

        k_anal: analytic 소비비율 κ*(z) (shape: (Nz,))
        pi_anal: analytic 투자비율 π*(z) (shape: (Nz,))

    출력:
        z_lin, t_lin, piP_grid, kP_grid, err_stats
          - z_lin: z = Y/X 축 (Nz,)
          - t_lin: t 축 (Nt,)
          - piP_grid: PMP 최적 투자비율 π(t,z) (Nt, Nz)
          - kP_grid : PMP 최적 소비비율 κ(t,z) (Nt, Nz)
          - err_stats: slice별 남은 만기별 정량 오차 딕셔너리
    """

    # -----------------------------
    # CONFIG
    # -----------------------------
    CFG = {
        "device": {"prefer_cuda_idx": 0, "seed": 1},

        "discount": {"kind": "exponential", "kappa": 0.5, "eta": 1.0, "delta": 0.1},

        "market": {
            "d": 1,
            "r": 0.03,
            "gamma": 4.0,
            "mu_ex": 0.01,
            "sigma": 0.1,
        },

        "income": {
            "y_min": 0.5, "y_max": 1.5,
            "y_init": 1.0,
            "mu_y": 0.01,
            "sigma_y": 0.2,
        },

        "simulation": {
            "T_max": float(T_max_target),
            "dt": 1.0 / 8.0,
            "W_min": 0.05,  # wealth가 너무 작지 않게만 컷
            "W_max": 20.0,
            "W_cap": 1e20,
            "lb_w": 1e-6,
        },

        "consumption": {
            "kappa_max": 200.0,
            "baseline_kappa": 0.01,
            "bequest_phi": 1.0,
        },

        "train": {
            "batch_n": 64,
            "iters_per_T": 40,
            "lr_policy": 5e-4,
            "lr_critic": 5e-4,
            "grad_clip": 100.0,
            "use_richardson": True,
            "repeats_costate": 64,
            "sub_batch_costate": 8,
            "actor_kappa_weight": 1.0,
            "critic_coef": 1.0,
            "hjb_coef": 1.0,
        },

        "eval": {
            "repeats": 128,
            "sub_batch": 16,
        },

        # z := Y/X (income-to-wealth ratio)
        "plot": {
            "out_dir": "jupyter_income_1asset_backward",
            "z_min": 0.05,
            "z_max": 1.5,
            "Nz": 33,
            "Nt": 33,
        },
    }

    # -----------------------------
    # Env / seed / device
    # -----------------------------
    torch.manual_seed(CFG["device"]["seed"])
    np.random.seed(CFG["device"]["seed"])

    if torch.cuda.is_available():
        dev = f'cuda:{min(CFG["device"]["prefer_cuda_idx"], torch.cuda.device_count()-1)}'
    else:
        dev = "cpu"
    print(f"[Device] {dev}")

    # ============================================================
    # (0) discount / utility
    # ============================================================

    def compute_kappa_pmp_safe(V_x, V_xx, x, k_min=0.0, k_max=5.0, eps=1e-6):
        x_safe = torch.nan_to_num(x, nan=eps, posinf=1e6, neginf=eps)
        x_safe = torch.clamp(x_safe, min=eps, max=1e6)

        V_x_safe = torch.nan_to_num(V_x, nan=0.0, posinf=1e6, neginf=-1e6)
        V_x_safe = torch.clamp(V_x_safe, min=-1e6, max=1e6)

        V_xx_safe = torch.nan_to_num(V_xx, nan=-eps, posinf=-eps, neginf=-1e6)
        V_xx_safe = torch.where(
            V_xx_safe > -eps,
            -eps * torch.ones_like(V_xx_safe),
            V_xx_safe
        )
        V_xx_safe = torch.clamp(V_xx_safe, min=-1e6, max=-eps)

        kappa = - V_x_safe / (V_xx_safe * x_safe)
        kappa = torch.nan_to_num(kappa, nan=0.0, posinf=k_max, neginf=k_min)
        kappa = torch.clamp(kappa, min=k_min, max=k_max)
        return kappa

    def compute_pi_pmp_safe(kappa, z, pi_min=-5.0, pi_max=5.0):
        z_safe = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)
        pi = kappa * z_safe
        pi = torch.nan_to_num(pi, nan=0.0, posinf=pi_max, neginf=pi_min)
        pi = torch.clamp(pi, min=pi_min, max=pi_max)
        return pi
    
    def safe_log(x, eps=1e-8):
        x = torch.nan_to_num(x, nan=eps, posinf=1e8, neginf=eps)
        return torch.log(torch.clamp(x, min=eps))

    def u_crra_safe(c, gamma, eps=1e-2):
        c = torch.nan_to_num(c, nan=eps, posinf=1e8, neginf=eps)
    
        c = torch.clamp(c, min=eps, max=1e8)

        if abs(float(gamma) - 1.0) < 1e-8:
            return torch.log(c)
        else:
            # (c^(1-gamma) - 1) / (1-gamma)
            return (torch.pow(c, 1.0 - gamma) - 1.0) / (1.0 - gamma)

    def discount_kernel_torch(t, spec):
        # spec: {"kind": "hyperbolic" or "exponential", "rho": ..., "beta": ...}
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)

        rho = float(spec.get("rho", 0.03))
        kind = spec.get("kind", "hyperbolic")

        t = torch.nan_to_num(t, nan=0.0, posinf=1e3, neginf=0.0)
        t = torch.clamp(t, min=0.0, max=1e3)

        if kind == "hyperbolic":
            return 1.0 / (1.0 + rho * t)
        else:  # exponential
            expo = -rho * t
            expo = torch.clamp(expo, min=-50.0, max=50.0)
            return torch.exp(expo)

    # ============================================================
    # (B) π/κ policy network  state=(τ,X,Y) -> (pi,kappa)
    # ============================================================
    def _sub_net(inp, out):
        net = nn.Sequential(
            nn.Linear(inp, 160),
            nn.LeakyReLU(),
            nn.Linear(160, 160),
            nn.LeakyReLU(),
            nn.Linear(160, 160),
            nn.LeakyReLU(),
            nn.Linear(160, out),
        )
        nn.init.normal_(net[-1].weight, std=1e-3)
        nn.init.zeros_(net[-1].bias)
        return net

    class PiKappaPolicy(nn.Module):
        def __init__(self, pi_myopic_scalar: float, kappa_max: float, hedge_amp: float = 1.0):
            super().__init__()
            self.pi_net = _sub_net(3, 1)
            self.pi_myopic = float(pi_myopic_scalar)
            self.hedge_amp = float(hedge_amp)
            self.kappa_head = _sub_net(3, 1)
            self.kappa_max = float(kappa_max)

        def forward(self, state):
            tau = state[:, 0:1]
            X = state[:, 1:2]
            Y = state[:, 2:3]
            z_input = torch.cat([tau, X, Y], dim=1)
            h_raw = self.pi_net(z_input)
            hedge_factor = 1.0 + self.hedge_amp * torch.tanh(h_raw)
            pi = self.pi_myopic * hedge_factor
            kappa = self.kappa_max * torch.sigmoid(self.kappa_head(z_input))
            return {"pi": pi, "kappa": kappa}

    # ============================================================
    # Critic: λ_x (costate) 네트워크
    # ============================================================
    class LambdaCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _sub_net(3, 1)

        def forward(self, state):
            return self.net(state)

    # ============================================================
    # (C) domain sampler — (T_fixed, X0, Y0)
    # ============================================================
    def uniform_domain_fixed_T(
        n, T_fixed, W_min, W_max, y_min, y_max, dt_fixed, dev, seed=None
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        T = torch.full((n, 1), float(T_fixed), device=dev)
        X0 = W_min + (W_max - W_min) * torch.rand(n, 1, device=dev)
        Y0 = y_min + (y_max - y_min) * torch.rand(n, 1, device=dev)
        dt = torch.full((n, 1), float(dt_fixed), device=dev)
        m_T = int(round(float(T_fixed) / float(dt_fixed)))
        m_T = max(m_T, 1)
        return T, X0, Y0, dt, m_T

# ============================================================
    # (D) Simulator — 1D X, GBM Y (Dimension Safe & NaN Fix)
    # ============================================================
    def _sim_once(
        model,
        pi_M,      # Accepted but ignored
        kappa_M,   # Accepted but ignored
        T,         # Tensor (B, 1)
        X0,
        Y0,
        dt,
        anti=1,
        seed=None,
        train=True,
        use_cv=False,
        # Keyword arguments passed by sim:
        r=None,
        mu_ex=None,
        sigma=None,
        mu_y=None,
        sigma_y=None,
        m=None,
        lb_w=None,
        W_cap=None,
        gamma=None,
        disc_spec=None,
        phi_bequest=0.0,
        **kwargs
    ):
        # 1. Setup Device and Constants
        device = X0.device
        n_steps = m
        B = X0.shape[0]

        # [수정 1] dt 처리: 텐서라면 무조건 (B, 1)로 변환하여 브로드캐스팅 방지
        if torch.is_tensor(dt):
            dt_torch = dt.clone().detach().to(dtype=torch.float32, device=device)
            dt_torch = dt_torch.reshape(-1, 1)  # 강제로 (N, 1) 혹은 (1, 1)
        else:
            # 스칼라 값인 경우
            dt_torch = torch.tensor(dt, dtype=torch.float32, device=device)

        # 2. Seeding
        if seed is not None:
            torch.manual_seed(seed)

        # 3. Parameter Safety (모두 텐서화)
        r_s       = r if torch.is_tensor(r) else torch.tensor(r, device=device)
        mu_ex_s   = mu_ex if torch.is_tensor(mu_ex) else torch.tensor(mu_ex, device=device)
        sigma_s   = sigma if torch.is_tensor(sigma) else torch.tensor(sigma, device=device)
        mu_y_s    = mu_y if torch.is_tensor(mu_y) else torch.tensor(mu_y, device=device)
        sigma_y_s = sigma_y if torch.is_tensor(sigma_y) else torch.tensor(sigma_y, device=device)

        # Initial State: (B, 1) 형태 보장
        X = X0.view(B, 1)
        X = torch.clamp(X, min=lb_w if lb_w else 1e-4, max=W_cap if W_cap else 1e6)
        Y = Y0.clone().view(B, 1)
        
        R_total = torch.zeros(B, dtype=torch.float32, device=device)

        # 4. Time Stepping
        for step in range(n_steps):
            # t_abs 계산: dt_torch가 (B,1)이거나 스칼라이므로 t_abs도 안전
            t_abs = dt_torch * step
            
            # 남은 시간 (tau) 계산: T(B,1) - t_abs
            # t_abs가 스칼라여도, (B,1)이어도 안전하게 연산됨
            tau_t = torch.clamp(T - t_abs, min=0.0)

            # Safety clamps
            X = torch.nan_to_num(X, nan=1e-4, posinf=1e6, neginf=1e-4)
            X = torch.clamp(X, min=lb_w if lb_w else 1e-4, max=W_cap if W_cap else 1e6)
            
            # Policy Network Input: (tau, X, Y) -> 모두 (B, 1) 형태여야 함
            state = torch.cat([tau_t.view(B, 1), X, Y], dim=1)
            
            pi_kappa = model.forward(state)
            pi_t     = pi_kappa["pi"]      # (B, 1)
            kappa_t  = pi_kappa["kappa"]   # (B, 1)

            # Brownian Motion 생성 시 (B, 1)로 생성
            eps_x = torch.randn(B, 1, dtype=torch.float32, device=device)
            eps_y = torch.randn(B, 1, dtype=torch.float32, device=device)
            
            if anti < 0:
                eps_x = -eps_x
                eps_y = -eps_y

            # dB 계산: 차원 안전함 ((B,1) * (B,1) -> (B,1))
            dBx = torch.sqrt(dt_torch) * eps_x
            dBy = torch.sqrt(dt_torch) * eps_y

            # Dynamics
            X_safe = torch.clamp(X, min=lb_w if lb_w else 1e-4)
            ratio_YX = Y / X_safe

            # Drift와 Diffusion 계산
            # 모든 항이 (B, 1) 또는 스칼라이므로 결과도 (B, 1) 유지
            drift_X = (r_s + pi_t * mu_ex_s + ratio_YX - kappa_t)
            var_X   = (pi_t ** 2) * (sigma_s ** 2)

            logX = safe_log(X_safe) + (drift_X - 0.5 * var_X) * dt_torch + pi_t * sigma_s * dBx
            X = torch.exp(logX)
            X = torch.clamp(X, min=lb_w if lb_w else 1e-4, max=W_cap if W_cap else 1e6)

            # Income Process
            Y = Y + Y * (mu_y_s * dt_torch + sigma_y_s * dBy)
            
            # Utility Calculation
            c_t = kappa_t * X
            
            # [Nan 해결 핵심] 소비량이 너무 작으면 CRRA 효용이 -무한대로 발산하므로 하한선 설정
            # Gamma=4일 때 1e-3 이하로 가면 기울기가 폭발합니다.
            c_t = torch.clamp(c_t, min=1e-3)

            U_c = u_crra_safe(c_t, gamma) # 결과는 (B, 1)
            
            # Discounting
            # discount_kernel_torch는 보통 1D 벡터를 입력으로 받으므로 flatten 해줌
            t_now_flat = t_abs.view(-1) 
            if t_now_flat.numel() == 1: # 스칼라일 경우 확장
                 t_now_flat = t_now_flat.expand(B)

            D_now = discount_kernel_torch(t_now_flat, disc_spec) # (B,)
            
            # D_now(B,) * U_c(B,1) * dt(B,1) -> broadcasting 주의
            # D_now를 (B, 1)로 변환하여 연산
            R_total = R_total + (D_now.view(B, 1) * U_c * dt_torch).view(B)

        return R_total, X.detach(), Y.detach()

    def sim(
        policy,
        pi_M,
        kappa_M,
        T,
        X0,
        Y0,
        dt,
        anti=1,
        seed=None,
        train=True,
        use_richardson=False,
        use_cv=False,
        *,
        r,
        mu_ex,
        sigma,
        mu_y,
        sigma_y,
        m,
        lb_w,
        W_cap,
        gamma,
        disc_spec,
        phi_bequest=0.0,
    ):
        # 1. Coarse Simulation
        out_c = _sim_once(
            policy, pi_M, kappa_M, T, X0, Y0, dt,
            anti=anti, seed=seed, train=train, use_cv=use_cv,
            r=r, mu_ex=mu_ex, sigma=sigma, mu_y=mu_y, sigma_y=sigma_y,
            m=m, lb_w=lb_w, W_cap=W_cap, gamma=gamma, disc_spec=disc_spec,
            phi_bequest=phi_bequest,
        )
        U_c_val = out_c[0] # Extract R_total

        if not use_richardson:
            return U_c_val

        # 2. Fine Simulation (Richardson Extrapolation)
        dt_fine = dt * 0.5
        m_fine = int(2 * m)

        out_f = _sim_once(
            policy, pi_M, kappa_M, T, X0, Y0, dt_fine,
            anti=anti, seed=seed, train=train, use_cv=use_cv,
            r=r, mu_ex=mu_ex, sigma=sigma, mu_y=mu_y, sigma_y=sigma_y,
            m=m_fine, lb_w=lb_w, W_cap=W_cap, gamma=gamma, disc_spec=disc_spec,
            phi_bequest=phi_bequest,
        )
        U_f_val = out_f[0] # Extract R_total

        # 3. Combine
        if not train:
            # Richardson step: 2 * Fine - Coarse
            U_R = 2.0 * U_f_val - U_c_val
            return U_R

        U_R_main = 2.0 * U_f_val - U_c_val
        return U_R_main, None

    # ============================================================
    # (E) Costates & PMP
    # ============================================================
    def estimate_costates(
        policy,
        pi_M,
        kappa_M,
        T0,
        X0,
        Y0,
        dt0,
        repeats,
        sub_batch,
        use_richardson,
        *,
        r,
        mu_ex,
        sigma,
        mu_y,
        sigma_y,
        m,
        lb_w,
        W_cap,
        gamma,
        disc_spec,
        phi_bequest,
    ):
        device = X0.device
        n_eval = X0.size(0)

        Xg = X0.detach().clone().requires_grad_(True)
        Yg = Y0.detach().clone().requires_grad_(True)

        U_sum = torch.zeros(n_eval, 1, device=device)
        lam_x_sum = torch.zeros_like(Xg)
        dxx_sum = torch.zeros_like(Xg)
        tot = 0

        for i in range(0, repeats, sub_batch):
            cur = min(sub_batch, repeats - i)

            Tb = T0.repeat(cur, 1)
            Xb = Xg.repeat(cur, 1)
            Yb = Yg.repeat(cur, 1)
            dtb = dt0.repeat(cur, 1)

            Up = sim(
                policy,
                pi_M,
                kappa_M,
                Tb,
                Xb,
                Yb,
                dtb,
                +1,
                seed=i,
                train=False,
                use_richardson=use_richardson,
                use_cv=False,
                r=r,
                mu_ex=mu_ex,
                sigma=sigma,
                mu_y=mu_y,
                sigma_y=sigma_y,
                m=m,
                lb_w=lb_w,
                W_cap=W_cap,
                gamma=gamma,
                disc_spec=disc_spec,
                phi_bequest=phi_bequest,
            )
            Un = sim(
                policy,
                pi_M,
                kappa_M,
                Tb,
                Xb,
                Yb,
                dtb,
                -1,
                seed=i,
                train=False,
                use_richardson=use_richardson,
                use_cv=False,
                r=r,
                mu_ex=mu_ex,
                sigma=sigma,
                mu_y=mu_y,
                sigma_y=sigma_y,
                m=m,
                lb_w=lb_w,
                W_cap=W_cap,
                gamma=gamma,
                disc_spec=disc_spec,
                phi_bequest=phi_bequest,
            )

            Umean = 0.5 * (Up + Un).view(cur, n_eval).mean(0)

            lam_x, lam_y = torch.autograd.grad(
                Umean.sum(), (Xg, Yg), create_graph=True, retain_graph=True
            )
            (dxx,) = torch.autograd.grad(lam_x.sum(), (Xg,))

            U_sum += Umean.unsqueeze(1).detach() * cur
            lam_x_sum += lam_x.detach() * cur
            dxx_sum += dxx.detach() * cur
            tot += cur

        inv = 1.0 / max(1, tot)
        return lam_x_sum * inv, dxx_sum * inv, (U_sum * inv).mean().item()

    def pmp_from_costates_1d(
        X,
        lam_x,
        dlam_x_dx,
        mu_ex,
        sigma,
        r,
        gamma,
        device="cpu",
        pi_clip=50.0,
        kappa_clip=2.0,
    ):
        X = torch.as_tensor(X, dtype=torch.float32, device=device).clamp_min(1e-4)
        la_raw = torch.as_tensor(lam_x, dtype=torch.float32, device=device)
        dX_raw = torch.as_tensor(dlam_x_dx, dtype=torch.float32, device=device)
    
        la_raw = torch.nan_to_num(la_raw, nan=1e-6, posinf=1e6, neginf=1e-6)
        dX_raw = torch.nan_to_num(dX_raw, nan=0.0, posinf=1e6, neginf=-1e6)

        mu_ex_s = float(mu_ex[0])
        sigma_s = float(sigma[0])

        la = la_raw.clamp_min(1e-6)

        # dX (Value function의 2계 미분과 관련)
        dX = -torch.abs(dX_raw)
    
        denom = X * dX
        denom = torch.clamp(denom, max=-1e-6)

        coeff = -1.0 / denom

        # 4) 총 투자 비중 π_tot
        pi_tot = coeff * la * mu_ex_s / (sigma_s**2)
        # 비중 클리핑
        pi_tot = torch.clamp(pi_tot, min=-pi_clip, max=pi_clip)

        # 5) 소비 비율 κ = λ^(-1/γ) / X
        kappa = la.pow(-1.0 / float(gamma)) / X
        kappa = torch.clamp(kappa, min=1e-4, max=kappa_clip) 

        return pi_tot, kappa

    # ============================================================
    # (F) Prepare params, networks, optimizers
    # ============================================================
    r = CFG["market"]["r"]
    gamma = CFG["market"]["gamma"]

    T_max_target_local = CFG["simulation"]["T_max"]
    dt_fixed = CFG["simulation"]["dt"]

    W_min = CFG["simulation"]["W_min"]
    W_max = CFG["simulation"]["W_max"]
    W_cap = CFG["simulation"]["W_cap"]
    lb_w = CFG["simulation"]["lb_w"]

    disc_spec = CFG["discount"]
    phi_bequest = CFG["consumption"]["bequest_phi"]

    y_min = CFG["income"]["y_min"]
    y_max = CFG["income"]["y_max"]
    mu_y = CFG["income"]["mu_y"]
    sigma_y = CFG["income"]["sigma_y"]

    mu_ex_val = CFG["market"]["mu_ex"]
    sigma_val = CFG["market"]["sigma"]

    mu_ex = torch.tensor([mu_ex_val], dtype=torch.float32, device=dev)
    sigma = torch.tensor([sigma_val], dtype=torch.float32, device=dev)

    Sigma = sigma.pow(2).reshape(1, 1)
    Sigma_inv = 1.0 / Sigma

    pi_star = (1.0 / gamma) * (Sigma_inv @ mu_ex.reshape(1, 1))
    pi_star_s = float(pi_star[0, 0])

    hedge_amp = 1.0

    policy = PiKappaPolicy(
        pi_myopic_scalar=pi_star_s,
        kappa_max=CFG["consumption"]["kappa_max"],
        hedge_amp=hedge_amp,
    ).to(dev)

    critic = LambdaCritic().to(dev)

    opt_policy = optim.Adam(policy.parameters(), lr=CFG["train"]["lr_policy"])
    opt_critic = optim.Adam(critic.parameters(), lr=CFG["train"]["lr_critic"])

    pi_M_const = torch.tensor([[pi_star_s]], device=dev)
    kappa_M_const = torch.full(
        (1, 1), float(CFG["consumption"]["baseline_kappa"]), device=dev
    )

    # ============================================================
    # (G) Training: Projected step + HJB residual
    # ============================================================
    def train_batch_pmp_hjb(T_fixed):
        T0, X0, Y0, dt0, m_T = uniform_domain_fixed_T(
            CFG["train"]["batch_n"],
            T_fixed,
            W_min,
            W_max,
            y_min,
            y_max,
            dt_fixed,
            dev,
        )

        X0 = X0.detach().clone().requires_grad_(True)
        Y0 = Y0.detach().clone().requires_grad_(True)

        lam_x, dxx, _ = estimate_costates(
            policy,
            pi_M_const,
            kappa_M_const,
            T0,
            X0,
            Y0,
            dt0,
            CFG["train"]["repeats_costate"],
            CFG["train"]["sub_batch_costate"],
            CFG["train"]["use_richardson"],
            r=r,
            mu_ex=mu_ex,
            sigma=sigma,
            mu_y=mu_y,
            sigma_y=sigma_y,
            m=m_T,
            lb_w=lb_w,
            W_cap=W_cap,
            gamma=gamma,
            disc_spec=disc_spec,
            phi_bequest=phi_bequest,
        )

        piP_row, kP_row = pmp_from_costates_1d(
            X0.cpu(),
            lam_x.cpu(),
            dxx.cpu(),
            mu_ex.cpu(),
            sigma.cpu(),
            r,
            gamma,
            device="cpu",
        )
        piP = piP_row.to(dev)
        kP = kP_row.to(dev)

        policy.train()
        critic.train()
        opt_policy.zero_grad()
        opt_critic.zero_grad()

        state = torch.cat([T0, X0, Y0], dim=1)
        out = policy(state)
        pi_theta = out["pi"]
        k_theta = out["kappa"]

        actor_loss = ((pi_theta - piP) ** 2).mean() + CFG["train"][
            "actor_kappa_weight"
        ] * ((k_theta - kP) ** 2).mean()

        lam_hat = critic(state)
        fit_loss = torch.mean((lam_hat - lam_x) ** 2)

        lam_hat_x = torch.autograd.grad(lam_hat.sum(), X0, create_graph=True)[0]

        mu_ex_s = float(mu_ex[0])
        sigma_s = float(sigma[0])

        drift_X = r + pi_theta * mu_ex_s + (Y0 / X0) - k_theta
        var_X = (pi_theta**2) * (sigma_s**2)

        hjb_res = (
            u_crra_safe((k_theta * X0).clamp_min(0.0), gamma)
            + lam_hat * drift_X
            + 0.5 * lam_hat_x * var_X
        )

        hjb_loss = torch.mean(hjb_res**2)

        critic_loss = fit_loss + CFG["train"]["hjb_coef"] * hjb_loss
        total_loss = actor_loss + CFG["train"]["critic_coef"] * critic_loss

        total_loss.backward()
        if CFG["train"]["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), CFG["train"]["grad_clip"]
            )
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), CFG["train"]["grad_clip"]
            )
        opt_policy.step()
        opt_critic.step()

        return actor_loss.item(), fit_loss.item(), hjb_loss.item()

    T_list = []
    Tmax_int = max(1, int(math.floor(T_max_target_local)))
    for T in range(1, Tmax_int + 1):
        T_list.append(float(T))
    if T_max_target_local not in T_list:
        T_list.append(float(T_max_target_local))
    T_list = sorted(set([T for T in T_list if T <= T_max_target_local + 1e-8]))
    print(f"[Train] horizon schedule T_list = {T_list}")

    for T_fixed in T_list:
        print(f"\n[Horizon] Training for remaining horizon T={T_fixed}")
        iters = CFG["train"]["iters_per_T"]
        for i in range(iters):
            aL, cL, hL = train_batch_pmp_hjb(T_fixed)
            if (i + 1) % 10 == 0:
                print(
                    f"[T={T_fixed:4.1f}, {i+1:04d}] "
                    f"Actor={aL: .3e}, CritFit={cL: .3e}, HJB={hL: .3e}"
                )

    # ============================================================
    # (H) Evaluation & Plots — (t, z=Y/X) grid
    # ============================================================
    @torch.no_grad()
    def make_tz_grid(T_eval, z_min, z_max, Nz, Nt):
        t_lin = np.array([0.95, 0.9, 0.7, 0.5, 0.0]) * (T_eval)
        z_lin = np.linspace(z_min, z_max, Nz)  # z = Y/X
        T_grid, Z_grid = np.meshgrid(t_lin, z_lin, indexing="ij")
        return t_lin, z_lin, T_grid, Z_grid

    def draw_grid(z_lin, t_lin, Zval, title, fname, outdir, cmap="jet"):
        plt.figure(figsize=(6.6, 5.2))
        pcm = plt.pcolormesh(z_lin, t_lin, Zval, shading="auto", cmap=cmap)
        plt.xlabel("z = Y/X")
        plt.ylabel("t")
        plt.title(title)
        cb = plt.colorbar(pcm)
        cb.set_label("value")
        cb.formatter.set_useOffset(False)
        cb.formatter.set_scientific(False)
        plt.tight_layout()
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close()

    def eval_pmp_on_tz_grid(
        policy,
        T_eval,
        t_lin,
        Z_grid,
        y_ref,
        repeats,
        sub_batch,
        use_richardson,
        *,
        r,
        mu_ex,
        sigma,
        mu_y,
        sigma_y,
        dt_fixed,
        lb_w,
        W_cap,
        gamma,
        disc_spec,
        phi_bequest,
        pi_M_const,
        kappa_M_const,
        dev,
        batch_pts=4096,
    ):
        Nt = len(t_lin)
        Nz = Z_grid.shape[1]

        piP_grid = np.zeros((Nt, Nz), dtype=np.float32)
        kP_grid = np.zeros((Nt, Nz), dtype=np.float32)

        for i_t, t_val in enumerate(t_lin):
            tau_i = float(T_eval - t_val)
            if tau_i <= 0.0:
                piP_grid[i_t, :] = 0.0
                kP_grid[i_t, :] = 0.0
                continue

            m_tau = int(round(tau_i / dt_fixed))
            m_tau = max(m_tau, 1)

            # z = Y / X  =>  X = Y / z
            z_row = Z_grid[i_t, :]
            Y0_flat = np.full_like(z_row, y_ref)
            X0_flat = Y0_flat / z_row

            T0 = torch.full((Nz, 1), tau_i, dtype=torch.float32, device=dev)
            X0 = torch.as_tensor(
                X0_flat.reshape(-1, 1), dtype=torch.float32, device=dev
            )
            Y0 = torch.as_tensor(
                Y0_flat.reshape(-1, 1), dtype=torch.float32, device=dev
            )
            dt0 = torch.full((Nz, 1), dt_fixed, dtype=torch.float32, device=dev)

            lam_x, dxx, _ = estimate_costates(
                policy,
                pi_M_const,
                kappa_M_const,
                T0,
                X0,
                Y0,
                dt0,
                repeats,
                sub_batch,
                use_richardson,
                r=r,
                mu_ex=mu_ex,
                sigma=sigma,
                mu_y=mu_y,
                sigma_y=sigma_y,
                m=m_tau,
                lb_w=lb_w,
                W_cap=W_cap,
                gamma=gamma,
                disc_spec=disc_spec,
                phi_bequest=phi_bequest,
            )
            piP_row, kP_row = pmp_from_costates_1d(
                X0.cpu(),
                lam_x.cpu(),
                dxx.cpu(),
                mu_ex.cpu(),
                sigma.cpu(),
                r,
                gamma,
                device="cpu",
            )
            piP_grid[i_t, :] = piP_row.numpy().reshape(-1)
            kP_grid[i_t, :] = kP_row.numpy().reshape(-1)

        return piP_grid, kP_grid

    OUTDIR = CFG["plot"]["out_dir"]
    T_eval = float(T_max_target_local)
    t_lin, z_lin, T_grid, Z_grid = make_tz_grid(
        T_eval,
        CFG["plot"]["z_min"],
        CFG["plot"]["z_max"],
        CFG["plot"]["Nz"],
        CFG["plot"]["Nt"],
    )
    y_ref = 1.0

    policy.eval()
    piP_grid, kP_grid = eval_pmp_on_tz_grid(
        policy,
        T_eval,
        t_lin,
        Z_grid,
        y_ref=y_ref,
        repeats=CFG["eval"]["repeats"],
        sub_batch=CFG["eval"]["sub_batch"],
        use_richardson=CFG["train"]["use_richardson"],
        r=r,
        mu_ex=mu_ex,
        sigma=sigma,
        mu_y=mu_y,
        sigma_y=sigma_y,
        dt_fixed=dt_fixed,
        lb_w=lb_w,
        W_cap=W_cap,
        gamma=gamma,
        disc_spec=disc_spec,
        phi_bequest=phi_bequest,
        pi_M_const=pi_M_const,
        kappa_M_const=kappa_M_const,
        dev=dev,
        batch_pts=4096,
    )

    draw_grid(z_lin, t_lin, piP_grid, "PMP π(t, z)", "PMP_pi_tz.png", OUTDIR)
    draw_grid(z_lin, t_lin, kP_grid, "PMP κ(t, z)", "PMP_kappa_tz.png", OUTDIR)

    # --------------------------------------------------
    # t-slice 단면 + analytic 곡선 + 정량 오차
    # --------------------------------------------------
    def plot_slices_pmp_only(
        z_lin, t_lin, piP, kP, T_max_target, outdir, k_anal, pi_anal
    ):
        z = z_lin
        os.makedirs(outdir, exist_ok=True)
        li = t_lin

        base_color = "C0"
        alphas = np.linspace(0.2, 1.0, len(li))

        mae_pi, rmse_pi = [], []
        mae_k, rmse_k = [], []
        TmT_list = []

        # π slices
        plt.figure(figsize=(6.4, 4.0))
        plt.xlabel("z = Y/X")
        plt.ylabel("π")
        plt.title("π(z) slices (PMP vs analytic)")

        for i, t0 in enumerate(li):
            idx = int(np.argmin(np.abs(t_lin - t0)))
            plt.plot(
                z,
                piP[idx, :],
                color=base_color,
                alpha=alphas[i],
                label=f"T_max - t ≈ {T_max_target - t_lin[idx]:.2f}",
            )
            resi = piP[idx, :] - pi_anal
            mae_t = np.mean(np.abs(resi))
            rmse_t = np.sqrt(np.mean(resi**2))
            mae_pi.append(mae_t)
            rmse_pi.append(rmse_t)
            TmT_list.append(float(T_max_target - t_lin[idx]))

        plt.plot(
            z,
            pi_anal,
            color="k",
            linewidth=2.5,
            linestyle="--",
            label=r"analytic, $T_{\max}\to\infty$",
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "pi_slices_pmp_vs_analytic.png"), dpi=200)
        plt.close()

        # κ slices
        plt.figure(figsize=(6.4, 4.0))
        plt.xlabel("z = Y/X")
        plt.ylabel("κ")
        plt.title("κ(z) slices (PMP vs analytic)")

        for i, t0 in enumerate(li):
            idx = int(np.argmin(np.abs(t_lin - t0)))
            plt.plot(
                z,
                kP[idx, :],
                color=base_color,
                alpha=alphas[i],
                label=f"T_max - t ≈ {T_max_target - t_lin[idx]:.2f}",
            )
            resi = kP[idx, :] - k_anal
            mae_t = np.mean(np.abs(resi))
            rmse_t = np.sqrt(np.mean(resi**2))
            mae_k.append(mae_t)
            rmse_k.append(rmse_t)

        plt.plot(
            z,
            k_anal,
            color="k",
            linewidth=2.5,
            linestyle="--",
            label=r"analytic, $T_{\max}\to\infty$",
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "k_slices_pmp_vs_analytic.png"), dpi=200)
        plt.close()

        # 에러 vs 남은 horizon
        TmT_arr = np.array(TmT_list)
        mae_pi = np.array(mae_pi)
        rmse_pi = np.array(rmse_pi)
        mae_k = np.array(mae_k)
        rmse_k = np.array(rmse_k)

        order = np.argsort(TmT_arr)
        TmT_arr = TmT_arr[order]
        mae_pi = mae_pi[order]
        rmse_pi = rmse_pi[order]
        mae_k = mae_k[order]
        rmse_k = rmse_k[order]

        plt.figure(figsize=(6.4, 4.0))
        plt.plot(TmT_arr, mae_pi, label="MAE(π)")
        plt.plot(TmT_arr, rmse_pi, label="RMSE(π)", linestyle="--")
        plt.xlabel(r"Remaining horizon $T_{\max} - t$")
        plt.ylabel("Error")
        plt.title("π error vs remaining horizon")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "pi_error_vs_Tmax.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(6.4, 4.0))
        plt.plot(TmT_arr, mae_k, label="MAE(κ)")
        plt.plot(TmT_arr, rmse_k, label="RMSE(κ)", linestyle="--")
        plt.xlabel(r"Remaining horizon $T_{\max} - t$")
        plt.ylabel("Error")
        plt.title("κ error vs remaining horizon")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "k_error_vs_Tmax.png"), dpi=200)
        plt.close()

        err_stats = {
            "TmT": TmT_arr,
            "mae_pi": mae_pi,
            "rmse_pi": rmse_pi,
            "mae_k": mae_k,
            "rmse_k": rmse_k,
        }
        return err_stats

    err_stats = plot_slices_pmp_only(
        z_lin, t_lin, piP_grid, kP_grid, T_max_target_local, OUTDIR, k_anal, pi_anal
    )

    print("\n[Error table] Remaining horizon vs MAE/RMSE (π, κ)")
    print(" T_rem  |  MAE_pi   RMSE_pi   MAE_k    RMSE_k")
    for T_rem, e1, e2, e3, e4 in zip(
        err_stats["TmT"],
        err_stats["mae_pi"],
        err_stats["rmse_pi"],
        err_stats["mae_k"],
        err_stats["rmse_k"],
    ):
        print(f"{T_rem:6.3f} | {e1:7.3e} {e2:7.3e} {e3:7.3e} {e4:7.3e}")

    return z_lin, t_lin, piP_grid, kP_grid, err_stats


# ------------------------------------------------------------
# 여러 T_max 에 대해 반복 실험
# ------------------------------------------------------------
def run_experiments_over_T(T_list, k_anal, pi_anal):
    results = {}
    for T_sup in T_list:
        print("\n" + "=" * 72)
        print(f"[Main] Start PG-DPO backward experiment: T_max = {T_sup}")
        z_lin, t_lin, piP_grid, kP_grid, err_stats = run_pgdpo_backward(
            T_sup, k_anal, pi_anal
        )
        results[T_sup] = err_stats
        print(f"[Main] Finished T_max={T_sup}")
    return results


# ------------------------------------------------------------
# Analytic curves (infinite-horizon, z = Y/X)
# ------------------------------------------------------------
def build_analytic_curves_example(
    z_min=0.05,
    z_max=1.5,
    Nz=33,
    *,
    r=0.03,
    mu_ex=0.01,  # mu_ex = mu - r (초과수익률)
    mu_y=0.02,  # 소득 성장률
    sigma=0.1,
    gamma=4.0,
    beta=0.04,
):
    # z = Y/X  (income-to-wealth ratio)
    z = np.linspace(z_min, z_max, Nz)

    theta = mu_ex / sigma
    pi_M = mu_ex / (gamma * sigma**2)  # no-income Merton 비중

    # no-income Merton 소비비율 k_M
    k_M = (beta - (1.0 - gamma) * (r + (theta**2) / (2.0 * gamma))) / gamma

    # Corollary 3.1 의 factor: (1 + z / (r - mu_y)), z = Y/X
    factor = 1.0 + z / (r - mu_y)

    pi_anal = pi_M * factor  # π*(z)
    k_anal = k_M * factor  # κ*(z)

    return z, k_anal, pi_anal


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    print("[Global] Start PG-DPO backward experiments")

    # analytic curves on the same z-grid as plot CFG
    z_ref, k_anal, pi_anal = build_analytic_curves_example(
        z_min=0.05, z_max=1.5, Nz=33
    )

    T_list = [5.0, 10.0, 20.0]
    results = run_experiments_over_T(T_list, k_anal, pi_anal)

    print("\n[Summary by T_max]")
    for T_sup in T_list:
        es = results[T_sup]
        T_rem0 = es["TmT"][0]
        print(
            f"T_max={T_sup:5.1f}, first slice T_rem={T_rem0:5.2f}, "
            f"MAE_pi={es['mae_pi'][0]:.3e}, MAE_k={es['mae_k'][0]:.3e}"
        )

    print(
        "\n[Global] All done. Figures saved in 'jupyter_income_1asset_backward/'"
    )