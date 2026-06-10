"""
hop.py — HOPPY v3: Control Cartesiano (Jacobiano Transpuesto) + GRF Bezier
Correcciones:
  1. Deteccion de contacto por FUERZAS NORMALES de MuJoCo (Rubrica 3.2),
     no por el pin de resorte (que dejaba de registrar contacto).
  2. Z_FOOT_D = -0.15 m (antes -0.22, casi extension total = singularidad).
  3. Compensacion del resorte paralelo de rodilla en vuelo.
  4. Emulacion de encoders (Rubrica 5.1): velocidad por derivacion numerica
     de qpos + filtro pasa-bajas discreto (NO usa qvel para el control).
"""

import argparse
import time
import csv
import math
from enum import Enum, auto
import numpy as np
import mujoco

HEADLESS = False  # se ajusta con --headless

# ─── 1. GEOMETRIA (MATLAB get_params.m) ──────────────────────────────────────
LH = 96e-3
LK = 154.5e-3
DK = 52e-3
L2 = float(np.hypot(DK, LK))          # 0.16302 m
KNEE_STIFFNESS = 0.643                # Nm/rad (resorte paralelo, XML)
TAU_MAX = 3.728

# ─── 2. VUELO (dyn_aerial.m) ─────────────────────────────────────────────────
KP_SW = np.diag([80.0, 80.0])
KD_SW = np.diag([2.0, 2.0])
KRH = 0.1
Z_FOOT_D = -0.20        # CORREGIDO: con cadera a ~0.20 m del suelo, -0.22
R_BOOM = 0.556          # dejaba la pierna en extension total (singular).

# ─── 3. APOYO (dyn_stance.m) ─────────────────────────────────────────────────
T_ST = 0.1
FX_BZ = np.array([0.0, 0.0,  23.0, 0.0, 0.0])
FZ_BZ = np.array([0.0, 30.0, 80.0, 10.0, 0.0])
KP_ST = 0.7
KD_ST = 0.7
Q_D_ST = np.array([np.pi / 3, -np.pi / 2])   # convencion MATLAB

KNEE_SIGN = -1.0   # MATLAB: flexion negativa | XML: flexion positiva

# ─── 4. FSM POR FUERZA DE CONTACTO ───────────────────────────────────────────
F_TD = 1.0          # [N] umbral de touchdown
F_LO = 0.1          # [N] umbral de lift-off (histeresis)
MIN_FLIGHT = 0.2   # [s]
MIN_STANCE = 0.1   # [s]


class Phase(Enum):
    FLIGHT = auto()
    STANCE = auto()


def foot_normal_force(model, data, foot_geom_ids):
    """Suma de fuerzas normales de contacto sobre los geoms del pie."""
    f6 = np.zeros(6)
    fn = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in foot_geom_ids or c.geom2 in foot_geom_ids:
            mujoco.mj_contactForce(model, data, i, f6)
            fn += abs(f6[0])          # componente normal en frame de contacto
    return fn


class ContactFSM:
    def __init__(self):
        self.phase = Phase.FLIGHT
        self.prev_phase = Phase.FLIGHT
        self.t_transition = 0.0
        self.n_touchdown = 0
        self.n_liftoff = 0

    @property
    def in_stance(self): return self.phase is Phase.STANCE
    @property
    def touchdown(self): return self.phase is Phase.STANCE and self.prev_phase is Phase.FLIGHT
    @property
    def liftoff(self):   return self.phase is Phase.FLIGHT and self.prev_phase is Phase.STANCE

    def update(self, t, fn):
        self.prev_phase = self.phase
        dt_phase = t - self.t_transition
        if self.phase is Phase.FLIGHT:
            if fn >= F_TD and dt_phase >= MIN_FLIGHT:
                self.phase = Phase.STANCE
                self.t_transition = t
        else:
            if fn <= F_LO and dt_phase >= MIN_STANCE:
                self.phase = Phase.FLIGHT
                self.t_transition = t
        if self.touchdown:
            self.n_touchdown += 1
            print(f"[FSM] v TOUCHDOWN #{self.n_touchdown:>3d}  t={t:.4f}s  Fn={fn:.1f}N")
        elif self.liftoff:
            self.n_liftoff += 1
            print(f"[FSM] ^ LIFT-OFF  #{self.n_liftoff:>3d}  t={t:.4f}s")


# ─── 5. EMULACION DE ENCODERS (Rubrica 5.1) ──────────────────────────────────
class EncoderVelocity:
    """dq = (q_k - q_{k-1})/dt  +  filtro pasa-bajas de 1er orden.
    alpha = dt / (dt + 1/(2*pi*fc));  fc = frecuencia de corte [Hz]."""

    def __init__(self, dt, fc=40.0, n=2):
        self.dt = dt
        self.alpha = dt / (dt + 1.0 / (2.0 * np.pi * fc))
        self.q_prev = None
        self.v_filt = np.zeros(n)

    def update(self, q):
        q = np.asarray(q, dtype=float)
        if self.q_prev is None:
            self.q_prev = q.copy()
            return self.v_filt
        v_raw = (q - self.q_prev) / self.dt
        self.q_prev = q.copy()
        self.v_filt = self.alpha * v_raw + (1.0 - self.alpha) * self.v_filt
        return self.v_filt


# ─── 6. CINEMATICA Y JACOBIANO (fcn_p_toe_HIP / fcn_J_toe_HIP) ───────────────
def fk_toe_hip(q_hip, q_knee):
    s3, c3 = np.sin(q_hip), np.cos(q_hip)
    s34, c34 = np.sin(q_hip + q_knee), np.cos(q_hip + q_knee)
    return np.array([LH * s3 + L2 * s34, -LH * c3 - L2 * c34])


def jac_toe_hip(q_hip, q_knee):
    c3, s3 = np.cos(q_hip), np.sin(q_hip)
    c34, s34 = np.cos(q_hip + q_knee), np.sin(q_hip + q_knee)
    return np.array([[LH * c3 + L2 * c34, L2 * c34],
                     [LH * s3 + L2 * s34, L2 * s34]])


def polyval_bz(alpha, s):
    n = len(alpha) - 1
    s = float(np.clip(s, 0.0, 1.0))
    return sum(a * math.comb(n, k) * s**k * (1 - s)**(n - k)
               for k, a in enumerate(alpha))


# ─── 7. CONTROLADORES (Rubrica 4.3 / 4.4) ────────────────────────────────────
def control_flight_cartesian(q, dq, yaw_rate):
    """tau = J^T [ Kp (p_d - p) - Kd (J dq) ] """
    vx = yaw_rate * R_BOOM
    p_d = np.array([KRH * vx, Z_FOOT_D])
    p = fk_toe_hip(*q)
    J = jac_toe_hip(*q)
    F = KP_SW @ (p_d - p) - KD_SW @ (J @ dq)
    return J.T @ F


def control_stance_grf(q, dq, t_in_stance):
    """tau = -J^T [Fx; Fz]_Bezier + PD articular suave   (Rubrica 4.4)"""
    s = t_in_stance / T_ST
    F = np.array([polyval_bz(FX_BZ, s), polyval_bz(FZ_BZ, s)])
    J = jac_toe_hip(*q)
    return -J.T @ F + KP_ST * (Q_D_ST - q) - KD_ST * dq


# ─── 8. BUCLE PRINCIPAL ──────────────────────────────────────────────────────
def run(xml_path, duration, log_file, headless=False):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    hip_dof = model.joint("hip").dofadr[0]
    knee_dof = model.joint("knee").dofadr[0]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_site")
    foot_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                  for g in ("foot_rubber", "foot_sensor_cylinder")}
    foot_geoms.discard(-1)

    # postura inicial: pierna flexionada en el aire
    data.qpos[hip_dof] = np.pi / 3
    data.qpos[knee_dof] = 0.9
    mujoco.mj_forward(model, data)

    fsm = ContactFSM()
    enc = EncoderVelocity(dt, fc=40.0, n=2)
    rows = []
    print(f"[HOPPY] dur={duration}s  Z_FOOT_D={Z_FOOT_D}  TD>{F_TD}N LO<{F_LO}N\n")

    def step_once():
        nonlocal rows
        mujoco.mj_step(model, data)

        # deteccion de contacto por fuerza normal (Rubrica 3.2)
        fn = foot_normal_force(model, data, foot_geoms)
        fsm.update(data.time, fn)
        t_in_stance = data.time - fsm.t_transition if fsm.in_stance else 0.0

        # sensores articulares (solo POSICIONES para el control, Rubrica 5.1)
        hip_q = data.sensor("hip_pos").data[0]
        knee_q = data.sensor("knee_pos").data[0]
        yaw_rate = data.sensor("gantry_rot_vel").data[0]

        # velocidad estimada: derivacion numerica + pasa-bajas
        dq_est_mu = enc.update([hip_q, knee_q])

        # convencion MATLAB
        q = np.array([hip_q, KNEE_SIGN * knee_q])
        dq = np.array([dq_est_mu[0], KNEE_SIGN * dq_est_mu[1]])

        if fsm.in_stance:
            tau = control_stance_grf(q, dq, t_in_stance)
        else:
            tau = control_flight_cartesian(q, dq, yaw_rate)

        # de vuelta a convencion MuJoCo + compensacion del resorte de rodilla
        tau_hip_mu = tau[0]
        tau_knee_mu = KNEE_SIGN * tau[1]
        if not fsm.in_stance:
            tau_knee_mu += KNEE_STIFFNESS * knee_q   # cancela el resorte en vuelo

        data.ctrl[0] = np.clip(tau_hip_mu, -TAU_MAX, TAU_MAX)
        data.ctrl[1] = np.clip(tau_knee_mu, -TAU_MAX, TAU_MAX)

        if int(round(data.time / dt)) % 5 == 0:
            p_cart = fk_toe_hip(*q)
            v_cart = jac_toe_hip(*q) @ dq
            rows.append({
                "t": round(data.time, 4),
                "phase": fsm.phase.name,
                "hip_pos": round(hip_q, 4),
                "knee_pos": round(knee_q, 4),
                "hip_vel_est": round(dq_est_mu[0], 4),
                "knee_vel_est": round(dq_est_mu[1], 4),
                "hip_vel_real": round(data.sensor("hip_vel").data[0], 4),
                "knee_vel_real": round(data.sensor("knee_vel").data[0], 4),
                "foot_x": round(p_cart[0], 4),
                "foot_z": round(p_cart[1], 4),
                "foot_vx": round(v_cart[0], 4),
                "foot_vz": round(v_cart[1], 4),
                "foot_z_world": round(data.site_xpos[site_id][2], 4),
                "tau_hip": round(float(data.ctrl[0]), 4),
                "tau_knee": round(float(data.ctrl[1]), 4),
                "Fn": round(fn, 2),
            })

    if headless:
        while data.time < duration:
            step_once()
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            wall_start = time.time()
            k = 0
            while viewer.is_running() and data.time < duration:
                step_once()
                k += 1
                ahead = data.time - (time.time() - wall_start)
                if ahead > 0:
                    time.sleep(ahead)
                if k % 16 == 0:
                    viewer.sync()

    print(f"\n[HOPPY] saltos: TD={fsm.n_touchdown}  LO={fsm.n_liftoff}")
    if rows:
        with open(log_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"[HOPPY] {len(rows)} muestras -> {log_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="hoppy.xml")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--log", default="hop_cartesiano_log.csv")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    run(args.xml, args.duration, args.log, headless=args.headless)