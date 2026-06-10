"""
hop.py  —  HOPPY v2: hop con FSM + push timeado desde touchdown
"""

import argparse
import time
import csv
from enum import Enum, auto

import numpy as np
import mujoco
import mujoco.viewer

# ─── FSM ─────────────────────────────────────────────────────────────────────

SPRING_K     = 50.0
COMPRESS_ON  = -0.0001   # más sensible: -0.1mm en lugar de -0.3mm
COMPRESS_OFF = -0.00005
DWELL_TIME   = 0.15
MIN_FLIGHT   = 0.05

class Phase(Enum):
    FLIGHT = auto()
    STANCE = auto()

class ContactFSM:
    def __init__(self):
        self.phase        = Phase.FLIGHT
        self.prev_phase   = Phase.FLIGHT
        self.t_transition = 0.0
        self.n_touchdown  = 0
        self.n_liftoff    = 0

    @property
    def in_stance(self):  return self.phase is Phase.STANCE
    @property
    def in_flight(self):  return self.phase is Phase.FLIGHT
    @property
    def touchdown(self):  return self.phase is Phase.STANCE and self.prev_phase is Phase.FLIGHT
    @property
    def liftoff(self):    return self.phase is Phase.FLIGHT and self.prev_phase is Phase.STANCE

    def update(self, data):
        self.prev_phase = self.phase
        x  = data.sensor("foot_compression").data[0]
        fn = SPRING_K * abs(x)
        if self.phase is Phase.FLIGHT:
            time_in_flight = data.time - self.t_transition
            if x <= COMPRESS_ON and time_in_flight >= MIN_FLIGHT:
                self.phase = Phase.STANCE
                self.t_transition = data.time
        else:
            if x > COMPRESS_OFF and (data.time - self.t_transition) >= DWELL_TIME:
                self.phase = Phase.FLIGHT
                self.t_transition = data.time
        if self.touchdown:
            self.n_touchdown += 1
            print(f"[FSM] ▼ TOUCHDOWN #{self.n_touchdown:>3d}  t={data.time:.4f}s  F={fn:.1f}N")
        elif self.liftoff:
            self.n_liftoff += 1
            print(f"[FSM] ▲ LIFT-OFF  #{self.n_liftoff:>3d}  t={data.time:.4f}s")
        return x, fn

# ─── parámetros ───────────────────────────────────────────────────────────────

TAU_MAX = 3.728
KNEE_STIFFNESS = 0.643   # Nm/rad  valor real del resorte

Q_HIP_FLEX  = 0.66
Q_KNEE_FLEX = 1.87
Q_HIP_EXT   = 0.0
Q_KNEE_EXT  = 0.0
a"""
hop.py — HOPPY v3: Control Cartesiano (Jacobiano transpuesto) + GRF Bézier.
Correcciones vs v2:
  (1) Compensación del resorte paralelo de rodilla (faltaba → equilibrio muerto)
  (2) Z_FOOT_D = -0.16 (aterrizar flexionado, no en singularidad)
  (3) FSM basada en fuerza normal real con umbrales físicos (N, no µm)
  (4) Stance dura hasta lift-off real; Bézier escalado a la duración del stance
"""

import argparse
import time
import csv
from enum import Enum, auto
import math
import numpy as np
import mujoco
import mujoco.viewer

# ─── 1. GEOMETRÍA (MATLAB get_params.m) ──────────────────────────────────────
LH = 96e-3
LK = 154.5e-3
DK = 52e-3
L2 = np.hypot(DK, LK)          # ≈ 0.16302 m ;  largo máx pierna = 0.259 m

# ─── 2. PARÁMETROS DE CONTROL ────────────────────────────────────────────────
# FLIGHT — PD Cartesiano
KP_SW = np.diag([150.0, 150.0])
KD_SW = np.diag([5.0, 5.0])
KRH   = 0.1
Z_FOOT_D = -0.16        # *** flexionado: deja ~0.10 m de carrera para empujar ***
R_BOOM   = 0.556

# STANCE — perfil Bézier de GRF
T_ST  = 0.25                                    # duración nominal del push [s]
FX_BZ = np.array([0.0, 0.0, -25.0, 0.0, 0.0])
FZ_BZ = np.array([0.0, 40.0, 130.0, 0.0, 0.0])  # pico ≈ 59 N (> mg) para saltar
KP_ST = 0.03
KD_ST = 0.08
Q_D_ST = np.array([np.pi/3, -np.pi/2])          # convención MATLAB (rodilla < 0)

# Resorte paralelo de rodilla (DEBE coincidir con stiffness del XML)
KNEE_STIFFNESS = 0.643          # Nm/rad, en convención MuJoCo

TAU_MAX = 3.728

# ─── 3. FSM POR FUERZA DE CONTACTO ───────────────────────────────────────────
F_TD = 5.0      # N — umbral de touchdown (≈15% del peso)
F_LO = 1.5      # N — umbral de lift-off (histéresis)
MIN_STANCE = 0.03
MIN_FLIGHT = 0.05

class Phase(Enum):
    FLIGHT = auto()
    STANCE = auto()

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

    def update(self, data, fn):
        """fn: fuerza normal real de contacto [N]."""
        self.prev_phase = self.phase
        dt_phase = data.time - self.t_transition

        if self.phase is Phase.FLIGHT:
            if fn >= F_TD and dt_phase >= MIN_FLIGHT:
                self.phase = Phase.STANCE
                self.t_transition = data.time
        else:
            if fn <= F_LO and dt_phase >= MIN_STANCE:
                self.phase = Phase.FLIGHT
                self.t_transition = data.time

        if self.touchdown:
            self.n_touchdown += 1
            print(f"[FSM] ▼ TOUCHDOWN #{self.n_touchdown:>3d}  t={data.time:.4f}s  Fn={fn:.1f}N")
        elif self.liftoff:
            self.n_liftoff += 1
            print(f"[FSM] ▲ LIFT-OFF  #{self.n_liftoff:>3d}  t={data.time:.4f}s")


def get_contact_force(model, data, foot_geom_id):
    """Suma de fuerza normal sobre el geom del pie usando contactos de MuJoCo."""
    f_total = 0.0
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == foot_geom_id or c.geom2 == foot_geom_id:
            mujoco.mj_contactForce(model, data, i, force)
            f_total += abs(force[0])     # componente normal en frame de contacto
    return f_total

# ─── 4. CINEMÁTICA + JACOBIANO (convención MATLAB) ───────────────────────────
def fk_toe_hip(q_hip, q_knee):
    s3, c3 = np.sin(q_hip), np.cos(q_hip)
    s34, c34 = np.sin(q_hip + q_knee), np.cos(q_hip + q_knee)
    return np.array([LH*s3 + L2*s34, -LH*c3 - L2*c34])

def jac_toe_hip(q_hip, q_knee):
    c3, s3 = np.cos(q_hip), np.sin(q_hip)
    c34, s34 = np.cos(q_hip + q_knee), np.sin(q_hip + q_knee)
    return np.array([[LH*c3 + L2*c34, L2*c34],
                     [LH*s3 + L2*s34, L2*s34]])

def polyval_bz(alpha, s):
    n = len(alpha) - 1
    s = float(np.clip(s, 0.0, 1.0))
    return sum(a * math.comb(n, k) * s**k * (1-s)**(n-k) for k, a in enumerate(alpha))

# ─── 5. CONTROLADORES ────────────────────────────────────────────────────────
def control_flight_cartesian(q, dq, yaw_rate):
    """tau = J^T [Kp (p_d - p) - Kd (J dq)]   (rúbrica 4.3)"""
    vx = yaw_rate * R_BOOM
    p_d = np.array([KRH * vx, Z_FOOT_D])
    p = fk_toe_hip(*q)
    J = jac_toe_hip(*q)
    F_virtual = KP_SW @ (p_d - p) - KD_SW @ (J @ dq)
    return J.T @ F_virtual

def control_stance_grf(q, dq, t_in_stance):
    """tau = -J^T [Fx; Fz] + PD articular suave   (rúbrica 4.4)"""
    s = t_in_stance / T_ST
    F = np.array([polyval_bz(FX_BZ, s), polyval_bz(FZ_BZ, s)])
    J = jac_toe_hip(*q)
    tau_ff = -J.T @ F
    tau_fb = KP_ST * (Q_D_ST - q) - KD_ST * dq
    return tau_ff + tau_fb

# ─── 6. BUCLE PRINCIPAL ──────────────────────────────────────────────────────
def run(xml_path, duration, log_file):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    hip_dof  = model.joint("hip").dofadr[0]
    knee_dof = model.joint("knee").dofadr[0]

    # *** AJUSTA el nombre del geom del pie según tu XML ***
    foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot")
    if foot_geom_id < 0:
        raise RuntimeError("Geom 'foot' no encontrado — cambia el nombre al de tu XML")

    # Postura inicial flexionada (cae y aterriza con carrera disponible)
    data.qpos[hip_dof]  = 0.5
    data.qpos[knee_dof] = 0.8
    mujoco.mj_forward(model, data)

    fsm = ContactFSM()
    KNEE_SIGN = -1.0    # MATLAB: flexión negativa / MuJoCo: positiva (verificar con XML)

    rows = []
    print(f"[HOPPY] Inicio. dur={duration}s  Z_d={Z_FOOT_D}  Fz_pico≈{polyval_bz(FZ_BZ,0.5):.0f}N\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()
        step = 0

        while viewer.is_running() and data.time < duration:
            mujoco.mj_step(model, data)
            step += 1

            # contacto real → FSM
            fn = get_contact_force(model, data, foot_geom_id)
            fsm.update(data, fn)
            t_in_stance = data.time - fsm.t_transition

            # sensores articulares (convención MuJoCo)
            hip_pos  = data.sensor("hip_pos").data[0]
            knee_pos = data.sensor("knee_pos").data[0]
            hip_vel  = data.sensor("hip_vel").data[0]
            knee_vel = data.sensor("knee_vel").data[0]
            yaw_rate = data.sensor("gantry_rot_vel").data[0]

            # → convención MATLAB
            q  = np.array([hip_pos,  knee_pos * KNEE_SIGN])
            dq = np.array([hip_vel,  knee_vel * KNEE_SIGN])

            if fsm.in_stance:
                tau = control_stance_grf(q, dq, t_in_stance)
            else:
                tau = control_flight_cartesian(q, dq, yaw_rate)

            # ← convención MuJoCo + COMPENSACIÓN DEL RESORTE PARALELO
            tau_hip_cmd  = tau[0]
            tau_knee_cmd = tau[1] * KNEE_SIGN + KNEE_STIFFNESS * knee_pos

            data.ctrl[0] = np.clip(tau_hip_cmd,  -TAU_MAX, TAU_MAX)
            data.ctrl[1] = np.clip(tau_knee_cmd, -TAU_MAX, TAU_MAX)

            if step % 5 == 0:
                p_cart = fk_toe_hip(*q)
                rows.append({
                    "t": round(data.time, 4),
                    "phase": fsm.phase.name,
                    "t_stance": round(t_in_stance if fsm.in_stance else 0, 4),
                    "hip_pos": round(hip_pos, 4),
                    "knee_pos": round(knee_pos, 4),
                    "foot_x": round(p_cart[0], 4),
                    "foot_z": round(p_cart[1], 4),
                    "tau_hip": round(data.ctrl[0], 4),
                    "tau_knee": round(data.ctrl[1], 4),
                    "Fn": round(fn, 2),
                })

            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)
            if step % 16 == 0:
                viewer.sync()

    if rows:
        with open(log_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n[HOPPY] {len(rows)} muestras → {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="hoppy.xml")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--log", default="hop_cartesiano_log.csv")
    args = parser.parse_args()
    run(args.xml, args.duration, args.log)
T_PUSH = 0.10   # s  duración del push desde touchdown

KP_FLEX = 30.0;  KD_FLEX = 2.0
KP_EXT  = 20.0;  KD_EXT  = 1.0

# ─── controladores ────────────────────────────────────────────────────────────

def control_flight(data, hip_pos, knee_pos, hip_vel, knee_vel):
    """PD hacia postura flexionada + compensación resorte."""
    spring_comp = KNEE_STIFFNESS * knee_pos
    tau_hip  = KP_FLEX * (Q_HIP_FLEX  - hip_pos)  - KD_FLEX * hip_vel
    tau_knee = KP_FLEX * (Q_KNEE_FLEX - knee_pos)  - KD_FLEX * knee_vel + spring_comp
    return np.clip([tau_hip, tau_knee], -TAU_MAX, TAU_MAX)


def control_stance(t_in_stance, hip_pos, knee_pos, hip_vel, knee_vel):
    """
    0 → T_PUSH  : extensión activa hacia (0,0)
    T_PUSH+     : vuelve a flexión (prepara siguiente salto)
    """
    spring_comp = KNEE_STIFFNESS * knee_pos
    if t_in_stance < T_PUSH:
        tau_hip  = KP_EXT * (Q_HIP_EXT  - hip_pos)  - KD_EXT * hip_vel
        tau_knee = KP_EXT * (Q_KNEE_EXT - knee_pos)  - KD_EXT * knee_vel + spring_comp
    else:
        tau_hip  = KP_FLEX * (Q_HIP_FLEX  - hip_pos)  - KD_FLEX * hip_vel
        tau_knee = KP_FLEX * (Q_KNEE_FLEX - knee_pos)  - KD_FLEX * knee_vel + spring_comp
    return np.clip([tau_hip, tau_knee], -TAU_MAX, TAU_MAX)

# ─── bucle principal ──────────────────────────────────────────────────────────

def run(xml_path, duration, log_file):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    hip_dof  = model.joint("hip").dofadr[0]
    knee_dof = model.joint("knee").dofadr[0]
    site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_site")

    data.qpos[hip_dof]  = Q_HIP_FLEX
    data.qpos[knee_dof] = Q_KNEE_FLEX
    mujoco.mj_forward(model, data)

    fsm = ContactFSM()
    dt  = model.opt.timestep

    print(f"[hop] T_PUSH={T_PUSH}s  KP_EXT={KP_EXT}  dur={duration}s\n")

    rows = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()
        step = 0

        while viewer.is_running() and data.time < duration:
            mujoco.mj_step(model, data)
            step += 1

            x, fn = fsm.update(data)

            hip_pos  = data.sensor("hip_pos").data[0]
            knee_pos = data.sensor("knee_pos").data[0]
            hip_vel  = data.sensor("hip_vel").data[0]
            knee_vel = data.sensor("knee_vel").data[0]

            t_in_stance = data.time - fsm.t_transition

            if fsm.in_stance:
                tau = control_stance(t_in_stance, hip_pos, knee_pos, hip_vel, knee_vel)
            else:
                tau = control_flight(data, hip_pos, knee_pos, hip_vel, knee_vel)

            data.ctrl[0] = tau[0]
            data.ctrl[1] = tau[1]

            if step % 5 == 0:
                rows.append({
                    "t":        round(data.time, 4),
                    "phase":    fsm.phase.name,
                    "t_stance": round(t_in_stance if fsm.in_stance else 0, 4),
                    "hip_pos":  round(hip_pos, 4),
                    "knee_pos": round(knee_pos, 4),
                    "tau_hip":  round(tau[0], 4),
                    "tau_knee": round(tau[1], 4),
                    "foot_z":   round(data.site_xpos[site_id][2], 4),
                    "compress": round(x * 1e3, 4),
                    "Fn":       round(fn, 2),
                })

            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead)
            if step % 16 == 0:
                viewer.sync()

    if rows:
        with open(log_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n[hop] {len(rows)} muestras → {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",      default="hoppy.xml")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--log",      default="hop_log.csv")
    args = parser.parse_args()
    run(args.xml, args.duration, args.log)