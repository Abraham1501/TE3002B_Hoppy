import argparse
import time
import csv
import math
from enum import Enum, auto
import numpy as np
import mujoco

HEADLESS = False  

OMEGA_NOLOAD = 223 * (2 * np.pi / 60)  
TAU_STALL    = 3.728                    

# ───  GEOMETRIA  ──────────────────────────────────────
LH = 96e-3
LK = 154.5e-3
DK = 52e-3
L2 = float(np.hypot(DK, LK))          
KNEE_STIFFNESS = 1.1              
TAU_MAX = 3.728

# ───  VUELO  ─────────────────────────────────────────────────
KP_SW = np.diag([150.0, 150.0])
KD_SW = np.diag([5, 5])
KRH = 0.18
Z_FOOT_D = -0.19
R_BOOM = 0.556

# ───  APOYO  ─────────────────────────────────────────────────
T_ST  = 0.068
FX_BZ = np.array([0.0, 0.0,  24.0, 0.0, 0.0])
FZ_BZ = np.array([0.0, 20.0, 100.0, 0.0, 0.0])
KP_ST = 2.5
KD_ST = 0.4
Q_D_ST = np.array([np.pi / 3, -np.pi / 2])  

KNEE_SIGN = -1.0  

# ─── FSM POR SENSOR ANALOGICO DE CONTACTO ─────────────────────────────────

_K_SPRING = 2.0    
_C_SPRING = 0.2    

F_TD = 0.001       
F_LO = 0.005       
MIN_FLIGHT = 0.2   
MIN_STANCE = 0.068   


def tau_saturate(tau_cmd, q_dot):
    tau_available = TAU_STALL * max(0.0, 1.0 - abs(q_dot) / OMEGA_NOLOAD)
    return np.clip(tau_cmd, -tau_available, tau_available)

def foot_sensor_force(data):
    q_s = data.sensor("foot_compression").data[0] 
    v_s = data.sensor("foot_spring_vel").data[0] 
    return _K_SPRING * max(0.0, -q_s) + _C_SPRING * max(0.0, -v_s)


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
            print(f"[FSM] v TOUCHDOWN #{self.n_touchdown:>3d}  t={t:.4f}s  Fn={fn:.3f}N")
        elif self.liftoff:
            self.n_liftoff += 1
            print(f"[FSM] ^ LIFT-OFF  #{self.n_liftoff:>3d}  t={t:.4f}s")


# ─── EMULACION DE ENCODERS ──────────────────────────────────
class EncoderVelocity:
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


# ─── CONTROLADORES  ────────────────────────────────────
def control_flight_cartesian(q, dq, yaw_rate):
    vx = yaw_rate * R_BOOM
    p_d = np.array([KRH * vx, Z_FOOT_D])
    p = fk_toe_hip(*q)
    J = jac_toe_hip(*q)
    F = KP_SW @ (p_d - p) - KD_SW @ (J @ dq)
    return J.T @ F


def control_stance_grf(q, dq, t_in_stance):
    s = t_in_stance / T_ST
    F = np.array([polyval_bz(FX_BZ, s), polyval_bz(FZ_BZ, s)])
    J = jac_toe_hip(*q)
    return -J.T @ F + KP_ST * (Q_D_ST - q) - KD_ST * dq


# ─── BUCLE PRINCIPAL ──────────────────────────────────────────────────────
def run(xml_path, duration, log_file, headless=False):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    hip_dof  = model.joint("hip").dofadr[0]
    knee_dof = model.joint("knee").dofadr[0]
    gantry_pitch_dof = model.joint("gantry_pitch").dofadr[0]
    site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_site")

    # postura inicial: pierna flexionada en el aire
    data.qpos[hip_dof]  = np.pi / 3
    data.qpos[knee_dof] = 0.9
    data.qpos[gantry_pitch_dof] = -0.22
    mujoco.mj_forward(model, data)

    fsm  = ContactFSM()
    enc  = EncoderVelocity(dt, fc=40.0, n=2)
    rows = []
    print(f"[HOPPY] dur={duration}s  Z_FOOT_D={Z_FOOT_D}"
          f"  TD>{F_TD}N  LO<{F_LO}N  (sensor analogico)\n")

    def step_once():
        nonlocal rows
        mujoco.mj_step(model, data)

        # ── deteccion de contacto por sensor analogico  ──────────
        fn = foot_sensor_force(data)
        fsm.update(data.time, fn)
        t_in_stance = data.time - fsm.t_transition if fsm.in_stance else 0.0

        # ── sensores articulares  ────────────────────────────────
        hip_q    = data.sensor("hip_pos").data[0]
        knee_q   = data.sensor("knee_pos").data[0]
        yaw_rate = data.sensor("gantry_rot_vel").data[0]

        # velocidad estimada
        dq_est_mu = enc.update([hip_q, knee_q])

        q  = np.array([hip_q,        KNEE_SIGN * knee_q])
        dq = np.array([dq_est_mu[0], KNEE_SIGN * dq_est_mu[1]])

        if fsm.in_stance:
            tau = control_stance_grf(q, dq, t_in_stance)
        else:
            tau = control_flight_cartesian(q, dq, yaw_rate)

        tau_hip_mu  = tau[0]
        tau_knee_mu = KNEE_SIGN * tau[1]
        if not fsm.in_stance:
            tau_knee_mu += KNEE_STIFFNESS * knee_q 

        hip_vel_real  = data.sensor("hip_vel").data[0]
        knee_vel_real = data.sensor("knee_vel").data[0]

        data.ctrl[0] = tau_saturate(tau_hip_mu,  hip_vel_real)
        data.ctrl[1] = tau_saturate(tau_knee_mu, knee_vel_real)

        if int(round(data.time / dt)) % 5 == 0:
            p_cart = fk_toe_hip(*q)
            v_cart = jac_toe_hip(*q) @ dq
            rows.append({
                "t":               round(data.time, 4),
                "phase":           fsm.phase.name,
                "hip_pos":         round(hip_q,           4),
                "knee_pos":        round(knee_q,           4),
                "hip_vel_est":     round(dq_est_mu[0],    4),
                "knee_vel_est":    round(dq_est_mu[1],    4),
                "hip_vel_real":    round(data.sensor("hip_vel").data[0],  4),
                "knee_vel_real":   round(data.sensor("knee_vel").data[0], 4),
                "foot_x":          round(p_cart[0],        4),
                "foot_z":          round(p_cart[1],        4),
                "foot_vx":         round(v_cart[0],        4),
                "foot_vz":         round(v_cart[1],        4),
                "foot_z_world":    round(data.site_xpos[site_id][2], 4),
                "tau_hip":         round(float(data.ctrl[0]), 4),
                "tau_knee":        round(float(data.ctrl[1]), 4),
                "foot_compression": round(data.sensor("foot_compression").data[0], 6),
                "foot_spring_vel":  round(data.sensor("foot_spring_vel").data[0],  4),
                "Fn_sensor":        round(fn, 4),
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
    ap.add_argument("--xml",      default="hoppy.xml")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--log",      default="hop_cartesiano_log.csv")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    run(args.xml, args.duration, args.log, headless=args.headless)