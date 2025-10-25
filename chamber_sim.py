import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Simulation core ---
def run_simulation(rh_set, dp_set, co2_source_rate, t_end,
                   kp_p, ki_p, kd_p, kp_rh, ki_rh, kd_rh,
                   V, filter_efficiency):

    # Constants
    R_air, R_w = 287.058, 461.5
    cp_air, h_vap = 1005.0, 2.45e6
    Ta, pa = 298.15, 101325.0
    RH_a, CO2_a = 0.50, 800e-6
    Cd, A_leak = 0.65, 1e-5

    def p_sat_tetens(T):
        return 610.94 * np.exp(17.625*(T-273.15)/(T-30.11))

    def mass_fraction_from_RH_T(RH, T, p):
        ps = p_sat_tetens(T)
        pw = RH * ps
        return (pw/p) * (R_air / R_w)

    def leak_mdot(p, pa, rho):
        dp = max(p - pa, 0.0)
        return Cd * A_leak * np.sqrt(max(2 * rho * dp, 0.0))

    def fan_mdot_from_speed(s, pa, Ta):
        Vdot_max = 0.01
        return s * Vdot_max * (pa/(R_air*Ta))

    def mist_mdot_from_cmd(u):
        return u * 1e-5

    class PID:
        def __init__(self, kp, ki, kd, umin=0.0, umax=1.0):
            self.kp, self.ki, self.kd = kp, ki, kd
            self.umin, self.umax = umin, umax
            self.i = 0.0
            self.prev_e = 0.0
        def step(self, e, dt):
            de = (e - self.prev_e)/dt
            u = self.kp*e + self.i + self.kd*de
            u_sat = np.clip(u, self.umin, self.umax)
            self.i += self.ki*e*dt + 0.5*(u_sat - u)
            self.prev_e = e
            return u_sat

    def first_order_meas(y_meas, y_true, tau, dt):
        alpha = 1.0 - np.exp(-dt/tau)
        return y_meas + alpha*(y_true - y_meas)

    # Initial state
    T = Ta
    m = (pa * V) / (R_air * T)
    xw = mass_fraction_from_RH_T(RH_a, Ta, pa)
    xc = 1.5e-3

    pid_p = PID(kp=kp_p, ki=ki_p, kd=kd_p)
    pid_rh = PID(kp=kp_rh, ki=ki_rh, kd=kd_rh)

    dt = 0.1
    N = int(t_end/dt)

    p_meas = (m * R_air * T) / V
    rh_meas = (xw * R_w / R_air) * ((m * R_air * T)/V) / p_sat_tetens(T)
    T_meas = T
    co2_meas = xc

    log = {"t":[], "p":[], "dp":[], "T":[], "RH":[], "xc":[], "fan":[], "mist":[]}

    for k in range(N):
        t = k*dt
        p = (m * R_air * T) / V
        rho = p / (R_air * T)
        ps = p_sat_tetens(T)
        pw = xw * (R_w / R_air) * p
        RH = min(pw/ps, 1.0)

        p_meas = first_order_meas(p_meas, p, tau=0.2, dt=dt)
        rh_meas = first_order_meas(rh_meas, RH, tau=3.0, dt=dt)
        T_meas  = first_order_meas(T_meas, T, tau=5.0, dt=dt)
        co2_meas= first_order_meas(co2_meas, xc, tau=4.0, dt=dt)

        dp_meas = p_meas - pa
        fan_cmd = pid_p.step(dp_set - dp_meas, dt)
        mdot_in = fan_mdot_from_speed(fan_cmd, pa, Ta) * filter_efficiency

        mist_cmd = pid_rh.step(rh_set - rh_meas, dt)
        mdot_mist = mist_mdot_from_cmd(mist_cmd)

        mdot_leak = leak_mdot(p, pa, rho)

        xw_a = mass_fraction_from_RH_T(RH_a, Ta, pa)
        xc_a = CO2_a
        dm = (mdot_in - mdot_leak) * dt
        m += dm

        dmw = (mdot_in * xw_a - mdot_leak * xw + mdot_mist) * dt
        mw = xw * (m - dm) + dmw
        pw = (mw/(m)) * (R_w/R_air) * p
        ps = p_sat_tetens(T)
        if pw > ps:
            mw = (ps/p) * (R_air/R_w) * m
        xw = max(mw / m, 1e-6)
        pw = xw * (R_w / R_air) * p
        RH = min(pw / ps, 1.0)

        dmc = (mdot_in * xc_a - mdot_leak * xc + co2_source_rate) * dt
        mc = xc * (m - dm) + dmc
        xc = mc / m

        Q_elec = 2.0
        dT = (mdot_in*cp_air*(Ta - T) - mdot_leak*cp_air*(T - Ta) + Q_elec - mdot_mist*h_vap) * dt / (m*cp_air)
        T += dT

        log["t"].append(t)
        log["p"].append(p)
        log["dp"].append(p - pa)
        log["T"].append(T)
        log["RH"].append(RH)
        log["xc"].append(xc)
        log["fan"].append(fan_cmd)
        log["mist"].append(mist_cmd)

    return log

# --- Streamlit UI ---
st.title("Interactive Chamber Simulation")

rh_set = st.slider("Target RH (%)", 50, 100, 90) / 100
dp_set = st.slider("Target Pressure Δ (Pa)", 0, 50, 15)
co2_source_rate = st.number_input("CO₂ Source Rate (kg/s)", value=2e-7, format="%.1e")
t_end = st.slider("Simulation Time (s)", 60, 3600, 1800)

st.sidebar.header("PID Gains")
kp_p = st.sidebar.number_input("Pressure Kp", value=2e-4, format="%.1e")
ki_p = st.sidebar.number_input("Pressure Ki", value=1e-3, format="%.1e")
kd_p = st.sidebar.number_input("Pressure Kd", value=5e-5, format="%.1e")
kp_rh = st.sidebar.number_input("Humidity Kp", value=3.0)
ki_rh = st.sidebar.number_input("Humidity Ki", value=0.5)
kd_rh = st.sidebar.number_input("Humidity Kd", value=0.1)

st.sidebar.header("Chamber Dimensions")
length_cm = st.sidebar.number_input("Length (cm)", value=35.0)
width_cm  = st.sidebar.number_input("Width (cm)", value=15.0)
height_cm = st.sidebar.number_input("Height (cm)", value=25.0)
volume_m3 = (length_cm / 100) * (width_cm / 100) * (height_cm / 100)
st.sidebar.write(f"Volume: {volume_m3:.5f} m³")

st.sidebar.header("Filter Type")
filter_type = st.sidebar.selectbox("Select Filter Type", ["HEPA", "Sünger", "Petek Baskı"])
filter_efficiency = {"HEPA": 0.75, "Sünger": 0.90, "Petek Baskı": 0.95}[filter_type]

if st.button("Run Simulation"):
    log = run_simulation(
        rh_set, dp_set, co2_source_rate, t_end,
        kp_p, ki_p, kd_p, kp_rh, ki_rh, kd_rh,
        volume_m3, filter_efficiency
    )

    t = np.array(log["t"])
    co2_ppm = np.array(log["xc"]) * 1e6

    st.subheader("CO₂ Concentration (ppm)")
    st.line_chart({"CO₂ (ppm)": co2_ppm})

    st.subheader("Relative Humidity (%)")
    st.line_chart({"RH": np.array(log["RH"]) * 100})

    st.subheader("Temperature (°C)")
    st.line_chart({"Temperature": np.array(log["T"]) - 273.15})

    st.subheader("Pressure Difference (Pa)")
    st.line_chart({"ΔP": log["dp"]})

    st.subheader("Fan Control Signal")
    st.line_chart({"Fan Command": log["fan"]})

    st.subheader("Mist Control Signal")
    st.line_chart({"Mist Command": log["mist"]})
