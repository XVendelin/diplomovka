function state_new = trackedRobotDynamicsWithTerrainNorm(state, M, dt, mapD, res)
% trackedRobotDynamicsWithTerrainNorm
% Dynamický model 4-pásového robota s NORMALIZOVANOU hĺbkovou mapou
%
% state = [x; y; theta; v; omega; z; zdot]
% M     = [M_FL; M_FR; M_RL; M_RR]   [N*m]
% dt    = časový krok [s]
% mapD  = normalizovaná hĺbková mapa ∈ [0,1]
% res   = rozlíšenie mapy [m/cell]

%% === PARAMETRE ROBOTA ===
m         = 40.0;      % [kg]
Jz        = 3.0;       % [kg*m^2]
y_offset  = 0.25;      % [m]
r_s       = 0.06;      % [m]
mu        = 0.7;       % [-]
eta_drive = 0.9;       % [-]
b_v       = 6.0;       % [N/(m/s)]
b_omega   = 0.3;       % [N*m*s/rad]
g         = 9.80665;   % [m/s^2]
L         = 0.45;      % [m]

% Odpruženie
k_s       = 10000.0;
c_s       = 200.0;
z_eq      = 0.20;
z_cmd     = 0.22;
act_kp    = 8000.0;
act_kd    = 300.0;

% Terén (NORMALIZOVANÝ)
alpha_max = deg2rad(30);   % max. ekvivalentný sklon
occ_th    = 1;          % nepriechodný terén

%% === ROZBALENIE STAVU ===
x     = state(1);
y     = state(2);
theta = state(3);
v     = state(4);
omega = state(5);
z     = state(6);
zdot  = state(7);

%% === TERÉN – NORMALIZOVANÁ MAPA ===
x_f = x + (L/2) * cos(theta);
y_f = y + (L/2) * sin(theta);
x_b = x - (L/2) * cos(theta);
y_b = y - (L/2) * sin(theta);

ix_f = max(1, min(round(x_f/res)+1, size(mapD,2)));
iy_f = max(1, min(round(y_f/res)+1, size(mapD,1)));
ix_b = max(1, min(round(x_b/res)+1, size(mapD,2)));
iy_b = max(1, min(round(y_b/res)+1, size(mapD,1)));

d_f = mapD(iy_f, ix_f);   % ∈ [0,1]
d_b = mapD(iy_b, ix_b);

% Nepriechodný terén → robot stojí
if d_f > occ_th || d_b > occ_th
    state_new = state;
    return;
end

% Virtuálny sklon z hĺbkovej mapy
alpha = alpha_max * (d_f - d_b);
alpha = max(min(alpha, alpha_max), -alpha_max);

%% === PREPOČET KRÚTIACEHO MOMENTU NA SILU ===
F_tang = (M ./ r_s) * eta_drive;

N_total = m * g * cos(alpha);
N_per_track = N_total / 4;
F_max = mu * N_per_track;

F = max(min(F_tang, F_max), -F_max);

F_FL = F(1); F_FR = F(2);
F_RL = F(3); F_RR = F(4);

F_L = F_FL + F_RL;
F_R = F_FR + F_RR;

%% === POZDĹŽNA A ROTAČNÁ DYNAMIKA ===
F_slope = m * g * sin(alpha);
F_x = F_L + F_R - F_slope;

Mz = (F_R - F_L) * y_offset;

% Zvýšený odpor na zlom teréne
terrain_drag = 1 + 3 * max(d_f, d_b);

a_long = (F_x - terrain_drag * b_v * v) / m;
a_yaw  = (Mz - b_omega * omega) / Jz;

%% === DYNAMIKA SVETLEJ VÝŠKY ===
F_spring = -k_s * (z - z_eq);
F_damp   = -c_s * zdot;
F_active = act_kp * (z_cmd - z) - act_kd * zdot;

Fz_total = F_spring + F_damp + F_active - m * g;
zddot = Fz_total / m;

%% === INTEGRÁCIA RÝCHLOSTÍ ===
v_new     = v     + a_long * dt;
omega_new = omega + a_yaw  * dt;
zdot_new  = zdot  + zddot  * dt;

%% === INTEGRÁCIA POLOHY ===
if abs(omega_new) < 1e-6
    x_new = x + v_new * dt * cos(theta);
    y_new = y + v_new * dt * sin(theta);
    theta_new = theta;
else
    theta_new = theta + omega_new * dt;
    x_new = x + (v_new / omega_new) * (sin(theta_new) - sin(theta));
    y_new = y - (v_new / omega_new) * (cos(theta_new) - cos(theta));
end

z_new = z + zdot_new * dt;
theta_new = mod(theta_new + pi, 2*pi) - pi;

%% === NOVÝ STAV ===
state_new = [x_new; y_new; theta_new; v_new; omega_new; z_new; zdot_new];

end
