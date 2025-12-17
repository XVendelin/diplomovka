function state_new = trackedRobotDynamics(state, M, dt)

% trackedRobotDynamics - dynamicky model 4-pásového robota
% state = [x; y; theta; v; omega; z; zdot]
% M = [M_FL; M_FR; M_RL; M_RR]  [N*m]
% dt - časový krok [s]
%
% výstup: state_new (rovnaký tvar)

%% === VOLITEĽNÉ PARAMETRE (default hodnoty) ===
m         = 40.0;      % [kg] hmotnosť robota
Jz        = 3.0;       % [kg*m^2] moment zotrvačnosti okolo vertikálnej osi
y_offset  = 0.25;      % [m] polovica rozchodu pásov
r_s       = 0.06;      % [m] polomer hnacieho kolesa pásu
mu        = 0.7;       % [-] koeficient trenia
eta_drive = 0.9;       % [-] účinnosť prenosu pohonu
b_v       = 6.0;       % [N/(m/s)] odpor pohybu
b_omega   = 0.2;       % [N*m*s/rad] rotačné tlmenie
g         = 9.80665;   % [m/s^2]

% parametre aktívnej zmeny svetlej výšky
k_s       = 10000.0;   % [N/m] tuhosť pruženia
c_s       = 200.0;     % [N*s/m] tlmenie
z_eq      = 0.20;      % [m] rovnovážna svetlá výška
z_cmd     = 0.22;      % [m] požadovaná svetlá výška (aktívne nastavenie)
act_kp    = 8000.0;    % [N/m] zisk pre riadenie výšky
act_kd    = 300.0;     % [N*s/m] zisk pre tlmenie výšky

%% === ROZBALENIE STAVU ===
x     = state(1);
y     = state(2);
theta = state(3);
v     = state(4);
omega = state(5);
z     = state(6);
zdot  = state(7);

%% === PREPOČET KRÚTIACEHO MOMENTU NA TRAKČNÚ SILU ===
F_tang = (M ./ r_s) * eta_drive;     % [N]
N_total = m * g;
N_per_track = N_total / 4;
F_max = mu * N_per_track;            % limit trením
F = max(min(F_tang, F_max), -F_max); % saturácia

F_FL = F(1); F_FR = F(2); F_RL = F(3); F_RR = F(4);
F_L = F_FL + F_RL;
F_R = F_FR + F_RR;

%% === DYNAMIKA POZDĹŽNA A ROTAČNÁ ===
F_x = F_L + F_R;
Mz  = (F_R - F_L) * y_offset;

a_long  = (F_x - b_v * v) / m;
a_yaw   = (Mz - b_omega * omega) / Jz;

%% === DYNAMIKA SVETLEJ VÝŠKY ===
% aktívne riadenie výšky okolo rovnovážneho bodu
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

%% === ZLOŽENIE NOVÉHO STAVU ===
state_new = [x_new; y_new; theta_new; v_new; omega_new; z_new; zdot_new];
end