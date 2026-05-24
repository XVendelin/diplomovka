# diplomovka

![Demo](agent_test.gif)

## Popis projektu
Tento projekt obsahuje MATLAB implementácie pre tréning robotického navigačného agenta pomocou reinforcement learning metód:

- `td3_zmena_krit_f.m` – tréning s TD3 agentom
- `PPO.m` – tréning s PPO agentom
- `LSTM_zmena_krit_f.m` – tréning s SAC agentom

Skript používa vlastné prostredie na základe `rlFunctionEnv`, lokálnu detekciu terénu a vlastný reward shaping.

## Štruktúra súborov

- `td3_zmena_krit_f.m` – definícia prostredia, TD3 agent, tréningové nastavenia a testovanie naučeného agenta.
- `PPO.m` – definícia prostredia, PPO agent, tréningové nastavenia a testovanie naučeného agenta.
- `LSTM_zmena_krit_f.m` – definícia prostredia, SAC agent, tréningové nastavenia a testovanie naučeného agenta.
- `druhy.m` – pomocný skript na načítanie a spracovanie mapy z obrázka.
- `kinematika_MR/` – knižnica s dynamikou traktora a ďalšími podpornými funkciami.
- `image.jpg`, `extraction.png`, `map2.png` – vstupné mapy a obrázky pre testovanie.

## Ako spustiť tréning

1. Otvorte MATLAB a nastavte pracovný adresár na `c:\Users\vendo\Desktop\temp\diplomovka`.
2. Uistite sa, že sú pridané potrebné cesty:
   ```matlab
   addpath("kinematika_MR");
   ```
3. Spustite požadovaný tréningový skript:
   - `td3_zmena_krit_f.m` pre TD3
   - `PPO.m` pre PPO
   - `LSTM_zmena_krit_f.m` pre SAC

Každý skript vytvorí vlastné prostredie, nastaví pozorovanie a akčné priestory, a následne spustí tréning.

## Tréningové nastavenia

- Mapy a trasy sú definované v skriptoch pomocou `routes1` a `routes2`.
- Pozorovanie obsahuje:
  - `sin(theta)` a `cos(theta)`
  - smer k cieľu
  - normovanú vzdialenosť k cieľu
  - rýchlosti lineárne a uhlové
  - lokálny terén v okolí robota
- Akcie sú motorické krútiace momenty pre 4 pohony s rozsahom `[-10, 10]`.
- Reward funkcia kombinuje progres k cieľu, penalizácie za nárazy do terénu a efektívnosť smerovania.

## Výstupy a uložené agenty

- Po tréningu sa vygenerujú súbory v adresároch `savedAgents_TD3`, `savedAgents_PPO` alebo `savedAgents`.
- Každý skript vyberie najlepší uložený agent podľa odmeny a uloží ho do `bestAgent.mat`.

## Testovanie naučeného agenta

Po tréningu sa spustí testovacia sekvencia, ktorá:

- načíta naučeného agenta
- použije greedy stratégiu pre rozhodovanie
- zobrazuje trajektóriu, akčné signály a lokálny terén

## Poznámky

- Predpokladá sa, že MATLAB Reinforcement Learning Toolbox je dostupný.
- Pre rýchlejšie tréningy sa využíva GPU, ak je dostupný.
- Skripty obsahujú aj alternatívne mapy a testovacie konfigurácie, ktoré sú momentálne zakomentované.
