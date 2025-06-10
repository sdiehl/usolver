Optimize a distillation column design to minimize total annual cost while meeting product purity specifications. Use HiGHS solver.

**Feed:** 

* 100 kmol/h Component A (light, volatile)
* 60 kmol/h Component B (heavy, less volatile)
* 80°C, 1.5 bar

The design variables to optimize are:

1. **Number_of_Trays** (N): Integer between 10-50 trays
2. **Reflux_Ratio** (RR): Real number between 0.5-5.0
3. **Feed_Tray_Location** (NF): Integer between 5 and min(45, N-5) (tray number from bottom)
4. **Column_Diameter** (D): Real number between 0.8-3.0 meters

The constraints are:

1. Product Purity Requirements

- Distillate purity of Component A: x_A_dist ≥ 0.95 (at least 95% pure)
- Bottoms purity of Component B: x_B_bot ≥ 0.90 (at least 90% pure)

2. Operating Constraints

- Maximum pressure drop across column: ΔP ≤ 5.0 kPa
- Minimum tray efficiency: η ≥ 0.60
- Maximum vapor velocity (flooding constraint): v_vapor ≤ 2.5 m/s
- Feed tray location must be between 5 and min(45, N-5) (not too close to ends)

3. Mass Balance Constraints

- Total material balance: F = D + B (Feed = Distillate + Bottoms)
- Component A balance: F × z_A = D × x_A_dist + B × x_A_bot
- Component B balance: F × z_B = D × x_B_dist + B × x_B_bot
- Feed composition: z_A = 100/160 = 0.625, z_B = 60/160 = 0.375
- Reflux relationship: L = RR × D (where L is liquid reflux)

4. Physical Constraints

- Minimum reflux ratio constraint: RR ≥ RR_min (typically 1.2 × minimum theoretical)
- Column diameter sizing based on vapor flow: D ≥ sqrt(4×V_vapor/(π×v_max×ρ_vapor)) 
- Tray spacing and column height: H = N × tray_spacing (assume 0.6 m per tray)
- Vapor density approximation: ρ_vapor ≈ 2.0 kg/m³ (at operating conditions)

5. Utility Constraints

- Maximum steam flow for reboiler: Steam ≤ 500 kg/h
- Maximum cooling water flow for condenser: CW ≤ 10,000 kg/h

The objective function is to minimize total annual cost:

Total_Cost = Capital_Cost + Operating_Cost

Where:
- **Capital_Cost** = Column_Cost + Tray_Cost
  - Column_Cost = 5000 × H ($/m of height)
  - Tray_Cost = 1500 × N ($/tray)

- **Operating_Cost** = Steam_Cost + Cooling_Water_Cost
  - Steam_Cost = 0.02 × Steam_Flow × 8760 ($/year)
  - Cooling_Water_Cost = 0.001 × CW_Flow × 8760 ($/year)

For this binary distillation optimization, use these approximations:

1. **Minimum trays** (Fenske equation): N_min = log((x_A_dist/(1-x_A_dist)) × ((1-x_A_bot)/x_A_bot)) / log(α_AB)
   - Assume relative volatility α_AB = 2.5 (Component A relative to Component B)

2. **Minimum reflux ratio** (Underwood method): RR_min = (x_A_dist - x_A_feed)/(x_A_feed - x_A_bot)
   - Where x_A_feed = z_A = 0.625 (liquid composition of A in feed)

3. **Vapor flow rate**: V = D × (RR + 1) (kmol/h)

4. **Steam requirement**: Steam ≈ 2.0 × B × MW_avg (kg/h, where MW_avg ≈ 50 kg/kmol)

5. **Cooling water requirement**: CW ≈ 12 × D × MW_avg (kg/h)

6. **Component recovery constraints**:
   - Recovery of A in distillate: R_A = (D × x_A_dist)/(F × z_A) ≥ 0.90
   - Recovery of B in bottoms: R_B = (B × x_B_bot)/(F × z_B) ≥ 0.85

Solve this optimization problem and provide:

1. Optimal values for all design variables
2. Total annual cost breakdown
3. Key performance metrics (recovery rates, energy consumption)

As markdown tables.