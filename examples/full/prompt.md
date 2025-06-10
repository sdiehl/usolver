Optimize a distillation column design to minimize total annual cost while meeting product purity specifications.

**Feed:** 

* 100 kmol/h Component 1 (light)
* 50 kmol/h Component 2 (intermediate)
* 10 kmol/h Component 3 (heavy)
* 80°C, 1.5 bar

The design variables to optimize are:

1. **Number_of_Trays** (N): Integer between 10-50 trays
2. **Reflux_Ratio** (RR): Real number between 0.5-5.0
3. **Feed_Tray_Location** (NF): Integer between 5-45 (tray number from bottom)
4. **Column_Diameter** (D): Real number between 0.8-3.0 meters

The constraints are:

1. Product Purity Requirements

- Distillate purity of Component 1: x₁_dist ≥ 0.95 (at least 95% pure)
- Bottoms purity of Component 3: x₃_bot ≥ 0.90 (at least 90% pure)

2. Operating Constraints

- Maximum pressure drop across column: ΔP ≤ 5.0 kPa
- Minimum tray efficiency: η ≥ 0.60
- Maximum vapor velocity (flooding constraint): v_vapor ≤ 2.5 m/s
- Feed tray location must be between 5 and N-5 (not too close to ends)

3. Mass Balance Constraints

- Total material balance: F = D + B (Feed = Distillate + Bottoms)
- Component balances for each component
- Reflux relationship: L = RR × D (where L is liquid reflux)

4. Physical Constraints

- Minimum reflux ratio constraint: RR ≥ RR_min (typically 1.2 × minimum theoretical)
- Column diameter sizing based on vapor flow: D ≥ sqrt(4×V_vapor/(π×v_max))
- Tray spacing and column height: H = N × tray_spacing (assume 0.6 m per tray)

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

For this optimization, use these approximations:

1. **Minimum trays** (Fenske equation): N_min = log(x₁_dist/x₁_bot × x₃_bot/x₃_dist) / log(α_avg)
   - Assume average relative volatility α_avg = 2.5

2. **Minimum reflux ratio**: RR_min = (x₁_dist - y₁_feed)/(y₁_feed - x₁_dist)
   - Assume y₁_feed ≈ 0.7 (vapor composition at feed tray)

3. **Vapor flow rate**: V = D × (RR + 1)

4. **Steam requirement**: Steam ≈ 2.5 × B (approximate reboiler duty)

5. **Cooling water requirement**: CW ≈ 15 × D (approximate condenser duty)

Solve this optimization problem and provide:

1. Optimal values for all design variables
2. Verification that all constraints are satisfied
3. Total annual cost breakdown
4. Key performance metrics (recovery rates, energy consumption) 