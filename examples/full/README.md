# Distillation Column Optimization Example

Example of using usolver to optimize a distillation column design end to end with Claude API.

## Problem Statement

There is a toy specification for a separation plant that takes a mixed feed stream containing two different chemicals (Component A and Component B) and separates them into two purified products. A very simplified version of the design behind a whiskey distillery or oil refinery, we start with a crude mixture and end up with separate, purified products. In this case, we're turning a binary mixture into two pure streams as output. The plant uses heat and cooling to exploit the fact that different chemicals boil at different temperatures, allowing for clean separation.

1. **Feed Input**: A binary mixture enters the system at 80°C
   - Component A (light, volatile): 100 kmol/h
   - Component B (heavy, less volatile): 60 kmol/h
   - Total feed: 160 kmol/h at 1.5 bar

2. **Distillation Column**: Separates the mixture by taking advantage of different boiling points:
   - **Lighter component** (A) rises to the top as vapor
   - **Heavier component** (B) sinks to the bottom as liquid

3. **Two Main Outputs**:
   - **Top Product (Distillate)**: A-rich product (97.9% pure A) - collected in Product Tank 1
   - **Bottom Product**: B-rich product (92.1% pure B) - collected in Product Tank 2

4. **Equipment**:
   - **Condenser**: Cools the top vapor back to liquid using cooling water
   - **Reboiler**: Heats the bottom liquid to create vapor using steam
   - **Cooling Tower**: Recycles cooling water
   - **Steam Generator**: Provides heat energy

The goal is to find the optimal design configuration that minimizes the total annual cost while meeting strict product purity requirements. We need to determine the optimal number of trays, reflux ratio, feed tray location, and column diameter that will achieve at least 95% purity for Component A in the distillate and 90% purity for Component B in the bottoms product.

The optimization must balance capital costs (equipment sizing) against operating costs (steam and cooling water consumption) while satisfying physical constraints of pressure drop limits, tray efficiency requirements, and flooding prevention.

The optimization determines:

1. **Number of Trays (N)**: 10-50 trays
2. **Reflux Ratio (RR)**: 0.5-5.0 
3. **Feed Tray Location (NF)**: Tray number from bottom (5 to min(45, N-5))
4. **Column Diameter (D)**: 0.8-3.0 meters

With the following constraints:

- **Purity Requirements**: ≥95% A in distillate, ≥90% B in bottoms
- **Recovery Requirements**: ≥90% A recovery, ≥85% B recovery
- **Operating Limits**: Pressure drop ≤5.0 kPa, vapor velocity ≤2.5 m/s
- **Utility Limits**: Steam ≤500 kg/h, cooling water ≤10,000 kg/h

## Process Flowsheet

```mermaid
flowchart TD
    FEED[["Feed<br/>Component A: 100 kmol/h<br/>Component B: 60 kmol/h<br/>80°C, 1.5 bar"]]
    DISTILLATION_COL[["Distillation Column"]]
    CONDENSER[["Condenser"]]
    REBOILER[["Reboiler"]]
    STEAM_GENERATOR[["Steam Generator"]]
    COOLING_TOWER[["Cooling Tower"]]
    PRODUCT_TANK_1[["Product Tank 1<br/>(A-rich Distillate)"]]
    PRODUCT_TANK_2[["Product Tank 2<br/>(B-rich Bottoms)"]]

    FEED -->|"S1: Feed Stream<br/>Binary Mixture"| DISTILLATION_COL
    DISTILLATION_COL -->|"S2: Distillate<br/>A: 95 kmol/h, B: 2 kmol/h<br/>65°C"| CONDENSER
    DISTILLATION_COL -->|"S3: Bottoms<br/>A: 5 kmol/h, B: 58 kmol/h<br/>120°C"| REBOILER
    CONDENSER -->|"S4: Reflux<br/>40°C"| DISTILLATION_COL
    CONDENSER -->|"S5: Net Distillate<br/>A-rich Product"| PRODUCT_TANK_1
    REBOILER -->|"S7: Final Bottoms<br/>B-rich Product<br/>110°C"| PRODUCT_TANK_2
    REBOILER -->|"S6: Reboiler Vapor"| DISTILLATION_COL
    STEAM_GENERATOR -->|"S8: Steam<br/>150°C"| REBOILER
    COOLING_TOWER -->|"S9: Cooling Water<br/>25°C"| CONDENSER
    CONDENSER -->|"S10: Heated Water<br/>40°C"| COOLING_TOWER
```

## Usage

1. **Start MCP Server**
   ```bash
   uv run mcp run usolver_mcp/server/main.py
   ```

2. **Set API Key and Run Client**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   uv run --with anthropic examples/full/client.py
   ``` 