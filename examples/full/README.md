# Distillation Column Optimization Example

Example of using usolver to optimize a distillation column design end to end with Claude API.

## Process Flowsheet

```mermaid
flowchart TD
    FEED[["Feed<br/>C1: 100 kmol/h<br/>C2: 50 kmol/h<br/>C3: 10 kmol/h<br/>80°C, 1.5 bar"]]
    DISTILLATION_COL[["Distillation Column"]]
    CONDENSER[["Condenser"]]
    REBOILER[["Reboiler"]]
    STEAM_GENERATOR[["Steam Generator"]]
    COOLING_TOWER[["Cooling Tower"]]
    PRODUCT_TANK_1[["Product Tank 1<br/>(Distillate)"]]
    PRODUCT_TANK_2[["Product Tank 2<br/>(Bottoms)"]]

    FEED -->|"S1: Feed Stream"| DISTILLATION_COL
    DISTILLATION_COL -->|"S2: Distillate<br/>C1: 85 kmol/h<br/>65°C"| CONDENSER
    DISTILLATION_COL -->|"S3: Bottoms<br/>C3: 9.5 kmol/h<br/>120°C"| REBOILER
    CONDENSER -->|"S4: Reflux<br/>35°C"| DISTILLATION_COL
    CONDENSER -->|"S5: Condensate<br/>Product"| PRODUCT_TANK_1
    REBOILER -->|"S7: Final Bottoms<br/>95°C"| PRODUCT_TANK_2
    REBOILER -->|"S6: Vapor"| STEAM_GENERATOR
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