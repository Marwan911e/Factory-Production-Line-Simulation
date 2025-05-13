
# Factory Production Line Simulation Report

## System Modeling
- **Entities**: Products arriving with exponential interarrival times (mean = 6 minutes).
- **Resources**: 3 machines with processing times [5, 7, 4] minutes (normal distribution, std=1).
- **Failures**: Machine failure probabilities [0.05, 0.03, 0.07], repair times follow Weibull distribution (scale=10, shape=1.5).
- **Routing**: Sequential processing through all machines.

## Simulation Table
Saved to 'simulation_table.csv'. Includes arrival, queue, processing, failure, repair, and departure events with machine status, queue lengths, and waiting times.

## Simulation Logic
- Products arrive, queue at each machine, get processed, and depart.
- Failures interrupt processing, followed by repair times.
- Events logged consistently with no skipped rules.

## Distributions Used
1. **Exponential**: Interarrival times (mean = 6 minutes).
2. **Normal**: Processing times (means = [5, 7, 4], std = 1).
3. **Weibull**: Repair times (scale = 10, shape = 1.5).

## Analysis
- **Throughput**: 64 products completed.
- **Average Waiting Time**: 5.90 minutes.
- **Downtime**: {'Machine_0': np.float64(37.09366913730771), 'Machine_1': np.float64(11.612978272382287), 'Machine_2': np.float64(52.4739911143917)}.
- **Bottleneck**: Machine_0 with average queue length 11.32.
- **Interpretation**: Machine 1 (if bottleneck) causes delays due to longer processing time (7 minutes) and moderate failure rate (0.03). Reducing processing time or failure rate could improve throughput.

## Visualizations
- **Downtime**: See 'downtime.png'.
- **Queue Lengths**: See 'queue_lengths.png'.

## Bonus
- Visualizations for downtime and queue lengths.
- Detailed simulation table with 20+ events.
