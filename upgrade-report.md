# 🏭 Factory Simulation Engine: Enhanced Edition

![version](https://img.shields.io/badge/version-2.0-blue)
![python](https://img.shields.io/badge/python-3.8+-green)
![simpy](https://img.shields.io/badge/simpy-4.0+-orange)

> A state-of-the-art discrete-event simulation of manufacturing systems with comprehensive queueing theory metrics and flexible termination conditions.

## 📋 Overview

This repository contains two versions of a factory production line simulation:
- **mainVersion.py**: The baseline simulation model
- **enhancedVersion.py**: An improved version with advanced features

Both simulations model a three-machine production line (M1, M2, M3) processing products with:
- Random interarrival times
- Variable processing times
- Machine failures and repairs
- Queue behavior

The enhanced version significantly improves the simulation's flexibility, realism, and analytical capabilities while maintaining the core structure of the original model.

## ✨ Key Enhancements

| Feature | mainVersion.py | enhancedVersion.py | Impact |
|---------|---------------|-------------------|--------|
| **Simulation Termination** | Fixed time (200 units) | Flexible: runs until all products are processed | Ensures complete processing and improves flexibility |
| **Number of Products** | Hardcoded (20) | Configurable (default 100) | Enables scalable, user-defined simulations |
| **Machine Failure Logic** | Machines fail anytime | Machines fail only when busy | More realistic failure model based on machine operation |
| **Machine State Tracking** | Basic (broken, queue length) | Enhanced (busy/idle states, idle time) | Precise state metrics supporting advanced analysis |
| **Queue Monitoring** | Event-based queue length | Continuous queue sampling | More accurate average queue length calculation |
| **Queueing Metrics** | Limited (utilization, downtime) | Comprehensive (probability of wait, queue length, etc.) | Deeper system analysis capabilities |
| **Interarrival Tracking** | None | Tracks all interarrival times | Enables key queueing theory metrics |
| **Active Products** | None | Tracks products in system | Enables precise termination and load monitoring |

## 🔍 Detailed Improvements

### 1. Flexible Simulation Termination

Instead of running for a fixed time period that might be too short or too long, the enhanced version runs until all specified products have been processed:

```python
# Enhanced version
def run_simulation(sim_time=None, total_products=100):
    # ...initialization...
    env.process(product_arrival(env, machines, total_products))
    
    if sim_time is None:
        estimated_arrival_time = 5 * total_products + 50
        env.run(until=estimated_arrival_time)
        while active_products > 0:
            env.run(until=env.now + 50)
            log(f"TIME: {env.now:.1f}, REMAINING PRODUCTS: {active_products}", env.now)
            if env.now > 10000:  # Safety timeout
                log("SIMULATION TIMEOUT", env.now)
                break
    else:
        env.run(until=sim_time)
    return machines, env.now
```

### 2. Configurable Number of Products

The enhanced version allows specifying the number of products to simulate:

```python
# Enhanced version
def product_arrival(env, machines, total_products=100):
    i = 0
    while i < total_products:
        yield env.timeout(interarrival_time())
        Product(env, f"Product-{i}", machines)
        i += 1
        if i % 10 == 0:
            log(f"GENERATED {i}/{total_products} products", env.now)
```

### 3. Enhanced Machine Failure Logic

Machines now only fail when they're actually being used:

```python
# Enhanced version
def break_machine(self):
    while True:
        yield self.env.timeout(failure_time())
        if not self.broken and self.busy:  # Only fail during operation
            self.broken = True
            # ...repair process...
            self.broken = False
```

### 4. Improved Machine State Tracking

The enhanced version tracks busy/idle states precisely:

```python
# Enhanced version
def update_busy_status(self, new_busy_state):
    current_time = self.env.now
    if self.busy and not new_busy_state:  # Becoming idle
        self.busy = False
    elif not self.busy and new_busy_state:  # Becoming busy
        self.total_idle_time += current_time - self.last_busy_state_change
        self.busy = True
    self.last_busy_state_change = current_time
```

### 5. Queue Monitoring Process

Continuous monitoring of queue lengths for more accurate metrics:

```python
# Enhanced version
def monitor_queue(self):
    while True:
        self.sum_queue_length += len(self.resource.queue)
        self.queue_length_samples += 1
        yield self.env.timeout(1)
```

### 6. Enhanced Queueing Theory Metrics

The enhanced version provides comprehensive queueing theory metrics:

```python
# Enhanced version - additional metrics
system_queueing_metrics = {
    'Average Waiting Time': total_system_wait / total_products if total_products > 0 else 0,
    'Probability of Wait': products_that_waited / total_products if total_products > 0 else 0,
    'Average Queue Length': sum(m.sum_queue_length for m in machines) / sum(m.queue_length_samples for m in machines) if sum(m.queue_length_samples for m in machines) > 0 else 0,
    'Average Service Time': product_df['Total Service Time'].mean(),
    'Average Interarrival Time': avg_interarrival_time,
    'Average Time in System': product_df['Lead Time'].mean(),
    'Server Utilization': 1 - (sum(m.total_idle_time for m in machines) / (sim_time * len(machines)))
}
```

### 7. Interarrival Time Tracking

The enhanced version tracks product arrivals to calculate key system metrics:

```python
# Enhanced version
def __init__(self, env, name, machines):
    # ...initialization...
    if len(arrival_times) > 0:
        interarrival_times.append(self.start_time - arrival_times[-1])
    arrival_times.append(self.start_time)
    # ...continue with processing...
```

### 8. Active Products Tracking

Tracks products in the system to enable precise simulation control:

```python
# Enhanced version
def __init__(self, env, name, machines):
    # ...initialization...
    global active_products
    active_products += 1
    # ...

def process(self):
    # ...processing code...
    global active_products
    active_products -= 1
```

## 💡 Benefits

The enhanced simulation provides several advantages:
- **Greater Realism**: Machines fail only during operation
- **Flexible Configuration**: Control product counts and simulation termination
- **Richer Analytics**: Comprehensive queueing theory metrics
- **Better Monitoring**: Continuous queue monitoring and state tracking
- **Precise Control**: Simulation terminates when all products complete processing

## 📊 Simulation Outputs

The enhanced version provides a rich set of outputs:
- Detailed event logs
- Product metrics (waiting times, processing times, lead times)
- Machine metrics (utilization, availability, downtime)
- System queueing metrics (probabilities, average times)
- Bottleneck identification

## 🚀 Getting Started

```python
from enhancedVersion import run_simulation, analyze_results

# Run simulation with 200 products
machines, sim_time = run_simulation(total_products=200)

# Analyze results
results = analyze_results(machines, sim_time)

# Access metrics
print(f"Average Lead Time: {results['system_metrics']['Average Lead Time']}")
print(f"System Efficiency: {results['system_metrics']['System Efficiency (%)']}")
print(f"Bottleneck: {results['bottleneck']}")
```

## 🔬 Conclusion

The enhancements in enhancedVersion.py transform a basic simulation into a powerful analytical tool for production line optimization. The improved metrics, flexible configuration, and realistic modeling make it suitable for complex manufacturing system analysis, bottleneck identification, and performance optimization.