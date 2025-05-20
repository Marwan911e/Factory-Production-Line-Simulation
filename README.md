# SimuLine: Factory Production Line Simulation 🏭

![Production Line](https://img.shields.io/badge/Production-Simulation-blue)
![Python](https://img.shields.io/badge/Python-3.6+-green)
![SimPy](https://img.shields.io/badge/SimPy-4.0+-orange)
![Last Updated](https://img.shields.io/badge/Last%20Updated-May%202025-brightgreen)

## 📋 Factory System Overview

### The Production Line
Our simulation models a factory with a linear production line consisting of three machines (M1, M2, M3) arranged in sequence. Each product must pass through these machines in order:

```
[Products Enter] → [Machine 1] → [Machine 2] → [Machine 3] → [Products Exit]
```

### Key Characteristics of Our Factory

1. **Single-Product Flow**: Each machine can process only one product at a time.
2. **Sequential Processing**: Products must visit M1, then M2, then M3.
3. **Variable Processing Times**: Each machine takes a different amount of time to process products.
4. **Machine Failures**: Machines can break down randomly during operation.
5. **Repair Process**: When a machine breaks, it must be repaired before resuming operation.
6. **Queueing System**: Products wait in line when a machine is busy or broken.

### Machine Characteristics
- **Machine 1 (M1)**: Uses normal distribution for processing times (mean=8, std=1.5)
- **Machine 2 (M2)**: Uses triangular distribution for processing times (min=6, mode=8, max=10)
- **Machine 3 (M3)**: Uses gamma distribution for processing times (shape=2, scale=3)

### Factory Operating Rules
1. Products arrive according to an exponential distribution (mean=5 time units between arrivals)
2. Machines break down according to a Weibull distribution (shape=2, scale=20)
3. Repairs take a random amount of time following an exponential distribution (mean=10)
4. When a machine breaks while processing a product, the processing pauses until repair is complete
5. Products are processed in a First-Come-First-Served (FCFS) manner

## 🔄 Simulation Flow Explained

The simulation progresses through these main stages:

### 1️⃣ Initialization
- Set up the simulation environment
- Create three machines (M1, M2, M3)
- Initialize data tracking variables for events, products, and machine states

### 2️⃣ Product Generation
- Start generating products at random intervals
- Each new product enters the system and begins its journey through the machines

### 3️⃣ Processing Flow
For each product:
1. **Arrival**: Product enters the system
2. **Queue at Machine 1**: Wait if M1 is busy
3. **Processing at Machine 1**: Product is processed by M1
4. **Queue at Machine 2**: Wait if M2 is busy
5. **Processing at Machine 2**: Product is processed by M2
6. **Queue at Machine 3**: Wait if M3 is busy
7. **Processing at Machine 3**: Product is processed by M3
8. **Completion**: Product exits the system

### 4️⃣ Parallel Events
While products are flowing through the system:
- Machines occasionally break down randomly
- Broken machines get repaired after some time
- Products wait if their next machine is busy or broken
- Statistics are collected on waiting times, processing times, etc.

### 5️⃣ Simulation Completion
- Simulation runs until all products have completed processing
- Data is analyzed to calculate performance metrics
- Results are visualized in charts and tables
- A comprehensive HTML report is generated

## 💻 Code Breakdown: Line-by-Line Explanations

### Imports and Setup

```python
import simpy               # Discrete-event simulation framework
import numpy as np         # For numerical operations and random distributions
import pandas as pd        # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating visualizations
import os                  # For file and directory operations
import datetime            # For timestamping reports
from tabulate import tabulate  # For creating text tables
import jinja2             # For HTML templating
import webbrowser         # For opening the report in a browser
```

### Probability Distributions

These functions generate random times based on statistical distributions:

```python
def interarrival_time():
    """Product arrivals: Exponential distribution (mean=5)"""
    return np.random.exponential(5)
    # Returns a random time between product arrivals.
    # Sometimes products arrive quickly after each other, sometimes with longer gaps.
    # The average time between arrivals is 5 time units.
```

```python
def processing_time(machine_id):
    """Different processing time distributions for each machine"""
    if machine_id == "M1":
        return np.random.normal(8, 1.5)  
        # M1 uses a normal (Gaussian) distribution
        # Mean processing time is 8 with standard deviation of 1.5
        # Most processing times will be between 5 and 11 time units
    elif machine_id == "M2":
        return np.random.triangular(6, 8, 10)  
        # M2 uses a triangular distribution
        # Minimum=6, Mode (most common)=8, Maximum=10
        # Processing times are always between 6 and 10, most commonly around 8
    else:
        return np.random.gamma(2, 3)  
        # M3 uses a gamma distribution with shape=2, scale=3
        # This gives a right-skewed distribution with mean=6
        # Occasionally will have longer processing times
```

```python
def failure_time():
    """Machine failures: Weibull distribution"""
    return np.random.weibull(2) * 20
    # Returns the time until a machine breaks down
    # Weibull distribution is commonly used to model equipment failures
    # Shape parameter=2 means failure rate increases with time (wear out)
    # Scale factor of 20 sets the characteristic life of the machine
```

```python
def repair_time():
    """Repair times: Exponential distribution (mean=10)"""
    return np.random.exponential(10)
    # Returns the time needed to repair a broken machine
    # Repairs follow an exponential distribution with mean=10
    # Most repairs are quick, but occasionally some take much longer
```

### Data Tracking Variables

```python
# Data tracking
event_log = []              # Record of all events that occur during simulation
product_metrics = []        # Detailed measurements for each product
machine_states = []         # Record of machine state changes (failure/repair)
interarrival_times = []     # Track time between arrivals
arrival_times = []          # Track when products arrive
active_products = 0         # Count of products currently in the system
```

### Machine Class

This class represents each machine in our factory:

```python
class Machine:
    def __init__(self, env, name):
        # Basic machine properties
        self.env = env                               # SimPy environment
        self.name = name                             # Machine name (M1, M2, M3)
        self.resource = simpy.Resource(env, capacity=1)  # Can process 1 product at a time
        self.broken = False                          # Starts in working condition
        self.total_downtime = 0                      # Time spent broken
        self.last_failure_time = 0                   # When the machine last broke
        self.utilization_time = 0                    # Time spent actively processing
        self.processing_count = 0                    # Number of products processed
        self.failures = 0                            # Number of breakdowns
        self.queue_length = 0                        # Current queue length
        
        # Queueing theory metrics
        self.total_waiting_time = 0       # Total time products waited in queue for this machine
        self.products_that_waited = 0     # Count of products that had to wait
        self.sum_queue_length = 0         # Sum of queue lengths over time (for calculating average)
        self.queue_length_samples = 0     # Number of times queue length was sampled
        self.total_service_time = 0       # Total time spent processing products
        self.total_idle_time = 0          # Total time machine was idle
        self.last_departure_time = 0      # Time when last product departed
        self.last_busy_state_change = 0   # Time of last change in busy status
        self.busy = False                 # Whether machine is currently busy
        
        # Start two background processes for this machine:
        env.process(self.break_machine())    # Process that handles random failures
        env.process(self.monitor_queue())    # Process that records queue statistics
```

The `break_machine` method simulates machines breaking down and being repaired:

```python
def break_machine(self):
    """Process that simulates random machine failures and repairs"""
    while True:
        # Wait for a random time until next failure
        yield self.env.timeout(failure_time())
        
        # Only break if machine is in use and not already broken
        if not self.broken and self.busy:
            self.broken = True                      # Mark as broken
            self.last_failure_time = self.env.now   # Record when it broke
            self.failures += 1                      # Increment failure counter
            
            # Log the failure event
            log(f"{self.name} FAILED", self.env.now)
            machine_states.append((self.env.now, self.name, "FAILED"))
            
            # Wait for repair time to complete
            yield self.env.timeout(repair_time())
            
            # Calculate how long it was broken and update statistics
            downtime = self.env.now - self.last_failure_time
            self.total_downtime += downtime
            
            # Log the repair event
            log(f"{self.name} REPAIRED", self.env.now)
            machine_states.append((self.env.now, self.name, "REPAIRED"))
            self.broken = False                     # Mark as repaired
```

The `monitor_queue` method collects statistics about the queue:

```python
def monitor_queue(self):
    """Process to monitor queue length over time"""
    while True:
        # Sample the queue length periodically
        self.sum_queue_length += len(self.resource.queue)  # Add current queue length to sum
        self.queue_length_samples += 1                     # Increment sample counter
        yield self.env.timeout(1)                          # Wait 1 time unit before next sample
```

The `update_busy_status` method tracks machine utilization:

```python
def update_busy_status(self, new_busy_state):
    """Update busy/idle status and calculate idle time"""
    current_time = self.env.now
    
    if self.busy and not new_busy_state:  
        # Machine is becoming idle
        self.busy = False
    elif not self.busy and new_busy_state:  
        # Machine is becoming busy
        # Add idle time from last state change until now
        self.total_idle_time += current_time - self.last_busy_state_change
        self.busy = True
    
    # Update timestamp of last state change
    self.last_busy_state_change = current_time
```

### Product Class

This class represents each product flowing through our factory:

```python
class Product:
    def __init__(self, env, name, machines):
        self.env = env                  # SimPy environment
        self.name = name                # Product name (e.g., "Product-7")
        self.machines = machines        # List of machines to visit [M1, M2, M3]
        self.start_time = env.now       # Time product entered the system
        self.end_time = None            # Will store time product exits system
        self.machine_times = {}         # Total time spent at each machine
        self.queue_times = {}           # Time waiting in queue for each machine
        self.processing_times = {}      # Actual processing time at each machine
        
        # Track number of products in system
        global active_products
        active_products += 1            # Increment count of active products
        
        # Record arrival time and calculate interarrival time
        if len(arrival_times) > 0:
            interarrival_times.append(self.start_time - arrival_times[-1])
        arrival_times.append(self.start_time)
        
        # Start the process of moving through machines
        env.process(self.process())
```

The `process` method handles the product's journey through the factory:

```python
def process(self):
    arrival_time = self.env.now
    log(f"{self.name} ARRIVED", self.env.now)  # Log arrival event

    # Visit each machine in sequence
    for machine in self.machines:
        queue_start = self.env.now       # Record time entering the queue
        machine.queue_length += 1        # Increment queue length counter

        # Request access to the machine and wait if it's busy
        with machine.resource.request() as request:
            # Update machine busy status
            machine.update_busy_status(True)  # Machine will be busy when we get it
            
            # Wait for the machine to be available
            yield request                # This pauses until we get the machine
            
            # Calculate time spent in queue
            queue_time = self.env.now - queue_start
            self.queue_times[machine.name] = queue_time
            machine.queue_length -= 1   # Decrement queue counter
            
            # Update queue statistics
            machine.total_waiting_time += queue_time
            if queue_time > 0:
                machine.products_that_waited += 1
            
            # Wait if machine is broken (keep checking until it's fixed)
            while machine.broken:
                log(f"{self.name} WAITING for {machine.name} repair", self.env.now)
                yield self.env.timeout(1)  # Wait 1 time unit, then check again
            
            # Start processing the product
            log(f"{self.name} START {machine.name}", self.env.now)
            start_time = self.env.now
            
            # Determine total processing time for this product on this machine
            remaining_time = processing_time(machine.name)
            self.processing_times[machine.name] = remaining_time
            
            # Update service time metrics
            machine.total_service_time += remaining_time
            
            # Process the product, accounting for possible machine failures
            while remaining_time > 0:
                if machine.broken:
                    # If machine breaks during processing, wait until it's fixed
                    log(f"{self.name} PAUSED {machine.name} due to failure", self.env.now)
                    yield self.env.timeout(1)  # Check again after 1 time unit
                else:
                    # Process for a small step (maximum 1 time unit)
                    step = min(1, remaining_time)
                    yield self.env.timeout(step)
                    remaining_time -= step
                    machine.utilization_time += step
            
            # Finished processing at this machine
            machine.processing_count += 1
            machine.last_departure_time = self.env.now
            log(f"{self.name} END {machine.name}", self.env.now)
            
            # Record total time spent at this machine (includes waiting due to breakdowns)
            self.machine_times[machine.name] = self.env.now - start_time
            
            # Update machine status to not busy
            machine.update_busy_status(False)

    # Product has completed all machines
    self.end_time = self.env.now
    total_time = self.end_time - arrival_time
    log(f"{self.name} FINISHED, total time: {total_time:.2f}", self.env.now)
    
    # Decrement active products counter
    global active_products
    active_products -= 1
    
    # Calculate total waiting and service time across all machines
    total_waiting_time = sum(self.queue_times.values())
    total_service_time = sum(self.processing_times.values())
    
    # Save detailed metrics for this product
    product_metrics.append({
        'Product': self.name,
        'Arrival': arrival_time,
        'Completion': self.end_time,
        'Lead Time': total_time,
        'Total Waiting Time': total_waiting_time,
        'Total Service Time': total_service_time,
        'M1 Queue Time': self.queue_times.get('M1', 0),
        'M1 Processing Time': self.processing_times.get('M1', 0),
        'M2 Queue Time': self.queue_times.get('M2', 0),
        'M2 Processing Time': self.processing_times.get('M2', 0),
        'M3 Queue Time': self.queue_times.get('M3', 0),
        'M3 Processing Time': self.processing_times.get('M3', 0),
        'M1 Time': self.machine_times.get('M1', 0),
        'M2 Time': self.machine_times.get('M2', 0),
        'M3 Time': self.machine_times.get('M3', 0),
        'System Time': total_time  # Total time in system
    })
```

### Event Logging Function

```python
def log(message, time):
    """Add an event to the log"""
    event_log.append((time, message))
    # Stores the timestamp and description of each event
```

### Product Arrival Process

```python
def product_arrival(env, machines, total_products=100):
    """Generate product arrivals"""
    i = 0
    while i < total_products:
        # Wait for interarrival time (random time between products)
        yield env.timeout(interarrival_time())
        
        # Create new product with unique name
        Product(env, f"Product-{i}", machines)
        i += 1
        
        # Log progress periodically
        if i % 10 == 0:
            log(f"GENERATED {i}/{total_products} products", env.now)
```

### Main Simulation Function

```python
def run_simulation(sim_time=None, total_products=100):
    """Run the simulation until all products are processed
    
    Parameters:
    - sim_time: Maximum simulation time (if None, runs until all products are processed)
    - total_products: Total number of products to generate
    """
    # Reset global trackers
    global event_log, product_metrics, machine_states
    global interarrival_times, arrival_times, active_products
    
    event_log = []
    product_metrics = []
    machine_states = []
    interarrival_times = []
    arrival_times = []
    active_products = 0
    
    # Initialize simulation environment
    env = simpy.Environment()
    
    # Create three machines
    machines = [Machine(env, "M1"), Machine(env, "M2"), Machine(env, "M3")]
    
    # Start the product arrival process
    env.process(product_arrival(env, machines, total_products))
    
    # If no specific simulation time is given, run until all products finish
    if sim_time is None:
        # First, run until all products are generated
        estimated_arrival_time = 5 * total_products + 50
        env.run(until=estimated_arrival_time)
        
        # Then run until all active products are processed
        while active_products > 0:
            # Run in smaller increments and check active_products
            env.run(until=env.now + 50)
            log(f"TIME: {env.now:.1f}, REMAINING PRODUCTS: {active_products}", env.now)
            
            # Safety check - stop if simulation runs too long
            if env.now > 10000:
                log("SIMULATION TIMEOUT - Some products may not have completed", env.now)
                break
    else:
        # Run for the specified simulation time
        env.run(until=sim_time)
    
    # Add explanation about utilization and waiting time
    explanation = (
        "NOTE: Even with moderate utilization rates (e.g., 67%), significant waiting times can occur. "
        "In queueing theory, as utilization increases, waiting times grow exponentially, not linearly. "
        "For example, at 67% utilization, queue length and waiting time can already be substantial, "
        "and they increase dramatically as utilization approaches 100%."
    )
    log(explanation, env.now)
    
    return machines, env.now
```

### Results Analysis

```python
def analyze_results(machines, sim_time):
    """Analyze simulation results"""
    # Convert logs to DataFrames
    event_df = pd.DataFrame(event_log, columns=['Time', 'Event'])
    event_df['Time'] = event_df['Time'].round(2)
    
    # Product metrics table
    product_df = pd.DataFrame(product_metrics)
    if len(product_df) > 0:  # Check if any products were processed
        product_df = product_df.round(2)
        
        # Print summary of products processed
        print(f"Processed {len(product_df)} products out of {len(product_df)} ({len(product_df)/len(product_df)*100:.1f}%)")
    else:
        print("No products were processed in the simulation!")
        # Return minimal results to avoid errors
        return {
            'event_log': event_df,
            'product_metrics': pd.DataFrame(),
            'machine_metrics': pd.DataFrame(),
            'system_metrics': {},
            'queue_metrics': {},
            'bottleneck': 'Unknown',
            'failure_prone': 'Unknown'
        }
    
    # Calculate System-Level Queueing Metrics
    total_products = len(product_df)
    
    # Average time between arrivals
    avg_interarrival_time = sum(interarrival_times) / len(interarrival_times) if interarrival_times else 0
    
    # Calculate system-level waiting metrics
    total_system_wait = product_df['Total Waiting Time'].sum()
    products_that_waited = len(product_df[product_df['Total Waiting Time'] > 0])
    
    # System-wide queueing metrics
    system_queueing_metrics = {
        'Average Waiting Time': total_system_wait / total_products if total_products > 0 else 0,
        'Probability of Wait': products_that_waited / total_products if total_products > 0 else 0,
        'Average Queue Length': sum(m.sum_queue_length for m in machines) / sum(m.queue_length_samples for m in machines) if sum(m.queue_length_samples for m in machines) > 0 else 0,
        'Average Service Time': product_df['Total Service Time'].mean(),
        'Average Interarrival Time': avg_interarrival_time,
        'Average Waiting Time (Those Who Wait)': total_system_wait / products_that_waited if products_that_waited > 0 else 0,
        'Average Time in System': product_df['Lead Time'].mean(),
        'Server Utilization': 1 - (sum(m.total_idle_time for m in machines) / (sim_time * len(machines)))
    }
    
    # Detailed machine metrics
    machine_data = []
    for m in machines:
        # Calculate percentages
        utilization_pct = (m.utilization_time / sim_time) * 100 if sim_time > 0 else 0
        availability_pct = ((sim_time - m.total_downtime) / sim_time) * 100 if sim_time > 0 else 0
        
        # Calculate queueing metrics for this machine
        avg_waiting_time = m.total_waiting_time / total_products if total_products > 0 else 0
        prob_wait = m.products_that_waited / total_products if total_products > 0 else 0
        avg_queue_length = m.sum_queue_length / m.queue_length_samples if m.queue_length_samples > 0 else 0
        prob_idle = m.total_idle_time / sim_time if sim_time > 0 else 0
        avg_service_time = m.total_service_time / total_products if total_products > 0 else 0
        avg_wait_those_who_wait = m.total_waiting_time / m.products_that_waited if m.products_that_waited > 0 else 0
        server_utilization = 1 - prob_idle
        
        # Add all metrics for this machine
        machine_data.append({
            'Machine': m.name,
            'Products Processed': m.processing_count,
            'Utilization (%)': round(utilization_pct, 2),
            'Availability (%)': round(availability_pct, 2),
            'Total Downtime': round(m.total_downtime, 2),
            'Number of Failures': m.failures,
            # Queueing metrics
            'Avg Waiting Time': round(avg_waiting_time, 2),
            'Probability of Wait': round(prob_wait, 2),
            'Avg Queue Length': round(avg_queue_length, 2),
            'Probability of Idle': round(prob_idle, 2),
            'Avg Service Time': round(avg_service_time, 2),
            'Avg Wait (Those Who Wait)': round(avg_wait_those_who_wait, 2),
            'Server Utilization': round(server_utilization, 2)
        })
    
    machine_df = pd.DataFrame(machine_data)
    
    # Bottleneck identification
    bottleneck = machine_df.loc[machine_df['Utilization (%)'].idxmax()]['Machine']
    failure_prone = machine_df.loc[machine_df['Number of Failures'].idxmax()]['Machine']
    
    # Calculate additional system-level metrics
    avg_waiting_time = product_df['Total Waiting Time'].mean()
    avg_service_time = product_df['Total Service Time'].mean()
    avg_lead_time = product_df['Lead Time'].mean()
    
    throughput = len(product_df) / sim_time if sim_time > 0 else 0
    efficiency = (avg_service_time / avg_lead_time) * 100 if avg_lead_time > 0 else 0
    
    system_metrics = {
        'Average Waiting Time': round(avg_waiting_time, 2),
        'Average Service Time': round(avg_service_time, 2),
        'Average Lead Time': round(avg_lead_time, 2),
        'Waiting Time Percentage': round((avg_waiting_time / avg_lead_time) * 100 if avg_lead_time > 0 else 0, 2),
        'Throughput (products/time unit)': round(throughput, 4),
        'System Efficiency (%)': round(efficiency, 2)
    }
    
    # Return all results in a dictionary
    return {
        'event_log': event_df,
        'product_metrics': product_df,
        'machine_metrics': machine_df,
        'system_metrics': system_metrics,
        'queue_metrics': system_queueing_metrics,
        'bottleneck': bottleneck,
        'failure_prone': failure_prone
    }
```

### Running the Simulation

```python
def run_and_display(total_products=100, display_products=20, selection_method='first', max_sim_time=None):
    """Run simulation and generate report"""
    print("\n=== FACTORY PRODUCTION LINE SIMULATION ===\n")
    print(f"Running simulation with {total_products} products...")
    
    # Add an explanation about utilization and waiting times
    print("\n📚 UNDERSTANDING UTILIZATION AND WAITING TIME:")
    print("Even with moderate utilization rates (e.g., 67%), long waiting times can occur.")
    print("As utilization increases, waiting times grow exponentially, not linearly:")
    print("- At 50% utilization, relative waiting time = 1x")
    print("- At 67% utilization, relative waiting time = 2x")
    print("- At 80% utilization, relative waiting time = 4x")
    print("- At 90% utilization, relative waiting time = 9x")
    print("- At 95% utilization, relative waiting time = 19x")
    print("This is why bottlenecks form even when machines aren't running at 100% capacity.\n")
    
    # Run the actual simulation
    print("Starting simulation - will run until all products are processed...")
    machines, sim_time = run_simulation(sim_time=max_sim_time, total_products=total_products)
    
    # Analyze the results
    print("\nAnalyzing results...")
    results = analyze_results(machines, sim_time)
    
    # Select products to display in the report
    print(f"Preparing first {display_products} products for display...")
    display_results = select_display_products(results, display_products, selection_method)
    
    # Generate HTML report
    print("Generating HTML report...")
    report_file = create_html_report(display_results)
    
    # Print summary to console
    print("\n✅ SIMULATION COMPLETE!\n")
    print(f"📊 Report generated: {report_file}")
    print(f"📋 Total simulation time: {sim_time:.2f} time units")
    print(f"📦 Products processed: {len(results['product_metrics'])}/{total_products}")
    
    # Display key findings
    bottleneck = results['bottleneck']
    failure_prone = results['failure_prone']
    bottleneck_util = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == bottleneck, 'Utilization (%)'].values[0]
    failure_count = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == failure_prone, 'Number of Failures'].values[0]
    
    print("\n📈 KEY FINDINGS:")
    print("═════════════════════════════════════════════")
    print(f"🔍 BOTTLENECK: {bottleneck} ({bottleneck_util:.2f}% utilization)")
    print(f"🔧 RELIABILITY ISSUE: {failure_prone} ({failure_count} failures)")
    print("─────────────────────────────────────────────")
    print(f"⏱️  Average Lead Time: {results['system_metrics']['Average Lead Time']:.2f} time units")
    print(f"⏳ Average Waiting Time: {results['system_metrics']['Average Waiting Time']:.2f} time units")
    print(f"⚙️  Average Processing Time: {results['system_metrics']['Average Service Time']:.2f} time units")
    print(f"📉 System Efficiency: {results['system_metrics']['System Efficiency (%)']:.2f}%")
    print(f"📊 Throughput: {results['system_metrics']['Throughput (products/time unit)']:.4f} products/time unit")
    
    # Open the HTML report in the default web browser
    print("\n📂 Opening report in web browser...")
    webbrowser.open('file://' + os.path.abspath(report_file))
    
    return results
```

### Main Program Execution

```python
# Run the simulation when script is executed
if __name__ == "__main__":
    # Set parameters for simulation
    TOTAL_PRODUCTS = 100      # Simulate 100 products
    DISPLAY_PRODUCTS = 20     # But only display 20 in reports/visualizations
    SELECTION_METHOD = 'first'  # Show the first 20 products
    
    # Set to None to run until all products are processed
    MAX_SIM_TIME = None
    
    # Run the simulation and display results
    results = run_and_display(TOTAL_PRODUCTS, DISPLAY_PRODUCTS, SELECTION_METHOD, MAX_SIM_TIME)
```

## 📊 Key Performance Indicators Explained

The simulation calculates these important metrics:

### 1. Machine-Level Metrics
- **Utilization (%)**: Percentage of time a machine is actively processing products
  - *Formula*: `(processing_time / total_simulation_time) * 100`
- **Availability (%)**: Percentage of time a machine is operational (not broken)
  - *Formula*: `((total_time - downtime) / total_time) * 100`
- **Server Utilization**: Fraction of time the machine is busy
  - *Formula*: `1 - (idle_time / total_time)`
- **Average Queue Length**: Average number of products waiting for a machine
  - *Formula*: `sum_of_queue_length_samples / number_of_samples`
- **Probability of Wait**: Chance that a product will need to wait for a machine
  - *Formula*: `products_that_waited / total_products`

### 2. System-Level Metrics
- **Throughput**: Number of products completed per time unit
  - *Formula*: `products_completed / total_simulation_time`
- **Average Lead Time**: Average time from entry to exit for a product
  - *Formula*: `sum_of_all_product_lead_times / number_of_products`
- **System Efficiency (%)**: How much of the lead time is spent in actual processing
  - *Formula*: `(average_service_time / average_lead_time) * 100`
- **Waiting Time Percentage**: Percentage of lead time spent waiting
  - *Formula*: `(average_waiting_time / average_lead_time) * 100`

### 3. Bottleneck Analysis
- **Bottleneck Identification**: Machine with highest utilization
- **Failure Prone Identification**: Machine with most failures

## 📊 Simulation Output Explained

The simulation generates a comprehensive HTML report with visualizations and data tables to help analyze factory performance. Below is a breakdown of each section and what it means:

### 1. Executive Summary Dashboard 📈

![Executive Summary](reports/machine_chart.png)

* **What it shows:** A quick overview of the entire simulation with key metrics
* **Key metrics:**
  * **Throughput Rate:** How many products the factory completes per time unit (e.g., 0.0952 products/time unit)
  * **Average Lead Time:** Average time from when a product enters until it exits (e.g., 299.8 time units)
  * **System Efficiency:** Percentage of time spent actually processing vs. waiting (e.g., 7.2%)
  * **Server Utilization:** Average percentage of time machines are busy (e.g., 73.1%)
* **Alerts:**
  * **Bottleneck Alert:** Identifies which machine is slowing down the entire system (e.g., M2 at 77.9% utilization)
  * **Reliability Alert:** Shows which machine breaks down most often (e.g., M2 with 19 failures)
* **Why it matters:** Helps quickly identify the biggest issues limiting production

### 2. Queueing Theory Metrics 📑

![Queueing Metrics](reports/queueing_metrics.png)

* **What it shows:** Advanced analysis of how products flow and wait in the system
* **System-Wide Metrics:**
  * **Average Waiting Time:** Average time products spend waiting (e.g., 274.31 time units)
  * **Probability of Wait:** Chance that a product will need to wait (e.g., 99%)
  * **Average Queue Length:** Average number of products waiting at any time (e.g., 8.71 products)
  * **Server Utilization:** Percentage of time machines are busy (e.g., 73%)
* **Utilization-Waiting Time Relationship:**
  * Shows how waiting times grow exponentially as utilization increases
  * Even at 80% utilization, waiting times are 4x longer than at 50% utilization
* **Why it matters:** Explains why bottlenecks form even when machines aren't running at full capacity

### 3. Machine Performance Analysis 🔍

![Machine Performance](reports/machine_chart.png)

* **What it shows:** Utilization and availability for each machine
* **Key information:**
  * **Utilization bars:** Percentage of time each machine spends actively processing
  * **Availability bars:** Percentage of time each machine is operational (not broken)
  * **Bottleneck highlight:** The machine with highest utilization (limiting throughput)
* **Why it matters:** Helps identify which machine to upgrade first to improve overall production

### 4. Wait Time vs. Processing Time ⏱️

![Wait Time Analysis](reports/waiting_time_chart.png)

* **What it shows:** Comparison between waiting time and actual processing time at each machine
* **Key information:**
  * **Blue bars:** Average time products spend waiting in queue
  * **Orange bars:** Average time products spend being processed
* **Key insight:** Products spend much more time waiting than being processed
* **Why it matters:** Reveals where time is being wasted in the production process

### 5. Machine Reliability Analysis 🔧

![Reliability Analysis](reports/machine_status.png)

* **What it shows:** Patterns of machine failures and repairs
* **Key information:**
  * **Uptime periods:** When machines were operational
  * **Downtime periods:** When machines were broken and being repaired
  * **Failure frequency:** How often each machine breaks down
* **Why it matters:** Identifies which machines need better maintenance or replacement

### 6. Product Flow Timeline 🚶

![Product Timeline](reports/timeline_chart.png)

* **What it shows:** How individual products move through the production line over time
* **Key information:**
  * **Horizontal bars:** Each product's journey through the system
  * **Color segments:** Time spent at different machines and in queues
  * **Timeline:** When products enter and exit the system
* **Why it matters:** Visualizes the actual flow and bottlenecks in the production process

### 7. Product Metrics Chart 📊

![Product Metrics](reports/product_metrics_chart.png)

* **What it shows:** Detailed breakdown of metrics for individual products
* **Key metrics per product:**
  * **Lead Time:** Total time from entry to exit
  * **Waiting Time:** Time spent in queues
  * **Processing Time:** Time being actively processed
* **Why it matters:** Helps identify patterns or unusual cases in production

### 8. Machine Performance Metrics Table 📋

* **What it shows:** Comprehensive statistics for each machine in tabular format
* **Key metrics:**
  * **Products Processed:** Number of products that went through each machine
  * **Utilization (%):** Percentage of time actively processing products
  * **Availability (%):** Percentage of time operational (not broken)
  * **Number of Failures:** How many times each machine broke down
  * **Avg Queue Length:** Average number of products waiting
  * **Avg Service Time:** Average time to process one product
* **Why it matters:** Provides detailed data to support improvement decisions

### 9. System Performance Metrics Table 📋

* **What it shows:** Overall factory performance metrics
* **Key metrics:**
  * **Average Waiting Time:** Mean time products spend waiting (e.g., 274.31 time units)
  * **Average Service Time:** Mean time products spend being processed (e.g., 21.54 time units)
  * **Average Lead Time:** Mean total time in system (e.g., 299.81 time units)
  * **Waiting Time Percentage:** Proportion of time spent waiting (e.g., 91.5%)
  * **Throughput:** Products completed per time unit (e.g., 0.1)
  * **System Efficiency:** Ratio of processing time to total lead time (e.g., 7.19%)
* **Why it matters:** Provides big-picture metrics to assess overall factory performance

### 10. Product Metrics Table 📋

* **What it shows:** Detailed data for each individual product
* **Key information per product:**
  * **Arrival & Completion Times:** When products entered and exited
  * **Lead Time:** Total time in system
  * **Queue Times:** How long products waited at each machine
  * **Processing Times:** How long each machine took to process products
* **Why it matters:** Allows for detailed analysis of individual product journeys

### 11. Simulation Event Log 📝

* **What it shows:** Chronological record of all events during simulation
* **Events tracked:**
  * Product arrivals
  * Processing starts and ends
  * Machine failures and repairs
  * System completion events
* **Why it matters:** Provides a detailed timeline for debugging or understanding specific events

## 🔍 Interpreting Simulation Results

### Understanding Bottlenecks
- A bottleneck is the resource that limits the overall system throughput
- In our simulation, it's identified as the machine with the highest utilization percentage
- Improving the bottleneck's capacity or reducing its processing time will have the greatest impact on overall system performance

### Improving Reliability
- The machine with the most failures is identified as "failure prone"
- Reducing breakdowns of this machine can significantly improve system throughput
- Real-world solutions might include preventive maintenance or equipment upgrades

### The Waiting Time Paradox
- One of the most important concepts illustrated by this simulation is how waiting times grow exponentially as utilization increases
- This explains why even systems that appear to have enough capacity (e.g., 80% utilization) can experience significant waiting times
- This principle applies to many real-world situations: factory lines, call centers, checkout lanes, and traffic systems

## 🏭 Real-World Applications

This simulation can help understand and optimize many real-world scenarios:

1. **Manufacturing**: Designing efficient production lines and identifying bottlenecks
2. **Healthcare**: Improving patient flow in hospitals and clinics
3. **Customer Service**: Optimizing staffing in call centers
4. **Logistics**: Managing warehouse operations and delivery systems
5. **Computing**: Understanding system resource allocation in servers

## 👥 Credits

This project was developed for the System Modeling and Simulation (CCS3003/CS305) course at the Arab Academy, under the supervision of **Prof. Dr. Khaled Mahar** and **TA Sara Mohamed**.

Last Updated: May 20, 2025 by Marwan911e

## 👥 Team Members
- Marwan Elsayed - 221003166
- Hussein Galal - 221001810

---

## 📎 License
MIT License – free to use and adapt with credit.

