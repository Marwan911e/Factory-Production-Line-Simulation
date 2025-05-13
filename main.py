"""
# Simplified Factory Production Line Simulation
# Models a factory with three machines in series, tracking:
# - Product flow through the system
# - Machine states
# - Queue lengths
"""

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


# Simplified probability distributions
def interarrival_time():
    """Product arrivals: Exponential distribution (mean=5)"""
    return np.random.exponential(5)


def processing_time(machine_id):
    """Different processing time distributions for each machine"""
    if machine_id == "M1":
        return np.random.normal(8, 1.5)  # Normal distribution
    elif machine_id == "M2":
        return np.random.triangular(6, 8, 10)  # Triangular distribution
    else:
        return np.random.gamma(2, 3)  # Gamma distribution


def failure_time():
    """Machine failures: Weibull distribution"""
    return np.random.weibull(2) * 20


def repair_time():
    """Repair times: Exponential distribution (mean=10)"""
    return np.random.exponential(10)


# Data tracking
event_log = []
product_metrics = []
machine_states = []


class Machine:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.resource = simpy.Resource(env, capacity=1)
        self.broken = False
        self.total_downtime = 0
        self.last_failure_time = 0
        self.utilization_time = 0
        self.processing_count = 0
        self.failures = 0
        self.queue_length = 0
        env.process(self.break_machine())

    def break_machine(self):
        """Process that simulates random machine failures and repairs"""
        while True:
            yield self.env.timeout(failure_time())
            if not self.broken:
                self.broken = True
                self.last_failure_time = self.env.now
                self.failures += 1

                # Log machine failure
                log(f"{self.name} FAILED", self.env.now)
                machine_states.append((self.env.now, self.name, "FAILED"))

                # Repair time
                yield self.env.timeout(repair_time())

                # Calculate downtime and log repair
                downtime = self.env.now - self.last_failure_time
                self.total_downtime += downtime
                log(f"{self.name} REPAIRED", self.env.now)
                machine_states.append((self.env.now, self.name, "REPAIRED"))
                self.broken = False


class Product:
    def __init__(self, env, name, machines):
        self.env = env
        self.name = name
        self.machines = machines
        self.start_time = env.now
        self.end_time = None
        self.machine_times = {}
        self.queue_times = {}
        self.processing_times = {}
        env.process(self.process())

    def process(self):
        """Process a product through all machines in sequence"""
        arrival_time = self.env.now
        log(f"{self.name} ARRIVED", self.env.now)

        for machine in self.machines:
            # Record machine start time
            machine_start = self.env.now
            queue_start = self.env.now

            # Increment queue counter
            machine.queue_length += 1

            # Wait for machine to be available
            with machine.resource.request() as request:
                yield request

                # Calculate queue time
                queue_time = self.env.now - queue_start
                self.queue_times[machine.name] = queue_time

                # Decrement queue
                machine.queue_length -= 1

                # Wait if machine is broken
                if machine.broken:
                    log(f"{self.name} WAITING for {machine.name} repair", self.env.now)

                while machine.broken:
                    yield self.env.timeout(1)

                # Start processing
                start = self.env.now
                p_time = processing_time(machine.name)
                self.processing_times[machine.name] = p_time

                # Update machine metrics
                machine.processing_count += 1
                machine.utilization_time += p_time

                log(f"{self.name} START {machine.name}", start)

                # Process for the required time
                yield self.env.timeout(p_time)

                # Processing complete
                end = self.env.now
                log(f"{self.name} END {machine.name}", end)
                self.machine_times[machine.name] = end - machine_start

        # Calculate total time and log completion
        self.end_time = self.env.now
        total_time = self.end_time - arrival_time
        log(f"{self.name} FINISHED, total time: {total_time:.2f}", self.env.now)

        # Store product metrics
        product_metrics.append({
            'Product': self.name,
            'Arrival': arrival_time,
            'Completion': self.end_time,
            'Lead Time': total_time,
            'M1 Queue Time': self.queue_times.get('M1', 0),
            'M1 Processing Time': self.processing_times.get('M1', 0),
            'M2 Queue Time': self.queue_times.get('M2', 0),
            'M2 Processing Time': self.processing_times.get('M2', 0),
            'M3 Queue Time': self.queue_times.get('M3', 0),
            'M3 Processing Time': self.processing_times.get('M3', 0),
            'M1 Time': self.machine_times.get('M1', 0),
            'M2 Time': self.machine_times.get('M2', 0),
            'M3 Time': self.machine_times.get('M3', 0)
        })


def log(message, time):
    """Add an event to the log"""
    event_log.append((time, message))


def product_arrival(env, machines):
    """Generate product arrivals"""
    i = 0
    while True:
        # Wait for interarrival time
        yield env.timeout(interarrival_time())

        # Create new product
        Product(env, f"Product-{i}", machines)
        i += 1

        # Stop after generating enough products
        if i >= 20:
            break


def run_simulation(sim_time=200):
    """Run the simulation"""
    global event_log, product_metrics, machine_states

    # Reset all trackers
    event_log = []
    product_metrics = []
    machine_states = []

    # Initialize simulation environment
    env = simpy.Environment()
    machines = [Machine(env, "M1"), Machine(env, "M2"), Machine(env, "M3")]

    # Start the product arrival process
    env.process(product_arrival(env, machines))

    # Run the simulation
    env.run(until=sim_time)

    return machines, env.now


def analyze_results(machines, sim_time):
    """Analyze simulation results"""
    # Convert logs to DataFrames
    event_df = pd.DataFrame(event_log, columns=['Time', 'Event'])
    event_df['Time'] = event_df['Time'].round(2)

    # Product metrics table
    product_df = pd.DataFrame(product_metrics)
    product_df = product_df.round(2)

    # Machine metrics table
    machine_data = []
    for m in machines:
        utilization_pct = (m.utilization_time / sim_time) * 100 if sim_time > 0 else 0
        availability_pct = ((sim_time - m.total_downtime) / sim_time) * 100 if sim_time > 0 else 0

        machine_data.append({
            'Machine': m.name,
            'Products Processed': m.processing_count,
            'Utilization (%)': round(utilization_pct, 2),
            'Availability (%)': round(availability_pct, 2),
            'Total Downtime': round(m.total_downtime, 2),
            'Number of Failures': m.failures
        })

    machine_df = pd.DataFrame(machine_data)

    # Bottleneck identification
    bottleneck = machine_df.loc[machine_df['Utilization (%)'].idxmax()]['Machine']
    failure_prone = machine_df.loc[machine_df['Number of Failures'].idxmax()]['Machine']

    return {
        'event_log': event_df,
        'product_metrics': product_df,
        'machine_metrics': machine_df,
        'bottleneck': bottleneck,
        'failure_prone': failure_prone
    }


def visualize_results(results):
    """Create visualizations of simulation results"""
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Gantt chart for product flow
    plt.subplot(2, 1, 1)
    product_df = results['product_metrics']

    # Sort by arrival time
    product_df = product_df.sort_values('Arrival')

    # Create Gantt chart
    for i, row in product_df.iterrows():
        # Total lead time bar
        plt.barh(row['Product'], row['Lead Time'], left=row['Arrival'],
                 height=0.5, color='lightblue')

        # Add machine processing segments with different colors
        left = row['Arrival']
        colors = {'M1': 'green', 'M2': 'blue', 'M3': 'purple'}

        for machine in ['M1', 'M2', 'M3']:
            # Add queue time in lighter color
            if row[f'{machine} Queue Time'] > 0:
                plt.barh(row['Product'], row[f'{machine} Queue Time'],
                         left=left, height=0.5, color=colors[machine], alpha=0.3)
                left += row[f'{machine} Queue Time']

            # Add processing time in darker color
            if row[f'{machine} Processing Time'] > 0:
                plt.barh(row['Product'], row[f'{machine} Processing Time'],
                         left=left, height=0.5, color=colors[machine], alpha=0.7)
                left += row[f'{machine} Processing Time']

    plt.title('Product Flow Timeline')
    plt.xlabel('Simulation Time')
    plt.ylabel('Product')
    plt.grid(axis='x', alpha=0.3)

    # Add a legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color='lightblue', label='Total Lead Time'),
        mpatches.Patch(color='green', alpha=0.3, label='M1 Queue'),
        mpatches.Patch(color='green', alpha=0.7, label='M1 Processing'),
        mpatches.Patch(color='blue', alpha=0.3, label='M2 Queue'),
        mpatches.Patch(color='blue', alpha=0.7, label='M2 Processing'),
        mpatches.Patch(color='purple', alpha=0.3, label='M3 Queue'),
        mpatches.Patch(color='purple', alpha=0.7, label='M3 Processing')
    ]
    plt.legend(handles=handles, loc='upper right', ncol=2)

    # 2. Machine states over time (failure events)
    plt.subplot(2, 1, 2)

    # Create machine state dataframe
    machine_state_df = pd.DataFrame(machine_states, columns=['Time', 'Machine', 'State'])

    # Plot machine failures/repairs
    for machine in ['M1', 'M2', 'M3']:
        machine_events = machine_state_df[machine_state_df['Machine'] == machine]

        # Plot markers for failures and repairs
        failures = machine_events[machine_events['State'] == 'FAILED']
        repairs = machine_events[machine_events['State'] == 'REPAIRED']

        if not failures.empty:
            plt.scatter(failures['Time'], [machine] * len(failures),
                        marker='v', color='red', s=100,
                        label=f'{machine} Failures' if machine == 'M1' else "")

        if not repairs.empty:
            plt.scatter(repairs['Time'], [machine] * len(repairs),
                        marker='^', color='green', s=100,
                        label=f'{machine} Repairs' if machine == 'M1' else "")

    plt.title('Machine Failures and Repairs')
    plt.xlabel('Simulation Time')
    plt.ylabel('Machine')
    plt.yticks(['M1', 'M2', 'M3'])
    plt.grid(axis='x', alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout()
    return fig


def run_and_display():
    """Run simulation and display results"""
    # Run simulation
    machines, sim_time = run_simulation(200)

    # Analyze results
    results = analyze_results(machines, sim_time)

    # Display tables
    print("\n=== FACTORY SIMULATION RESULTS ===\n")

    print("MACHINE METRICS:")
    print(tabulate(results['machine_metrics'], headers='keys', tablefmt='pretty', showindex=False))

    print("\nPRODUCT METRICS:")
    # Select key columns for cleaner display
    product_display = results['product_metrics'][['Product', 'Arrival', 'Completion', 'Lead Time',
                                                'M1 Queue Time', 'M1 Processing Time',
                                                'M2 Queue Time', 'M2 Processing Time',
                                                'M3 Queue Time', 'M3 Processing Time']]
    print(tabulate(product_display, headers='keys', tablefmt='pretty', showindex=False))

    print("\nKEY FINDINGS:")
    print(f"- Bottleneck machine: {results['bottleneck']}")
    print(f"- Most failure-prone machine: {results['failure_prone']}")

    # Create and display visualizations
    fig = visualize_results(results)
    plt.show()

    return results


# Run the simulation and display results
if __name__ == "__main__":
    results = run_and_display()