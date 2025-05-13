"""
# Factory Production Line Simulation
# Team Members: [Your Name]
# Course: [Course Name]
# Project Type: Manufacturing System Simulation

This simulation models a factory with three machines in series. Products arrive at the system
according to an exponential distribution, are processed by each machine in sequence, and then
exit the system. Machines can randomly fail and require repair.

The simulation tracks:
- Product flow through the system
- Machine states (busy, idle, failed)
- Queue lengths at each machine
- System bottlenecks
- Resource utilization
- Detailed simulation table with step-by-step events
"""

import simpy
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import matplotlib.dates as mdates
from datetime import datetime, timedelta


# ============================================================
# PROBABILITY DISTRIBUTIONS - Using 4 different distributions
# ============================================================

def interarrival_time():
    """
    Exponential distribution for product arrivals
    Mean interarrival time: 5 time units
    """
    return np.random.exponential(5)


def processing_time(machine_id):
    """
    Different distributions for each machine:
    - M1: Normal distribution (mean=8, std=1.5)
    - M2: Triangular distribution (min=6, mode=8, max=10)
    - M3: Gamma distribution (shape=2, scale=3)
    """
    if machine_id == "M1":
        return np.random.normal(8, 1.5)
    elif machine_id == "M2":
        return np.random.triangular(6, 8, 10)
    else:
        return np.random.gamma(2, 3)


def failure_time():
    """
    Weibull distribution for machine failures
    Shape parameter = 2, Scale adjusted to give mean time between failures of ~20
    """
    return np.random.weibull(2) * 20


def repair_time():
    """
    Exponential distribution for repair times
    Mean repair time: 10 time units
    """
    return np.random.exponential(10)


# ============================================================
# DATA TRACKING STRUCTURES
# ============================================================

# Complete event log with timestamps
event_log = []

# Detailed metrics for each product
product_metrics = []

# Machine state changes over time
machine_states = []

# Detailed simulation table showing all events and system state
simulation_table = []

# Current system state
current_state = {
    'time': 0,
    'queues': {'M1': 0, 'M2': 0, 'M3': 0},
    'machine_status': {'M1': 'Idle', 'M2': 'Idle', 'M3': 'Idle'},
    'products_in_system': 0,
    'completed_products': 0
}


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
        self.busy = False
        self.current_product = None
        env.process(self.break_machine())

    def break_machine(self):
        while True:
            yield self.env.timeout(failure_time())
            if not self.broken:
                self.broken = True
                self.last_failure_time = self.env.now
                self.failures += 1
                log(f"{self.name} FAILED", self.env.now)
                machine_states.append((self.env.now, self.name, "FAILED"))

                # Update simulation table
                update_simulation_table(self.env.now, f"{self.name} failed",
                                       "Machine failure event", self.name)

                # Update machine status in current state
                current_state['machine_status'][self.name] = 'Failed'

                yield self.env.timeout(repair_time())
                downtime = self.env.now - self.last_failure_time
                self.total_downtime += downtime
                log(f"{self.name} REPAIRED", self.env.now)
                machine_states.append((self.env.now, self.name, "REPAIRED"))

                # Update simulation table
                update_simulation_table(self.env.now, f"{self.name} repaired",
                                       "Machine repair completed", self.name)

                # Update machine status in current state
                if self.busy:
                    current_state['machine_status'][self.name] = 'Busy'
                else:
                    current_state['machine_status'][self.name] = 'Idle'

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
        self.current_machine = None
        env.process(self.process())

        # Update product counter when created
        current_state['products_in_system'] += 1

    def process(self):
        arrival_time = self.env.now
        log(f"{self.name} ARRIVED", self.env.now)

        # Update simulation table for arrival
        update_simulation_table(self.env.now, f"{self.name} arrived",
                               "New product entered the system", None)

        for machine in self.machines:
            # Record which machine we're at
            self.current_machine = machine.name
            machine_start = self.env.now
            queue_start = self.env.now

            # Increment queue counter
            machine.queue_length += 1
            current_state['queues'][machine.name] += 1

            # Update simulation table to show queue increment
            update_simulation_table(self.env.now, f"{self.name} entered {machine.name} queue",
                                   f"Queue length: {machine.queue_length}", machine.name)

            with machine.resource.request() as request:
                # Wait for machine to be available
                yield request

                # Calculate queue time
                queue_time = self.env.now - queue_start
                self.queue_times[machine.name] = queue_time

                # Decrement queue since we're now being processed
                machine.queue_length -= 1
                current_state['queues'][machine.name] -= 1

                # Update simulation table for exiting queue
                update_simulation_table(self.env.now, f"{self.name} left {machine.name} queue",
                                       f"Queue time: {queue_time:.2f}, Queue length: {machine.queue_length}",
                                       machine.name)

                # Wait if machine is broken
                if machine.broken:
                    log(f"{self.name} WAITING for {machine.name} repair", self.env.now)

                    # Update simulation table
                    update_simulation_table(self.env.now, f"{self.name} waiting for repair on {machine.name}",
                                           "Machine is currently broken", machine.name)

                while machine.broken:
                    yield self.env.timeout(1)

                # Start processing
                start = self.env.now
                p_time = processing_time(machine.name)
                self.processing_times[machine.name] = p_time

                machine.processing_count += 1
                machine.utilization_time += p_time
                machine.busy = True
                machine.current_product = self.name

                # Update machine status
                current_state['machine_status'][machine.name] = 'Busy'

                log(f"{self.name} START {machine.name}", start)

                # Update simulation table for processing start
                update_simulation_table(self.env.now, f"{self.name} started processing on {machine.name}",
                                       f"Processing time: {p_time:.2f}", machine.name)

                # Process for the required time
                yield self.env.timeout(p_time)

                # Processing complete
                end = self.env.now
                machine.busy = False
                machine.current_product = None

                # Update machine status
                if not machine.broken:
                    current_state['machine_status'][machine.name] = 'Idle'

                log(f"{self.name} END {machine.name}", end)
                self.machine_times[machine.name] = end - machine_start

                # Update simulation table for processing end
                update_simulation_table(self.env.now, f"{self.name} completed processing on {machine.name}",
                                       f"Total time on machine: {self.machine_times[machine.name]:.2f}",
                                       machine.name)

        self.end_time = self.env.now
        total_time = self.end_time - arrival_time

        # Update product counter when finished
        current_state['products_in_system'] -= 1
        current_state['completed_products'] += 1

        log(f"{self.name} FINISHED, total time: {total_time:.2f}", self.env.now)

        # Update simulation table for product completion
        update_simulation_table(self.env.now, f"{self.name} finished processing",
                               f"Total lead time: {total_time:.2f}", None)

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
    event_log.append((time, message))


def update_simulation_table(time, event, details, machine=None):
    """Add an event to the simulation table with current system state"""

    # Update the current state time
    current_state['time'] = time

    # Create a deep copy of the current state for the table
    state_copy = current_state.copy()
    queue_copy = current_state['queues'].copy()
    status_copy = current_state['machine_status'].copy()

    # Create a row for the simulation table
    row = {
        'Time': round(time, 2),
        'Event': event,
        'Details': details,
        'Machine': machine if machine else '-',
        'M1 Status': status_copy['M1'],
        'M2 Status': status_copy['M2'],
        'M3 Status': status_copy['M3'],
        'M1 Queue': queue_copy['M1'],
        'M2 Queue': queue_copy['M2'],
        'M3 Queue': queue_copy['M3'],
        'Products in System': state_copy['products_in_system'],
        'Completed Products': state_copy['completed_products']
    }

    simulation_table.append(row)


def product_arrival(env, machines):
    i = 0
    while True:
        # Generate interarrival time
        interarrival = interarrival_time()

        # Update simulation table for interarrival event
        update_simulation_table(env.now, "Scheduling next arrival",
                               f"Next arrival in {interarrival:.2f} time units", None)

        # Wait for interarrival time
        yield env.timeout(interarrival)

        # Create new product
        Product(env, f"Product-{i}", machines)
        i += 1

        # Stop after generating enough products
        if i >= 20:
            break


# Setup and run simulation
def run_simulation(sim_time=200):
    global event_log, product_metrics, machine_states, simulation_table, current_state

    # Reset all trackers
    event_log = []
    product_metrics = []
    machine_states = []
    simulation_table = []
    current_state = {
        'time': 0,
        'queues': {'M1': 0, 'M2': 0, 'M3': 0},
        'machine_status': {'M1': 'Idle', 'M2': 'Idle', 'M3': 'Idle'},
        'products_in_system': 0,
        'completed_products': 0
    }

    # Initialize simulation environment
    env = simpy.Environment()
    machines = [Machine(env, "M1"), Machine(env, "M2"), Machine(env, "M3")]

    # Start the product arrival process
    env.process(product_arrival(env, machines))

    # Initialize simulation table with starting state
    update_simulation_table(0, "Simulation start", "Initial system state", None)

    # Run the simulation
    env.run(until=sim_time)

    return machines, env.now


# ---- Analysis ----
def analyze_results(machines, sim_time):
    # Create DataFrames for better display
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

    # Simulation table
    sim_table_df = pd.DataFrame(simulation_table)

    # Ensure we have at least 20 rows in the simulation table
    if len(sim_table_df) < 20:
        print(f"Warning: Simulation produced only {len(sim_table_df)} events. Consider running longer.")

    # Bottleneck identification
    bottleneck = machine_df.loc[machine_df['Utilization (%)'].idxmax()]['Machine']
    failure_prone = machine_df.loc[machine_df['Number of Failures'].idxmax()]['Machine']

    return {
        'event_log': event_df,
        'product_metrics': product_df,
        'machine_metrics': machine_df,
        'simulation_table': sim_table_df,
        'bottleneck': bottleneck,
        'failure_prone': failure_prone
    }


# ---- Visualization ----
def visualize_results(results):
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 12))

    # 1. Gantt chart for product flow
    ax1 = plt.subplot(2, 1, 1)
    product_df = results['product_metrics']

    # Sort by arrival time
    product_df = product_df.sort_values('Arrival')

    # Create Gantt chart
    for i, row in product_df.iterrows():
        plt.barh(row['Product'], row['Lead Time'], left=row['Arrival'],
                 height=0.5, color='lightblue')

        # Add machine processing segments with different colors
        left = row['Arrival']

        # Add segments for each machine
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

    # Add a legend for machine colors
    import matplotlib.patches as mpatches
    m1_queue = mpatches.Patch(color='green', alpha=0.3, label='M1 Queue')
    m1_proc = mpatches.Patch(color='green', alpha=0.7, label='M1 Processing')
    m2_queue = mpatches.Patch(color='blue', alpha=0.3, label='M2 Queue')
    m2_proc = mpatches.Patch(color='blue', alpha=0.7, label='M2 Processing')
    m3_queue = mpatches.Patch(color='purple', alpha=0.3, label='M3 Queue')
    m3_proc = mpatches.Patch(color='purple', alpha=0.7, label='M3 Processing')
    total_patch = mpatches.Patch(color='lightblue', label='Total Lead Time')
    plt.legend(handles=[total_patch, m1_queue, m1_proc, m2_queue, m2_proc, m3_queue, m3_proc],
               loc='upper right', ncol=2)

    # 2. Machine states over time (failure events)
    ax2 = plt.subplot(2, 1, 2)

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
                        marker='v', color='red', s=100, label=f'{machine} Failures' if machine == 'M1' else "")

        if not repairs.empty:
            plt.scatter(repairs['Time'], [machine] * len(repairs),
                        marker='^', color='green', s=100, label=f'{machine} Repairs' if machine == 'M1' else "")

    plt.title('Machine Failures and Repairs')
    plt.xlabel('Simulation Time')
    plt.ylabel('Machine')
    plt.yticks(['M1', 'M2', 'M3'])
    plt.grid(axis='x', alpha=0.3)

    # Add a legend
    plt.legend(loc='upper right')

    plt.tight_layout()

    return fig


def run_and_display():
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

    print("\nDETAILED SIMULATION TABLE (first 20 events):")
    print(tabulate(results['simulation_table'].head(20), headers='keys', tablefmt='pretty', showindex=False))

    # Create and display visualizations
    fig = visualize_results(results)
    plt.show()

    return results


# Run the simulation and display results
if __name__ == "__main__":
    results = run_and_display()