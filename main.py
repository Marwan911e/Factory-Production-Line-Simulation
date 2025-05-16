"""
# Simplified Factory Production Line Simulation
# Models a factory with three machines in series, tracking:
# - Product flow through the system
# - Machine states
# - Queue lengths
# - Waiting time and service time metrics
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
        arrival_time = self.env.now
        log(f"{self.name} ARRIVED", self.env.now)

        for machine in self.machines:
            queue_start = self.env.now
            machine.queue_length += 1

            with machine.resource.request() as request:
                yield request
                queue_time = self.env.now - queue_start
                self.queue_times[machine.name] = queue_time
                machine.queue_length -= 1

                # Wait if machine is broken
                while machine.broken:
                    log(f"{self.name} WAITING for {machine.name} repair", self.env.now)
                    yield self.env.timeout(1)

                # Start processing
                log(f"{self.name} START {machine.name}", self.env.now)
                start_time = self.env.now
                remaining_time = processing_time(machine.name)
                self.processing_times[machine.name] = remaining_time

                # Update utilization when time is actually processed
                while remaining_time > 0:
                    if machine.broken:
                        log(f"{self.name} PAUSED {machine.name} due to failure", self.env.now)
                        yield self.env.timeout(1)
                    else:
                        step = min(1, remaining_time)
                        yield self.env.timeout(step)
                        remaining_time -= step
                        machine.utilization_time += step

                machine.processing_count += 1
                log(f"{self.name} END {machine.name}", self.env.now)
                self.machine_times[machine.name] = self.env.now - start_time

        self.end_time = self.env.now
        total_time = self.end_time - arrival_time
        log(f"{self.name} FINISHED, total time: {total_time:.2f}", self.env.now)

        # Calculate total waiting time and service time
        total_waiting_time = sum(self.queue_times.values())
        total_service_time = sum(self.processing_times.values())

        # Save product metrics
        product_metrics.append({
            'Product': self.name,
            'Arrival': arrival_time,
            'Completion': self.end_time,
            'Lead Time': total_time,
            'Total Waiting Time': total_waiting_time,  # Sum of all queue times
            'Total Service Time': total_service_time,  # Sum of all processing times
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

    # Calculate additional system-level metrics
    avg_waiting_time = product_df['Total Waiting Time'].mean()
    avg_service_time = product_df['Total Service Time'].mean()
    avg_lead_time = product_df['Lead Time'].mean()

    system_metrics = {
        'Average Waiting Time': round(avg_waiting_time, 2),
        'Average Service Time': round(avg_service_time, 2),
        'Average Lead Time': round(avg_lead_time, 2),
        'Waiting Time Percentage': round((avg_waiting_time / avg_lead_time) * 100 if avg_lead_time > 0 else 0, 2)
    }

    return {
        'event_log': event_df,
        'product_metrics': product_df,
        'machine_metrics': machine_df,
        'system_metrics': system_metrics,
        'bottleneck': bottleneck,
        'failure_prone': failure_prone
    }


def visualize_results(results):
    """Create visualizations of simulation results"""
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 18))

    # 1. Gantt chart for product flow
    plt.subplot(4, 1, 1)
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
    plt.subplot(4, 1, 2)

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

    # 3. Waiting Time vs Service Time comparison
    plt.subplot(4, 1, 3)

    # Add a new visualization for waiting time vs service time per product
    bar_width = 0.35
    products = product_df['Product']
    waiting_times = product_df['Total Waiting Time']
    service_times = product_df['Total Service Time']

    x = np.arange(len(products))

    plt.bar(x - bar_width/2, waiting_times, bar_width, label='Waiting Time', color='salmon')
    plt.bar(x + bar_width/2, service_times, bar_width, label='Service Time', color='skyblue')

    plt.xlabel('Product')
    plt.ylabel('Time')
    plt.title('Waiting Time vs Service Time by Product')
    plt.xticks(x, products, rotation=90)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # 4. System Dynamics - Products in System Over Time
    plt.subplot(4, 1, 4)

    # Extract system dynamics data from event log
    event_df = results['event_log']

    # Find arrival and completion events to track system inventory
    arrivals = event_df[event_df['Event'].str.contains('ARRIVED')]
    completions = event_df[event_df['Event'].str.contains('FINISHED')]

    # Create timeline of system state
    timeline = []
    products_in_system = 0

    # Add arrival events
    for _, row in arrivals.iterrows():
        timeline.append((row['Time'], 1))  # +1 for arrivals

    # Add completion events
    for _, row in completions.iterrows():
        timeline.append((row['Time'], -1))  # -1 for completions

    # Sort by time
    timeline.sort(key=lambda x: x[0])

    # Generate cumulative count
    times = [0]  # Start at time 0
    counts = [0]  # Start with 0 products

    for time, change in timeline:
        # Add point just before change
        times.append(time)
        counts.append(counts[-1])

        # Add point after change
        times.append(time)
        counts.append(counts[-1] + change)

    plt.step(times, counts, where='post', color='blue', linewidth=2)
    plt.fill_between(times, counts, step='post', alpha=0.3, color='blue')
    plt.title('Products in System Over Time')
    plt.xlabel('Simulation Time')
    plt.ylabel('Number of Products')
    plt.grid(alpha=0.3)

    # Add machine failure periods as shaded areas
    failure_starts = machine_state_df[machine_state_df['State'] == 'FAILED']
    failure_ends = machine_state_df[machine_state_df['State'] == 'REPAIRED']

    failure_periods = []
    for machine in ['M1', 'M2', 'M3']:
        machine_failures = failure_starts[failure_starts['Machine'] == machine]
        machine_repairs = failure_ends[failure_ends['Machine'] == machine]

        # Match failures with repairs
        if len(machine_failures) == len(machine_repairs):
            for i in range(len(machine_failures)):
                start_time = machine_failures.iloc[i]['Time']
                end_time = machine_repairs.iloc[i]['Time']
                failure_periods.append((start_time, end_time, machine))

    # Color mapping for machines
    machine_colors = {'M1': 'red', 'M2': 'orange', 'M3': 'purple'}

    # Add shaded regions for machine failures
    for start, end, machine in failure_periods:
        plt.axvspan(start, end, alpha=0.2, color=machine_colors[machine],
                   label=f'{machine} Failure' if machine not in plt.gca().get_legend_handles_labels()[1] else "")

    # If we have failure periods, add a legend
    if failure_periods:
        plt.legend()

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

    # Display step-by-step event progression (at least 20 events)
    print("SIMULATION EVENT PROGRESSION:")
    print("-----------------------------")
    event_df = results['event_log'].head(30)  # Show at least the first 30 events

    # Create a snapshot of system state at each event
    system_snapshots = []

    # Start with initial state
    current_state = {
        'Time': 0,
        'Event': 'SIMULATION START',
        'M1_State': 'Idle',
        'M2_State': 'Idle',
        'M3_State': 'Idle',
        'M1_Queue': 0,
        'M2_Queue': 0,
        'M3_Queue': 0,
        'Products_In_System': 0,
        'Products_Completed': 0
    }
    system_snapshots.append(current_state.copy())

    # Process each event to update system state
    products_in_system = {}  # Track products currently in system
    machine_states_dict = {'M1': 'Idle', 'M2': 'Idle', 'M3': 'Idle'}
    machine_queues = {'M1': 0, 'M2': 0, 'M3': 0}
    products_completed = 0

    for _, row in event_df.iterrows():
        time = row['Time']
        event = row['Event']

        # Deep copy previous state as starting point
        current_state = system_snapshots[-1].copy()
        current_state['Time'] = time
        current_state['Event'] = event

        # Update system state based on event
        parts = event.split()
        if len(parts) >= 2:
            product = parts[0]
            action = parts[1]

            # Product arrival
            if action == 'ARRIVED':
                products_in_system[product] = 'Arrived'
                current_state['Products_In_System'] = len(products_in_system)
                machine_queues['M1'] += 1
                current_state['M1_Queue'] = machine_queues['M1']

            # Product starts processing on a machine
            elif action == 'START':
                machine = parts[2]
                products_in_system[product] = f'Processing-{machine}'
                machine_states_dict[machine] = f'Processing {product}'
                machine_queues[machine] -= 1
                current_state[f'{machine}_State'] = machine_states_dict[machine]
                current_state[f'{machine}_Queue'] = machine_queues[machine]

            # Product ends processing on a machine
            elif action == 'END':
                machine = parts[2]
                machine_states_dict[machine] = 'Idle'
                current_state[f'{machine}_State'] = machine_states_dict[machine]

                # If finished with M3, product is complete
                if machine == 'M3':
                    products_in_system.pop(product, None)
                    products_completed += 1
                    current_state['Products_Completed'] = products_completed
                    current_state['Products_In_System'] = len(products_in_system)
                else:
                    # Move to next machine queue
                    next_machine = 'M2' if machine == 'M1' else 'M3'
                    machine_queues[next_machine] += 1
                    current_state[f'{next_machine}_Queue'] = machine_queues[next_machine]
                    products_in_system[product] = f'Queued-{next_machine}'

            # Product finished entire process
            elif action == 'FINISHED,':
                # Make sure it's removed from system if not already
                if product in products_in_system:
                    products_in_system.pop(product)
                    products_completed += 1
                    current_state['Products_Completed'] = products_completed
                    current_state['Products_In_System'] = len(products_in_system)

            # Machine failed
            elif action == 'FAILED':
                machine = parts[0]
                machine_states_dict[machine] = 'Failed'
                current_state[f'{machine}_State'] = machine_states_dict[machine]

            # Machine repaired
            elif action == 'REPAIRED':
                machine = parts[0]
                machine_states_dict[machine] = 'Idle'
                current_state[f'{machine}_State'] = machine_states_dict[machine]

            # Waiting for repair
            elif action == 'WAITING':
                machine = parts[3]
                products_in_system[product] = f'Waiting-{machine}-repair'

        # Add the updated state snapshot
        system_snapshots.append(current_state.copy())

    # Display step-by-step system state changes
    snapshot_df = pd.DataFrame(system_snapshots)
    pd.set_option('display.max_rows', None)  # Show all rows
    print(tabulate(snapshot_df, headers='keys', tablefmt='pretty', showindex=False))
    print("\n")

    print("MACHINE METRICS:")
    print(tabulate(results['machine_metrics'], headers='keys', tablefmt='pretty', showindex=False))

    print("\nSYSTEM METRICS:")
    system_metrics_df = pd.DataFrame([results['system_metrics']])
    print(tabulate(system_metrics_df, headers='keys', tablefmt='pretty', showindex=False))

    print("\nPRODUCT METRICS:")
    # Select key columns for cleaner display including the new metrics
    product_display = results['product_metrics'][['Product', 'Arrival', 'Completion', 'Lead Time',
                                                'Total Waiting Time', 'Total Service Time',
                                                'M1 Queue Time', 'M1 Processing Time',
                                                'M2 Queue Time', 'M2 Processing Time',
                                                'M3 Queue Time', 'M3 Processing Time']]
    print(tabulate(product_display, headers='keys', tablefmt='pretty', showindex=False))

    print("\nKEY FINDINGS:")
    print(f"- Bottleneck machine: {results['bottleneck']}")
    print(f"- Most failure-prone machine: {results['failure_prone']}")
    print(f"- Average waiting time: {results['system_metrics']['Average Waiting Time']:.2f}")
    print(f"- Average service time: {results['system_metrics']['Average Service Time']:.2f}")
    print(f"- Waiting time percentage: {results['system_metrics']['Waiting Time Percentage']:.2f}%")

    # Create and display visualizations
    fig = visualize_results(results)
    plt.show()

    return results


# Run the simulation and display results
if __name__ == "__main__":
    results = run_and_display()