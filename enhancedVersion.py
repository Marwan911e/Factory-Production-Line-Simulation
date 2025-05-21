import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from tabulate import tabulate
import jinja2
import webbrowser

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
interarrival_times = []  # Track time between arrivals
arrival_times = []  # Track arrival timestamps
active_products = 0  # Track number of products currently in the system

class Machine:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.resource = simpy.Resource(env, capacity=2)
        self.broken = False
        self.total_downtime = 0
        self.last_failure_time = 0
        self.utilization_time = 0
        self.processing_count = 0
        self.failures = 0
        self.queue_length = 0
        
        # Queueing theory metrics
        self.total_waiting_time = 0  # Total time products waited in queue
        self.products_that_waited = 0  # Count of products that had to wait
        self.sum_queue_length = 0  # Sum of queue lengths over time
        self.queue_length_samples = 0  # Number of samples for queue length
        self.total_service_time = 0  # Total time spent processing products
        self.total_idle_time = 0  # Total time machine was idle
        self.last_departure_time = 0  # Time when last product departed
        self.last_busy_state_change = 0  # Time of last change in busy status
        self.busy = False  # Whether machine is currently busy
        
        env.process(self.break_machine())
        env.process(self.monitor_queue())

    def break_machine(self):
        """Process that simulates random machine failures and repairs"""
        while True:
            yield self.env.timeout(failure_time())
            if not self.broken and self.busy:  # Only break if machine is in use
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
    
    def monitor_queue(self):
        """Process to monitor queue length over time"""
        while True:
            # Sample the queue length periodically
            self.sum_queue_length += len(self.resource.queue)
            self.queue_length_samples += 1
            yield self.env.timeout(1)  # Sample every time unit
    
    def update_busy_status(self, new_busy_state):
        """Update busy/idle status and calculate idle time"""
        current_time = self.env.now
        if self.busy and not new_busy_state:  # Becoming idle
            self.busy = False
            # No idle time to add when going from busy to idle
        elif not self.busy and new_busy_state:  # Becoming busy
            # Add idle time from last state change until now
            self.total_idle_time += current_time - self.last_busy_state_change
            self.busy = True
        
        self.last_busy_state_change = current_time

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
        
        # Track number of products in system
        global active_products
        active_products += 1
        
        # Record arrival time
        if len(arrival_times) > 0:
            # Calculate interarrival time
            interarrival_times.append(self.start_time - arrival_times[-1])
        arrival_times.append(self.start_time)
        
        env.process(self.process())

    def process(self):
        arrival_time = self.env.now
        log(f"{self.name} ARRIVED", self.env.now)

        for machine in self.machines:
            queue_start = self.env.now
            machine.queue_length += 1

            with machine.resource.request() as request:
                # Update machine busy status
                machine.update_busy_status(True)
                
                # Wait for the resource to be available
                yield request
                
                # Calculate queue time
                queue_time = self.env.now - queue_start
                self.queue_times[machine.name] = queue_time
                machine.queue_length -= 1
                
                # Update queueing metrics
                machine.total_waiting_time += queue_time
                if queue_time > 0:
                    machine.products_that_waited += 1

                # Wait if machine is broken
                while machine.broken:
                    log(f"{self.name} WAITING for {machine.name} repair", self.env.now)
                    yield self.env.timeout(1)

                # Start processing
                log(f"{self.name} START {machine.name}", self.env.now)
                start_time = self.env.now
                remaining_time = processing_time(machine.name)
                self.processing_times[machine.name] = remaining_time
                
                # Update service time metrics
                machine.total_service_time += remaining_time

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
                machine.last_departure_time = self.env.now
                log(f"{self.name} END {machine.name}", self.env.now)
                self.machine_times[machine.name] = self.env.now - start_time
                
                # Update machine busy status
                machine.update_busy_status(False)

        self.end_time = self.env.now
        total_time = self.end_time - arrival_time
        log(f"{self.name} FINISHED, total time: {total_time:.2f}", self.env.now)
        
        # Decrement active products count when product finishes
        global active_products
        active_products -= 1

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
            'M3 Time': self.machine_times.get('M3', 0),
            'System Time': total_time  # Total time in system
        })

def product_arrival(env, machines, total_products=100):
    """Generate product arrivals"""
    i = 0
    while i < total_products:
        # Wait for interarrival time
        yield env.timeout(interarrival_time())

        # Create new product
        Product(env, f"Product-{i}", machines)
        i += 1
        
        # Log progress periodically
        if i % 10 == 0:
            log(f"GENERATED {i}/{total_products} products", env.now)


def log(message, time):
    """Add an event to the log"""
    event_log.append((time, message))

def run_simulation(sim_time=None, total_products=100):
    """Run the simulation until all products are processed
    
    Parameters:
    - sim_time: Maximum simulation time (if None, runs until all products are processed)
    - total_products: Total number of products to generate
    """
    global event_log, product_metrics, machine_states
    global interarrival_times, arrival_times, active_products

    # Reset all trackers
    event_log = []
    product_metrics = []
    machine_states = []
    interarrival_times = []
    arrival_times = []
    active_products = 0

    # Initialize simulation environment
    env = simpy.Environment()
    machines = [Machine(env, "M1"), Machine(env, "M2"), Machine(env, "M3")]

    # Start the product arrival process with specified total products
    env.process(product_arrival(env, machines, total_products))

    # If no specific simulation time is given, run until all products finish processing
    if sim_time is None:
        # First, run until all products are generated (based on expected arrival time)
        # Use estimated time: mean interarrival time (5) * total_products + buffer time (50)
        estimated_arrival_time = 5 * total_products + 50
        env.run(until=estimated_arrival_time)
        
        # Then run until all active products are processed
        while active_products > 0:
            # Run in smaller increments and check active_products
            env.run(until=env.now + 50)
            log(f"TIME: {env.now:.1f}, REMAINING PRODUCTS: {active_products}", env.now)
            
            # Safety check - if we've run for too long, stop
            if env.now > 10000:  # Set a reasonable maximum simulation time
                log("SIMULATION TIMEOUT - Some products may not have completed", env.now)
                break
    else:
        # Run for the specified simulation time
        env.run(until=sim_time)

    # Add explanation about utilization and waiting time to the event log
    explanation = (
        "NOTE: Even with moderate utilization rates (e.g., 67%), significant waiting times can occur. "
        "In queueing theory, as utilization increases, waiting times grow exponentially, not linearly. "
        "For example, at 67% utilization, queue length and waiting time can already be substantial, "
        "and they increase dramatically as utilization approaches 100%."
    )
    log(explanation, env.now)
    
    return machines, env.now

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
    
    # System-wide metrics
    system_queueing_metrics = {
        'Average Waiting Time': total_system_wait / total_products if total_products > 0 else 0,
        'Probability of Wait': products_that_waited / total_products if total_products > 0 else 0,
        'Average Queue Length': sum(m.sum_queue_length for m in machines) / sum(m.queue_length_samples for m in machines) if sum(m.queue_length_samples for m in machines) > 0 else 0,
        'Average Service Time': product_df['Total Service Time'].mean(),
        'Average Interarrival Time': avg_interarrival_time,
        'Average Waiting Time (Those Who Wait)': total_system_wait / products_that_waited if products_that_waited > 0 else 0,
        'Average Time in System': product_df['Lead Time'].mean(),
        'Server Utilization': min(1.0, sum(min(1.0, (m.utilization_time / sim_time) / m.resource.capacity) for m in machines) / len(machines)) if sim_time > 0 else 0
    }
    
    # Machine metrics table with queueing theory metrics
    machine_data = []
    for m in machines:
        utilization_pct = (m.utilization_time / (sim_time * m.resource.capacity)) * 100 if sim_time > 0 else 0

        availability_pct = ((sim_time - m.total_downtime) / sim_time) * 100 if sim_time > 0 else 0
        
        # Calculate queueing metrics for this machine
        avg_waiting_time = m.total_waiting_time / total_products if total_products > 0 else 0
        prob_wait = m.products_that_waited / total_products if total_products > 0 else 0
        avg_queue_length = m.sum_queue_length / m.queue_length_samples if m.queue_length_samples > 0 else 0
        prob_idle = m.total_idle_time / sim_time if sim_time > 0 else 0
        avg_service_time = m.total_service_time / total_products if total_products > 0 else 0
        avg_wait_those_who_wait = m.total_waiting_time / m.products_that_waited if m.products_that_waited > 0 else 0
        server_utilization = min(1.0, (m.utilization_time / sim_time) / m.resource.capacity) if sim_time > 0 else 0


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

    return {
        'event_log': event_df,
        'product_metrics': product_df,
        'machine_metrics': machine_df,
        'system_metrics': system_metrics,
        'queue_metrics': system_queueing_metrics,
        'bottleneck': bottleneck,
        'failure_prone': failure_prone
    }

def select_display_products(results, num_to_display=20, selection_method='first'):
    """Select a subset of products to display in reports"""
    all_products_df = results['product_metrics']
    
    if len(all_products_df) == 0:
        return results  # Return unchanged if no products
    
    if selection_method == 'first':
        # Sort by product name numerically (extract the number from Product-X)
        # This ensures Product-0 comes before Product-1, etc.
        all_products_df['product_num'] = all_products_df['Product'].str.extract(r'Product-(\d+)').astype(int)
        all_products_df = all_products_df.sort_values('product_num')
        display_products = all_products_df.head(num_to_display)
        # Drop the temporary column used for sorting
        display_products = display_products.drop('product_num', axis=1)
    elif selection_method == 'last':
        # Select last n products
        display_products = all_products_df.tail(num_to_display)
    elif selection_method == 'random':
        # Select n random products
        if len(all_products_df) > num_to_display:
            display_products = all_products_df.sample(num_to_display)
        else:
            display_products = all_products_df
    elif selection_method == 'representative':
        # Try to select a representative sample (spread across time)
        if len(all_products_df) > num_to_display:
            step = len(all_products_df) // num_to_display
            indices = [i * step for i in range(num_to_display)]
            display_products = all_products_df.iloc[indices]
        else:
            display_products = all_products_df
    else:
        # Default to first n products
        display_products = all_products_df.head(num_to_display)
    
    # Create a copy of the results with only selected products for display
    display_results = results.copy()
    display_results['display_product_metrics'] = display_products
    
    return display_results

# Visualization functions
def create_simple_machine_chart(results):
    """Create a simple bar chart showing machine utilization and availability"""
    machine_df = results['machine_metrics']
    machines = machine_df['Machine']
    utilization = machine_df['Utilization (%)']
    availability = machine_df['Availability (%)']
    failures = machine_df['Number of Failures']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.4
    
    # Set position of bars on x axis
    r1 = np.arange(len(machines))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    util_bars = ax.bar(r1, utilization, width=bar_width, label='Utilization (%)', color='skyblue')
    avail_bars = ax.bar(r2, availability, width=bar_width, label='Availability (%)', color='lightgreen')
    
    # Highlight bottleneck machine
    bottleneck_idx = list(machines).index(results['bottleneck'])
    util_bars[bottleneck_idx].set_color('red')
    
    # Highlight failure-prone machine
    failure_idx = list(machines).index(results['failure_prone'])
    avail_bars[failure_idx].set_color('orange')
    
    # Add labels and title
    ax.set_xlabel('Machine')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Machine Performance Analysis')
    ax.set_xticks([r + bar_width/2 for r in range(len(machines))])
    ax.set_xticklabels(machines)
    ax.set_ylim(0, 110)  # Set y-axis range to 0-100% with some padding
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Add annotations
    for i, (util, avail, fail) in enumerate(zip(utilization, availability, failures)):
        ax.text(r1[i], util + 2, f"{util:.1f}%", ha='center', va='bottom')
        ax.text(r2[i], avail + 2, f"{avail:.1f}%", ha='center', va='bottom')
        ax.text(r2[i], avail/2, f"{fail} failures", ha='center', va='center')
    
    # Add bottleneck annotation
    ax.annotate('BOTTLENECK', xy=(r1[bottleneck_idx], utilization[bottleneck_idx]),
                xytext=(r1[bottleneck_idx], utilization[bottleneck_idx] + 15),
                arrowprops=dict(facecolor='red', shrink=0.05),
                ha='center', color='red')
    
    plt.tight_layout()
    return fig

def create_simple_timeline_chart(results):
    """Create a simple timeline showing product flow through the system"""
    # Use display products instead of all products
    product_df = results['display_product_metrics'].sort_values('Arrival')
    
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size to accommodate more products
    
    # Plot each product's journey
    products = []
    y_positions = []
    
    for i, row in enumerate(product_df.itertuples()):
        product_name = row.Product
        products.append(product_name)
        y_pos = i + 1
        y_positions.append(y_pos)
        
        # Plot the entire journey line
        ax.plot([row.Arrival, row.Completion], [y_pos, y_pos], 'gray', linewidth=2)
        
        # Plot arrival point
        ax.plot(row.Arrival, y_pos, 'bo', markersize=6)  # Reduced marker size
        
        # Plot completion point
        ax.plot(row.Completion, y_pos, 'go', markersize=6)  # Reduced marker size
        
        # Add text annotation with breakdown
        wait_time = row._9  # Total Waiting Time index
        proc_time = row._10  # Total Service Time index
        mid_point = (row.Arrival + row.Completion) / 2
        ax.text(mid_point, y_pos + 0.2, f"Wait: {wait_time:.1f} | Process: {proc_time:.1f}", 
                ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7), fontsize=8)
    
    # Set labels and title
    ax.set_yticks(y_positions)
    ax.set_yticklabels(products)
    ax.set_xlabel('Simulation Time')
    ax.set_ylabel('Product')
    ax.set_title(f'Product Timeline: First {len(product_df)} Products from {len(results["product_metrics"])} Total')  # Updated title
    
    # Add legend
    ax.plot([], [], 'bo', markersize=6, label='Arrival')
    ax.plot([], [], 'go', markersize=6, label='Completion')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def create_simple_machine_status_bar(results):
    """Create a simple bar chart showing machine uptime vs downtime"""
    # Create a DataFrame from the machine metrics
    machine_df = results['machine_metrics']
    
    # Get the simulation duration
    sim_end = results['event_log']['Time'].max()
    
    # Calculate uptime (based on availability percentage and total time)
    machines = machine_df['Machine']
    availability = machine_df['Availability (%)']
    failures = machine_df['Number of Failures']
    
    # Calculate uptime and downtime
    uptime = [(avail/100) * sim_end for avail in availability]
    downtime = [sim_end - up for up in uptime]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create stacked bar chart
    ax.barh(machines, uptime, color='green', label='Uptime')
    ax.barh(machines, downtime, left=uptime, color='red', label='Downtime')
    
    # Add annotations
    for i, (machine, up, down, fail) in enumerate(zip(machines, uptime, downtime, failures)):
        # Add uptime percentage
        ax.text(up/2, i, f"{availability[i]:.1f}% available", 
                ha='center', va='center', color='white', fontweight='bold')
        
        # Add failure count if downtime exists
        if down > 0:
            ax.text(up + down/2, i, f"{fail} failures", 
                    ha='center', va='center', color='white', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Simulation Time')
    ax.set_title('Machine Uptime vs. Downtime')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def create_simple_waiting_time_chart(results):
    """Create a simple stacked bar chart showing waiting time vs processing time by machine"""
    # Use all products for calculating averages
    product_df = results['product_metrics']
    
    # Calculate average wait and processing times for each machine
    machine_times = []
    machines = []
    wait_times = []
    proc_times = []
    total_times = []
    
    for machine in ['M1', 'M2', 'M3']:
        wait_col = f'{machine} Queue Time'
        proc_col = f'{machine} Processing Time'
        
        avg_wait = product_df[wait_col].mean()
        avg_proc = product_df[proc_col].mean()
        
        machines.append(machine)
        wait_times.append(avg_wait)
        proc_times.append(avg_proc)
        total_times.append(avg_wait + avg_proc)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.5
    
    # Set position of bars on x axis
    r = np.arange(len(machines))
    
    # Create stacked bars
    ax.bar(r, wait_times, width=bar_width, label='Wait Time', color='#ff9999')
    ax.bar(r, proc_times, width=bar_width, bottom=wait_times, label='Processing Time', color='#66b3ff')
    
    # Add labels
    ax.set_xlabel('Machine')
    ax.set_ylabel('Time')
    ax.set_title('Average Wait Time vs. Processing Time by Machine')
    ax.set_xticks(r)
    ax.set_xticklabels(machines)
    ax.legend()
    
    # Add text annotations
    for i in range(len(machines)):
        # Wait time label
        if wait_times[i] >= 1:  # Only add label if there's enough space
            ax.text(i, wait_times[i]/2, f"{wait_times[i]:.1f}", ha='center', va='center')
        
        # Process time label
        if proc_times[i] >= 1:
            ax.text(i, wait_times[i] + proc_times[i]/2, f"{proc_times[i]:.1f}", ha='center', va='center')
        
        # Total time label
        ax.text(i, total_times[i] + 0.5, f"Total: {total_times[i]:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_product_metrics_table_chart(results):
    """Create a visualization of product metrics"""
    # Use display products
    product_df = results['display_product_metrics'].sort_values('Arrival')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 1. First chart: Lead time breakdown for each product
    products = product_df['Product']
    waiting_times = product_df['Total Waiting Time']
    processing_times = product_df['Total Service Time']
    
    # Create stacked bars
    y_pos = np.arange(len(products))
    ax1.barh(y_pos, waiting_times, color='#ff9999', label='Waiting Time')
    ax1.barh(y_pos, processing_times, left=waiting_times, color='#66b3ff', label='Processing Time')
    
    # Add annotations for total lead time
    for i, (wait, proc) in enumerate(zip(waiting_times, processing_times)):
        total = wait + proc
        ax1.text(total + 0.5, i, f"Total: {total:.1f}", va='center')
    
    # Customize chart
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(products)
    ax1.set_xlabel('Time')
    ax1.set_title(f'Lead Time Breakdown for First {len(product_df)} Products')
    ax1.legend(loc='upper right')
    
    # 2. Second chart: Machine-specific processing times for each product
    # Extract machine-specific times
    m1_times = product_df['M1 Time']
    m2_times = product_df['M2 Time']
    m3_times = product_df['M3 Time']
    
    # Create grouped bars
    width = 0.25
    ax2.barh(y_pos - width, m1_times, width, color='lightblue', label='M1 Time')
    ax2.barh(y_pos, m2_times, width, color='lightgreen', label='M2 Time')
    ax2.barh(y_pos + width, m3_times, width, color='lightsalmon', label='M3 Time')
    
    # Customize chart
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(products)
    ax2.set_xlabel('Time')
    ax2.set_title(f'Processing Time by Machine for First {len(product_df)} Products')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

def create_comparison_chart(results):
    """Create a chart comparing statistics of all products vs displayed products"""
    all_df = results['product_metrics']
    display_df = results['display_product_metrics']
    
    metrics = ['Lead Time', 'Total Waiting Time', 'Total Service Time']
    all_means = [all_df[metric].mean() for metric in metrics]
    display_means = [display_df[metric].mean() for metric in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create grouped bars
    ax.bar(x - width/2, all_means, width, label=f'All Products (n={len(all_df)})', color='skyblue')
    ax.bar(x + width/2, display_means, width, label=f'First {len(display_df)} Products', color='lightgreen')
    
    # Add annotations
    for i, (all_val, disp_val) in enumerate(zip(all_means, display_means)):
        ax.text(i - width/2, all_val + 0.5, f"{all_val:.1f}", ha='center', va='bottom')
        ax.text(i + width/2, disp_val + 0.5, f"{disp_val:.1f}", ha='center', va='bottom')
    
    # Customize chart
    ax.set_ylabel('Time')
    ax.set_title('Comparison: All Products vs. First 20 Products')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_queueing_metrics_chart(results):
    """Create a chart showing queueing theory metrics for each machine"""
    machine_df = results['machine_metrics']
    
    # Select relevant queueing metrics
    metrics = ['Avg Waiting Time', 'Probability of Wait', 'Avg Queue Length', 
              'Probability of Idle', 'Avg Service Time', 'Server Utilization']
    
    # Prepare data
    machines = machine_df['Machine']
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = machine_df[metric]
        
        # Create bar chart
        bars = ax.bar(machines, values, color=['skyblue', 'lightgreen', 'lightsalmon'])
        
        # Add title and labels
        ax.set_title(metric)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)  # Add some space for labels
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def save_charts_to_directory(results):
    """Save all charts to a directory instead of creating an HTML report"""
    # Create reports directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # Create subdirectory with timestamp for this report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"reports/factory_simulation_{timestamp}"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Generate and save each chart
    charts = {
        'machine_performance': create_simple_machine_chart(results),
        'product_timeline': create_simple_timeline_chart(results),
        'machine_uptime': create_simple_machine_status_bar(results),
        'waiting_time': create_simple_waiting_time_chart(results),
        'product_metrics': create_product_metrics_table_chart(results),
        'comparison_chart': create_comparison_chart(results),
        'queueing_metrics': create_queueing_metrics_chart(results)
    }
    
    # Save each chart as PNG
    for name, fig in charts.items():
        filepath = os.path.join(report_dir, f"{name}.png")
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    # Create a simple text summary file
    summary_path = os.path.join(report_dir, "simulation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("FACTORY SIMULATION SUMMARY\n")
        f.write("=========================\n\n")
        
        # Total products info
        f.write(f"Total Products Simulated: {len(results['product_metrics'])}\n")
        f.write(f"Products Displayed in Tables: {len(results['display_product_metrics'])}\n\n")
        
        # System metrics
        f.write("SYSTEM METRICS:\n")
        for key, value in results['system_metrics'].items():
            f.write(f"{key}: {value}\n")
        
        # Queueing theory metrics
        f.write("\nQUEUEING THEORY METRICS:\n")
        for key, value in results['queue_metrics'].items():
            f.write(f"{key}: {round(value, 2)}\n")
        f.write("\n")
        
        # Bottleneck information
        bottleneck = results['bottleneck']
        bottleneck_util = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == bottleneck, 'Utilization (%)'].values[0]
        f.write(f"BOTTLENECK: {bottleneck} (Utilization: {bottleneck_util:.2f}%)\n")
        
        # Failure information
        failure_prone = results['failure_prone']
        failure_count = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == failure_prone, 'Number of Failures'].values[0]
        f.write(f"MOST FAILURES: {failure_prone} (Count: {failure_count})\n")
        
        # Product metrics table (first 20 products only)
        f.write("\nPRODUCT METRICS:\n")
        product_table = tabulate(
            results['display_product_metrics'].sort_values('Arrival').round(2), 
            headers='keys', 
            tablefmt='grid', 
            showindex=False
        )
        f.write(product_table)
    
    return report_dir

def prepare_all_products_table(results):
    """Prepare HTML table for all products"""
    # Convert all products DataFrame to HTML table
    all_products_df = results['product_metrics'].round(2)
    all_products_table = all_products_df.to_html(classes='table table-striped table-hover', index=False)
    return all_products_table

def create_html_report(results):
    """Create a comprehensive HTML report with all analysis results"""
    # Create report directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    # Get current timestamp for report name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"reports/factory_simulation_report_{timestamp}.html"
    
    # Generate visualization figures and save them
    report_dir = os.path.dirname(report_filename)
    
    # Save charts as images
    machine_chart_path = os.path.join(report_dir, "machine_chart.png")
    timeline_chart_path = os.path.join(report_dir, "timeline_chart.png") 
    machine_status_path = os.path.join(report_dir, "machine_status.png")
    waiting_time_chart_path = os.path.join(report_dir, "waiting_time_chart.png")
    product_metrics_chart_path = os.path.join(report_dir, "product_metrics_chart.png")
    comparison_chart_path = os.path.join(report_dir, "comparison_chart.png")
    queueing_metrics_path = os.path.join(report_dir, "queueing_metrics.png")
    
    # Create and save each chart
    create_simple_machine_chart(results).savefig(machine_chart_path, dpi=100, bbox_inches='tight')
    create_simple_timeline_chart(results).savefig(timeline_chart_path, dpi=100, bbox_inches='tight')
    create_simple_machine_status_bar(results).savefig(machine_status_path, dpi=100, bbox_inches='tight')
    create_simple_waiting_time_chart(results).savefig(waiting_time_chart_path, dpi=100, bbox_inches='tight')
    create_product_metrics_table_chart(results).savefig(product_metrics_chart_path, dpi=100, bbox_inches='tight')
    create_comparison_chart(results).savefig(comparison_chart_path, dpi=100, bbox_inches='tight')
    create_queueing_metrics_chart(results).savefig(queueing_metrics_path, dpi=100, bbox_inches='tight')
    
    # Close all plots to free memory
    plt.close('all')
    
    # Convert the DataFrames to HTML tables - use display products only for product table
    product_table = results['display_product_metrics'].round(2).to_html(classes='table table-striped table-hover', index=False)
    machine_table = results['machine_metrics'].round(2).to_html(classes='table table-striped table-hover', index=False)
    system_metrics = pd.DataFrame([results['system_metrics']]).round(2)
    system_table = system_metrics.to_html(classes='table table-striped table-hover', index=False)
    queueing_metrics = pd.DataFrame([results['queue_metrics']]).round(2)
    queueing_table = queueing_metrics.to_html(classes='table table-striped table-hover', index=False)
    
    # Prepare the table with all products
    all_products_table = prepare_all_products_table(results)
    
    # Convert first 20 events to HTML table
    event_table = results['event_log'].head(20).to_html(classes='table table-striped table-hover', index=False)
    
    # Create HTML template using Jinja2
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Factory Production Line Simulation Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px;
                padding-bottom: 50px;
                color: #333;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 30px;
                background-color: #f0f2f5;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }
            .section {
                margin-bottom: 40px;
                padding: 25px;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 3px 15px rgba(0,0,0,0.08);
            }
            .highlight-box {
                background-color: #f8f9fa;
                border-left: 5px solid #0d6efd;
                padding: 20px;
                margin: 25px 0;
                border-radius: 6px;
            }
            .alert-box {
                padding: 15px;
                margin: 15px 0;
                border-radius: 6px;
            }
            .alert-box.bottleneck {
                background-color: rgba(255, 0, 0, 0.1);
                border-left: 5px solid #dc3545;
            }
            .alert-box.failure {
                background-color: rgba(255, 193, 7, 0.1);
                border-left: 5px solid #ffc107;
            }
            .table-container {
                overflow-x: auto;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                border-radius: 6px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
            }
            .key-metric {
                border-radius: 8px;
                padding: 25px;
                margin: 15px;
                text-align: center;
                background-color: #f8f9fa;
                box-shadow: 0 3px 10px rgba(0,0,0,0.08);
                transition: all 0.2s ease;
            }
            .key-metric:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0,0,0,0.15);
            }
            .key-metric h3 {
                margin: 0;
                font-size: 2.5rem;
                font-weight: 600;
                color: #0d6efd;
            }
            .key-metric p {
                margin: 10px 0 0 0;
                color: #6c757d;
                font-weight: 500;
            }
            .recommendation {
                background-color: #e8f4ff;
                border-left: 5px solid #0d6efd;
                padding: 15px;
                margin: 20px 0;
                border-radius: 6px;
            }
            img.chart {
                width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin: 15px 0;
            }
            .info-banner {
                background-color: #e8f4ff;
                padding: 10px 15px;
                margin: 10px 0;
                border-radius: 6px;
                font-weight: 500;
                color: #0d6efd;
            }
            .formula-box {
                background-color: #f8f9fa;
                padding: 10px;
                margin: 10px 0;
                border-radius: 6px;
                border-left: 5px solid #6c757d;
                font-family: monospace;
            }
            /* Button styles */
            .view-all-btn {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
                margin: 10px 0;
                transition: all 0.2s ease;
            }
            .view-all-btn:hover {
                background-color: #0b5ed7;
            }
            /* Hide the full product table by default */
            #fullProductTable {
                display: none;
            }
            @media print {
                body { 
                    margin: 0;
                    padding: 0;
                }
                .section {
                    page-break-inside: avoid;
                    box-shadow: none;
                    border: 1px solid #ddd;
                }
            }
        </style>
        <script>
            function toggleProductTable() {
                var displayTable = document.getElementById('displayProductTable');
                var fullTable = document.getElementById('fullProductTable');
                var button = document.getElementById('toggleButton');
                
                if (fullTable.style.display === 'none' || fullTable.style.display === '') {
                    displayTable.style.display = 'none';
                    fullTable.style.display = 'block';
                    button.innerText = 'View Limited Products (' + {{ display_products }} + ')';
                } else {
                    displayTable.style.display = 'block';
                    fullTable.style.display = 'none';
                    button.innerText = 'View All Products (' + {{ total_products }} + ')';
                }
            }
        </script>
    </head>
    <body>
        <div class="header">
            <h1>Factory Production Line Simulation Report</h1>
            <p class="text-muted">Generated on {{ timestamp }}</p>
            <div class="info-banner">
                <p>Simulation run with {{ total_products }} products</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="highlight-box">
                <p>This simulation models a factory production line with three machines in series (M1, M2, M3). 
                Each machine processes products sequentially, and machines can experience random failures.</p>
                
                <div class="alert-box bottleneck">
                    <h4>⚠️ Bottleneck Identified: <strong>{{ bottleneck }}</strong></h4>
                    <p>This machine has the highest utilization ({{ bottleneck_util }}%) and is limiting the overall system throughput.</p>
                </div>
                
                <div class="alert-box failure">
                    <h4>⚙️ Reliability Issue: <strong>{{ failure_prone }}</strong></h4>
                    <p>This machine experienced the most failures during the simulation, with {{ failure_count }} breakdowns.</p>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-3">
                    <div class="key-metric">
                        <h3>{{ throughput }}</h3>
                        <p>Products per Time Unit</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="key-metric">
                        <h3>{{ avg_lead_time }}s</h3>
                        <p>Average Lead Time</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="key-metric">
                        <h3>{{ efficiency }}%</h3>
                        <p>System Efficiency</p>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="key-metric">
                        <h3>{{ server_util }}%</h3>
                        <p>Server Utilization</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Queueing Theory Metrics</h2>
            <p>This section presents metrics calculated using queueing theory principles.</p>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="formula-box">
                        <p>Average Waiting Time = Total time Customer wait in queue / Total number of Customers</p>
                        <p>Probability (Wait) = number of customer who wait(count) / Total number of customer</p>
                        <p>Queue length = sum of all in queue / total number of customer</p>
                        <p>Probability of idle = Total idle time of Server / Total run time in Simulation</p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="formula-box">
                        <p>Average Service Time = Total Service Time / Total number of customer</p>
                        <p>Average time between arrivals = sum of interarrival times / total number of Customers -1</p>
                        <p>Average waiting time for those who wait = Total time Customer wait in queue / Total number of Customers who wait(count)</p>
                        <p>Server Utilization = 1 - probability of idle</p>
                    </div>
                </div>
            </div>
            
            <div class="table-container mt-4">
                <h5>System-Wide Queueing Metrics</h5>
                {{ queueing_table|safe }}
            </div>
            
            <p class="mt-4">The chart below shows queueing theory metrics for each individual machine:</p>
            <img src="queueing_metrics.png" alt="Queueing Theory Metrics by Machine" class="chart">
            
            <div class="recommendation">
                <h4>💡 Queueing Theory Insight:</h4>
                <p>The probability of wait and queue length metrics help identify where congestion occurs in the system.
                Machines with high server utilization and long waiting times indicate bottlenecks that should be prioritized for improvement.</p>
                
                <h5>Understanding Utilization and Waiting Time:</h5>
                <p>Even with moderate utilization rates (e.g., 67%), long waiting times can occur. This is because as utilization increases, 
                waiting times grow exponentially, not linearly:</p>
                <ul>
                    <li>At 50% utilization, relative waiting time = 1x</li>
                    <li>At 67% utilization, relative waiting time = 2x</li>
                    <li>At 80% utilization, relative waiting time = 4x</li>
                    <li>At 90% utilization, relative waiting time = 9x</li>
                    <li>At 95% utilization, relative waiting time = 19x</li>
                </ul>
                <p>This explains why bottlenecks form even when machines aren't running at 100% capacity.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Machine Performance Analysis</h2>
            <p>This chart shows the utilization and availability of each machine. The bottleneck machine is highlighted in red.</p>
            <img src="machine_chart.png" alt="Machine Performance Chart" class="chart">
            
            <div class="recommendation">
                <h4>💡 Performance Insight:</h4>
                <p>Machine <strong>{{ bottleneck }}</strong> is operating at high utilization ({{ bottleneck_util }}%), 
                indicating it's limiting system throughput. Consider adding capacity to this machine or reducing its processing time.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Wait Time vs. Processing Time</h2>
            <p>This chart shows the average waiting and processing times for each machine. Longer waiting times indicate potential bottlenecks.</p>
            <img src="waiting_time_chart.png" alt="Wait Time vs Processing Time Chart" class="chart">
            
            <div class="recommendation">
                <h4>💡 Queue Management Insight:</h4>
                <p>Products spend significant time waiting for {{ bottleneck }}. 
                This confirms that this machine is a bottleneck and should be prioritized for improvement.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Machine Reliability Analysis</h2>
            <p>This chart shows the uptime and downtime for each machine during the simulation.</p>
            <img src="machine_status.png" alt="Machine Reliability Chart" class="chart">
            
            <div class="recommendation">
                <h4>💡 Reliability Insight:</h4>
                <p>Machine <strong>{{ failure_prone }}</strong> had the most failures. 
                Improving maintenance or upgrading this machine could increase overall system throughput by reducing downtime.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Sample Comparison</h2>
            <p>This chart compares key metrics between all simulated products and the products displayed in this report.</p>
            <img src="comparison_chart.png" alt="Comparison Chart" class="chart">
            
            <div class="recommendation">
                <h4>💡 Sample Insight:</h4>
                <p>This chart helps verify if the first products shown in this report are representative of the entire simulation.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Product Flow Timeline</h2>
            <p>This chart shows how products moved through the production line.</p>
            <img src="timeline_chart.png" alt="Product Timeline Chart" class="chart">
        </div>
        
        <div class="section">
            <h2>Product Metrics Analysis</h2>
            <p>This chart provides a detailed breakdown of performance metrics for each product.</p>
            <img src="product_metrics_chart.png" alt="Product Metrics Chart" class="chart">
        </div>
        
        <div class="section">
            <h2>Machine Performance Metrics</h2>
            <div class="table-container">
                {{ machine_table|safe }}
            </div>
        </div>
        
        <div class="section">
            <h2>System Performance Metrics</h2>
            <div class="table-container">
                {{ system_table|safe }}
            </div>
        </div>
        
        <div class="section">
            <h2>Product Metrics</h2>
            <!-- Add the toggle button here -->
            <button id="toggleButton" class="view-all-btn" onclick="toggleProductTable()">View All Products ({{ total_products }})</button>
            
            <!-- Display products table (shown by default) -->
            <div id="displayProductTable">
                <p>Showing the first {{ display_products }} products out of {{ total_products }} total simulated products.</p>
                <div class="table-container">
                    {{ product_table|safe }}
                </div>
            </div>
            
            <!-- Full products table (hidden by default) -->
            <div id="fullProductTable">
                <p>Showing all {{ total_products }} simulated products.</p>
                <div class="table-container">
                    {{ all_products_table|safe }}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Simulation Event Log (First 20 Events)</h2>
            <div class="table-container">
                {{ event_table|safe }}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Get additional data for template
    bottleneck_util = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == results['bottleneck'], 'Utilization (%)'].values[0]
    failure_count = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == results['failure_prone'], 'Number of Failures'].values[0]
    server_util = results['queue_metrics']['Server Utilization'] * 100
    
    # Prepare template data
    template_data = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'bottleneck': results['bottleneck'],
        'bottleneck_util': f"{bottleneck_util:.1f}",
        'failure_prone': results['failure_prone'],
        'failure_count': failure_count,
        'throughput': f"{results['system_metrics']['Throughput (products/time unit)']:.4f}",
        'avg_lead_time': f"{results['system_metrics']['Average Lead Time']:.1f}",
        'efficiency': f"{results['system_metrics']['System Efficiency (%)']:.1f}",
        'server_util': f"{server_util:.1f}",
        'system_table': system_table,
        'machine_table': machine_table,
        'product_table': product_table,
        'all_products_table': all_products_table,  # Add all products table
        'event_table': event_table,
        'queueing_table': queueing_table,
        'total_products': len(results['product_metrics']),
        'display_products': len(results['display_product_metrics'])
    }
    
    # Render template
    template = jinja2.Template(template_str)
    html_content = template.render(**template_data)
    
    # Write HTML file
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return report_filename

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
    
    # Run simulation with specified number of products
    print("Starting simulation - will run until all products are processed...")
    machines, sim_time = run_simulation(sim_time=max_sim_time, total_products=total_products)
    
    # Analyze results
    print("\nAnalyzing results...")
    results = analyze_results(machines, sim_time)
    
    # Check if any products were processed
    if len(results['product_metrics']) == 0:
        print("⚠️ No products were processed in the simulation! Check the settings and try again.")
        return None
    
    # Select products to display
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
    
    # Extract key metrics
    bottleneck = results['bottleneck']
    failure_prone = results['failure_prone']
    bottleneck_util = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == bottleneck, 'Utilization (%)'].values[0]
    failure_count = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == failure_prone, 'Number of Failures'].values[0]
    
    # Extract queueing metrics
    qm = results['queue_metrics']
    
    # Display key findings in console with improved formatting
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
    print("─────────────────────────────────────────────")
    print("QUEUEING THEORY METRICS:")
    print(f"🔍 Probability of Wait: {qm['Probability of Wait']:.2f}")
    print(f"📏 Average Queue Length: {qm['Average Queue Length']:.2f}")
    print(f"⏸️  Probability of Idle: {1-qm['Server Utilization']:.2f}")
    print(f"⚡ Server Utilization: {qm['Server Utilization']*100:.2f}%")
    print(f"⌛ Average Time Between Arrivals: {qm['Average Interarrival Time']:.2f}")
    print(f"⏲️  Avg Wait Time (Those who wait): {qm['Average Waiting Time (Those Who Wait)']:.2f}")
    print("─────────────────────────────────────────────")
    print(f"🏭 Total Products Simulated: {len(results['product_metrics'])}")
    print(f"🔍 Products Displayed in Tables: {display_products}")
    print("═════════════════════════════════════════════")
    
    # Open the HTML report in the default web browser
    print("\n📂 Opening report in web browser...")
    webbrowser.open('file://' + os.path.abspath(report_file))
    
    return results

def get_user_input():
    while True:
        try:
            total_products = int(input("Enter the number of products to process: "))
            if total_products <= 0:
                print("Please enter a positive number.")
                continue
            return total_products
        except ValueError:
            print("Please enter a valid integer.")
# Run the simulation when script is executed
if __name__ == "__main__":
    # Set parameters for simulation
    # Get the number of products from user input
    TOTAL_PRODUCTS = get_user_input()
    DISPLAY_PRODUCTS = 20  # But only display 20 in reports/visualizations
    SELECTION_METHOD = 'first'  # Show the first 20 products
    
    # Set to None to run until all products are processed
    MAX_SIM_TIME = None
    
    results = run_and_display(TOTAL_PRODUCTS, DISPLAY_PRODUCTS, SELECTION_METHOD, MAX_SIM_TIME)