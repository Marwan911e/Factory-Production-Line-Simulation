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

def product_arrival(env, machines, total_products=100):
    """Generate product arrivals"""
    i = 0
    while True:
        # Wait for interarrival time
        yield env.timeout(interarrival_time())

        # Create new product
        Product(env, f"Product-{i}", machines)
        i += 1

        # Stop after generating enough products
        # Modified to generate specified number of products
        if i >= total_products:
            break

def run_simulation(sim_time=500, total_products=100):
    """Run the simulation"""
    global event_log, product_metrics, machine_states

    # Reset all trackers
    event_log = []
    product_metrics = []
    machine_states = []

    # Initialize simulation environment
    env = simpy.Environment()
    machines = [Machine(env, "M1"), Machine(env, "M2"), Machine(env, "M3")]

    # Start the product arrival process with specified total products
    env.process(product_arrival(env, machines, total_products))

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
        'bottleneck': bottleneck,
        'failure_prone': failure_prone
    }

def select_display_products(results, num_to_display=20, selection_method='random'):
    """Select a subset of products to display in reports"""
    all_products_df = results['product_metrics']
    
    if selection_method == 'first':
        # Select first n products
        display_products = all_products_df.head(num_to_display)
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

# New simplified visualization functions
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
    ax.set_title(f'Product Timeline: Selected {len(product_df)} Products from {len(results["product_metrics"])} Total')  # Updated title
    
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
    """Create a visualization of selected products metrics"""
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
    ax1.set_title(f'Lead Time Breakdown for {len(product_df)} Selected Products')
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
    ax2.set_title(f'Processing Time by Machine for {len(product_df)} Selected Products')
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
    ax.bar(x + width/2, display_means, width, label=f'Selected Products (n={len(display_df)})', color='lightgreen')
    
    # Add annotations
    for i, (all_val, disp_val) in enumerate(zip(all_means, display_means)):
        ax.text(i - width/2, all_val + 0.5, f"{all_val:.1f}", ha='center', va='bottom')
        ax.text(i + width/2, disp_val + 0.5, f"{disp_val:.1f}", ha='center', va='bottom')
    
    # Customize chart
    ax.set_ylabel('Time')
    ax.set_title('Comparison: All Products vs. Selected Products')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
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
        'comparison_chart': create_comparison_chart(results)  # New comparison chart
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
        f.write(f"Products Selected for Display: {len(results['display_product_metrics'])}\n\n")
        
        # System metrics
        f.write("SYSTEM METRICS:\n")
        for key, value in results['system_metrics'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Bottleneck information
        bottleneck = results['bottleneck']
        bottleneck_util = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == bottleneck, 'Utilization (%)'].values[0]
        f.write(f"BOTTLENECK: {bottleneck} (Utilization: {bottleneck_util:.2f}%)\n")
        
        # Failure information
        failure_prone = results['failure_prone']
        failure_count = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == failure_prone, 'Number of Failures'].values[0]
        f.write(f"MOST FAILURES: {failure_prone} (Count: {failure_count})\n")
        
        # Product metrics table (selected products only)
        f.write("\nSELECTED PRODUCT METRICS:\n")
        product_table = tabulate(
            results['display_product_metrics'].sort_values('Arrival').round(2), 
            headers='keys', 
            tablefmt='grid', 
            showindex=False
        )
        f.write(product_table)
    
    return report_dir

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
    comparison_chart_path = os.path.join(report_dir, "comparison_chart.png")  # New chart
    
    # Create and save each chart
    create_simple_machine_chart(results).savefig(machine_chart_path, dpi=100, bbox_inches='tight')
    create_simple_timeline_chart(results).savefig(timeline_chart_path, dpi=100, bbox_inches='tight')
    create_simple_machine_status_bar(results).savefig(machine_status_path, dpi=100, bbox_inches='tight')
    create_simple_waiting_time_chart(results).savefig(waiting_time_chart_path, dpi=100, bbox_inches='tight')
    create_product_metrics_table_chart(results).savefig(product_metrics_chart_path, dpi=100, bbox_inches='tight')
    create_comparison_chart(results).savefig(comparison_chart_path, dpi=100, bbox_inches='tight')  # New chart
    
    # Close all plots to free memory
    plt.close('all')
    
    # Convert the DataFrames to HTML tables - use display products only for product table
    product_table = results['display_product_metrics'].round(2).to_html(classes='table table-striped table-hover', index=False)
    machine_table = results['machine_metrics'].round(2).to_html(classes='table table-striped table-hover', index=False)
    system_metrics = pd.DataFrame([results['system_metrics']]).round(2)
    system_table = system_metrics.to_html(classes='table table-striped table-hover', index=False)
    
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
    </head>
    <body>
        <div class="header">
            <h1>Factory Production Line Simulation Report</h1>
            <p class="text-muted">Generated on {{ timestamp }}</p>
            <div class="info-banner">
                <p>Simulation run with {{ total_products }} products; displaying {{ display_products }} selected products.</p>
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
                <div class="col-md-4">
                    <div class="key-metric">
                        <h3>{{ throughput }}</h3>
                        <p>Products per Time Unit</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="key-metric">
                        <h3>{{ avg_lead_time }}s</h3>
                        <p>Average Lead Time</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="key-metric">
                        <h3>{{ efficiency }}%</h3>
                        <p>System Efficiency</p>
                    </div>
                </div>
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
            <p>This chart compares key metrics between all simulated products and the selected sample displayed in this report.</p>
            <img src="comparison_chart.png" alt="Comparison Chart" class="chart">
            
            <div class="recommendation">
                <h4>💡 Sample Insight:</h4>
                <p>This chart helps verify if the selected products shown in this report are representative of the entire simulation.</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Product Flow Timeline</h2>
            <p>This chart shows how the selected {{ display_products }} products moved through the production line.</p>
            <img src="timeline_chart.png" alt="Product Timeline Chart" class="chart">
        </div>
        
        <div class="section">
            <h2>Product Metrics Analysis</h2>
            <p>This chart provides a detailed breakdown of the selected products' performance metrics.</p>
            <img src="product_metrics_chart.png" alt="Product Metrics Chart" class="chart">
        </div>
        
        <div class="section">
            <h2>System Performance Metrics</h2>
            <div class="table-container">
                {{ system_table|safe }}
            </div>
        </div>
        
        <div class="section">
            <h2>Machine Performance Metrics</h2>
            <div class="table-container">
                {{ machine_table|safe }}
            </div>
        </div>
        
        <div class="section">
            <h2>Selected Product Metrics</h2>
            <p>Showing {{ display_products }} products out of {{ total_products }} total simulated products.</p>
            <div class="table-container">
                {{ product_table|safe }}
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
        'system_table': system_table,
        'machine_table': machine_table,
        'product_table': product_table,
        'event_table': event_table,
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

def run_and_display(total_products=100, display_products=20, selection_method='random'):
    """Run simulation and generate report"""
    print("\n=== FACTORY PRODUCTION LINE SIMULATION ===\n")
    print(f"Running simulation with {total_products} products...")
    
    # Run simulation with specified number of products
    machines, sim_time = run_simulation(sim_time=500, total_products=total_products)
    
    # Analyze results
    print("Analyzing results...")
    results = analyze_results(machines, sim_time)
    
    # Select products to display
    print(f"Selecting {display_products} products for display...")
    display_results = select_display_products(results, display_products, selection_method)
    
    # Generate HTML report
    print("Generating HTML report...")
    report_file = create_html_report(display_results)
    
    # Print summary to console
    print("\n✅ SIMULATION COMPLETE!\n")
    print(f"📊 Report generated: {report_file}")
    
    # Extract key metrics
    bottleneck = results['bottleneck']
    failure_prone = results['failure_prone']
    bottleneck_util = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == bottleneck, 'Utilization (%)'].values[0]
    failure_count = results['machine_metrics'].loc[results['machine_metrics']['Machine'] == failure_prone, 'Number of Failures'].values[0]
    
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
    print(f"🏭 Total Products Simulated: {len(results['product_metrics'])}")
    print(f"🔍 Products Selected for Display: {display_products}")
    print("═════════════════════════════════════════════")
    
    # Open the HTML report in the default web browser
    print("\n📂 Opening report in web browser...")
    webbrowser.open('file://' + os.path.abspath(report_file))
    
    return results

# Run the simulation when script is executed
if __name__ == "__main__":
    # Set parameters for simulation
    TOTAL_PRODUCTS = 100  # Simulate 100 products
    DISPLAY_PRODUCTS = 20  # But only display 20 in reports/visualizations
    SELECTION_METHOD = 'random'  # 'random', 'first', 'last', or 'representative'
    
    results = run_and_display(TOTAL_PRODUCTS, DISPLAY_PRODUCTS, SELECTION_METHOD)