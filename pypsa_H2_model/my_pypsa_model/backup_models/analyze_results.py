import pypsa
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')  # Use TkAgg to display in a new window (interactive)
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-white")
import os
import linopy
import linopy.expressions
import pickle
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches



# Font settings (to match LaTeX document)
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 12  # General font size (adjust as needed)
mpl.rcParams['axes.titlesize'] = 12  # Title font size
mpl.rcParams['axes.labelsize'] = 12  # Axis label font size
mpl.rcParams['legend.fontsize'] = 12  # Legend font size
mpl.rcParams['xtick.labelsize'] = 12  # X-tick label size
mpl.rcParams['ytick.labelsize'] = 12  # Y-tick label size

# Style Customization
mpl.rcParams['axes.grid'] = True 
mpl.rcParams['grid.linestyle'] = (0, (7, 5))  # Dash pattern: (start_offset, (dash_length, space_length))
mpl.rcParams['grid.color'] = 'gray'    # Grid line color
mpl.rcParams['grid.alpha'] = 0.4       # Grid transparency
mpl.rcParams['grid.linewidth'] = 0.8   # Line thickness
mpl.rcParams['axes.linewidth'] = 0.6


def load_network_from_netcdf(scenario_id):
    filename = f"results/scenario_{scenario_id}/network_model_scenario_{scenario_id}.nc"
    return pypsa.Network(filename)


def analyze_results(n, scenario_id):
    
    output_dir = f"results/scenario_{scenario_id}"
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_scenario_{scenario_id}"

    print(n.generators.p_nom_opt)
    print(n.links.p_nom_opt)
    print(n.storage_units.p_nom_opt)
    print(f"Objective function value (total costs): {n.objective:.2f} EUR")
    print(f"LCOH: {n.lcoh:.2f} EUR/kg")

    flows0 = n.links_t.p0
    flows1 = n.links_t.p1
    flows2 = n.links_t.p2

    flows0.rename(columns={'Electrolyzer': 'PEM input', 'Compressor': 'Compressor input (MW_H2)'}, inplace=True)
    flows1.rename(columns={'Electrolyzer': 'PEM output', 'Compressor': 'Compressor output (MW_H2)'}, inplace=True)
    flows2.rename(columns={'Electrolyzer': 'Electrolyzer BoP consumption  (MW)', 'Compressor': 'Compressor'}, inplace=True)

    statistics = pd.DataFrame(n.statistics().round(2))
    statistics = statistics.reset_index()
    statistics.columns.values[0] = 'Type'
    statistics.columns.values[1] = 'Component'
    statistics.loc[statistics['Type'] == 'Load', 'Component'] = 'Load'
    statistics.set_index('Component', inplace=True)
    statistics.rename(index={'on_wind': 'Onshore Wind','pv': 'PV','AC': 'Compressor Power','hydrogen': 'PEM', 'hydrogen_storage': 'Hydrogen Storage', 'grid': 'Grid' }, inplace=True)
    
    
    
    curtailment_raw = pd.DataFrame(n.statistics.curtailment(aggregate_time=False).round(2))
    curtailment_d = curtailment_raw.droplevel('component')
    curtailment = curtailment_d.transpose().fillna(0)
    curtailment['Curtailment'] = curtailment.get('pv', 0) + curtailment.get('on_wind', 0)

    results = pd.concat([n.generators_t.p, n.storage_units_t.p, flows0, flows1, flows2, n.loads_t.p], axis=1).round(2)
    desired_cols = ['on_wind', 'pv', 'grid', 'PEM input', 'battery_storage', 'Compressor', 'hydrogen_storage']
    available_cols = [col for col in desired_cols if col in results.columns]
    results_helper = results[available_cols].copy()
    dispatch = pd.concat([results_helper, curtailment['Curtailment']], axis=1, join='inner')
    dispatch['PEM input'] *= -1
    dispatch['Compressor'] *= -1
    dispatch['Curtailment'] *= -1

    load_duration = dispatch.copy()
    load_duration['PEM input'] *= -1
    load_duration['Compressor'] *= -1
    load_duration['Curtailment'] *= -1
    if 'battery_storage' in load_duration: 
        load_duration.drop('battery_storage', axis=1, inplace=True)
    if 'hydrogen_storage' in load_duration:
        load_duration.drop('hydrogen_storage', axis=1, inplace=True)
    ldc = load_duration.apply(lambda col: col.sort_values(ascending=False).reset_index(drop=True))

    energy_balance = pd.DataFrame(n.statistics.energy_balance(aggregate_time=False).round(2))
    storage_units = n.storage_units_t.p

    
    statistics.to_csv(os.path.join(output_dir, f"Statistics{suffix}.csv"), sep=";", decimal=".")
    curtailment.to_csv(os.path.join(output_dir, f"Curtailment{suffix}.csv"), sep=";", decimal=".")
    energy_balance.to_csv(os.path.join(output_dir, f"Energy Balance{suffix}.csv"), sep=";", decimal=".")
    storage_units.to_csv(os.path.join(output_dir, f"Stores{suffix}.csv"), sep=";", decimal=".")
    results.to_csv(os.path.join(output_dir, f"Results{suffix}.csv"), sep=";", decimal=".")
    dispatch.to_csv(os.path.join(output_dir, f"Dispatch{suffix}.csv"), sep=";", decimal=".")

    # Define consistent colors
    color_map = {
        'on_wind': '#4169E1',        # light blue
        'pv': '#ffdd57',             # yellow
        'grid': '#9467bd',           # purple
        'PEM input': '#2ca02c',      # green
        'PEM output': '#2ca02c',      # green
        'battery_storage': '#e377c2',# pink
        'hydrogen_storage': '#00B4CE',# turquisa
        'Compressor': '#FFA500',     # orange
        'Curtailment': '#8c564b'     # brown
    }
    plot_colors = [color_map.get(col, None) for col in dispatch.columns if col in color_map]

    dispatch_subset = dispatch.loc[1000:1175]
    dispatch_hourly_graph = dispatch_subset.plot.area(stacked=False, figsize=(6, 3), ylabel="Power in MW", xlabel="Hour", linewidth=0.4, color=plot_colors)
    plt.tight_layout()
    plt.legend().set_visible(False)
    plt.savefig(os.path.join(output_dir, f'dispatch_hourly_graph{suffix}.png'), format='png', dpi=300, bbox_inches='tight')

    dispatch_daily_grouped = dispatch.copy()
    dispatch_daily_grouped['day'] = (dispatch_daily_grouped.index // 24).astype(int)
    dispatch_daily = dispatch_daily_grouped.groupby('day').sum()/1000
    dispatch_daily_graph = dispatch_daily.plot.area(stacked=False, figsize=(6, 3), ylabel="Generation in GWh", xlabel="Day", linewidth=0.4, color=plot_colors)
    dispatch_daily_graph.set_xticks(range(0, len(dispatch_daily), 50))
    dispatch_daily_graph.set_xticklabels(range(0, len(dispatch_daily), 50))
    plt.tight_layout()
    plt.legend().set_visible(True)
    plt.savefig(os.path.join(output_dir, f'dispatch_daily_area_graph{suffix}.png'), format='png', dpi=300, bbox_inches='tight')

    month_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    #month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dispatch['month'] = (dispatch.index // 730).astype(int)
    dispatch_monthly = dispatch.groupby('month').sum()/1000
    dispatch_monthly_graph = dispatch_monthly.plot(kind='bar', stacked=True, figsize=(6, 3), ylabel="Generation in GWh", xlabel="Month", linewidth=0.4, width=0.4, color=plot_colors)
    dispatch_monthly_graph.set_xticklabels(month_names)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend().set_visible(False)
    plt.savefig(os.path.join(output_dir, f'dispatch_monthly_bar_graph{suffix}.png'), format='png',  dpi=300, bbox_inches='tight')

    ldc_colors = [color_map.get(col, None) for col in ldc.columns]
    ldc_graph = ldc.plot(kind='line', figsize=(6, 3), ylabel="Power in MW", xlabel="Time in h", color=ldc_colors)
    ldc_graph.set_xticks(range(0, len(ldc), 1000))
    ldc_graph.set_xticklabels(range(0, len(ldc), 1000))
    plt.tight_layout()
    plt.legend().set_visible(False)
    plt.savefig(os.path.join(output_dir, f'ldc_graph{suffix}.png'), format='png',  dpi=300, bbox_inches='tight')

    if 'battery_storage' in storage_units:
        storage_bess = storage_units.drop('hydrogen_storage', axis=1) / 1000
        storage_bess_split = pd.DataFrame(index=storage_bess.index)
        storage_bess_split['power_retrieved'] = storage_bess[storage_bess.columns[0]].clip(lower=0)
        storage_bess_split['power_stored'] = storage_bess[storage_bess.columns[0]].clip(upper=0)
        storage_bess_split['month'] = (storage_bess_split.index // 730).astype(int)
        storage_bess_monthly = storage_bess_split.groupby('month').sum()
        bess_monthly_graph = storage_bess_monthly.plot(kind='bar', stacked=True, figsize=(6, 3), ylabel="Electricity in GWh", linewidth=0.4, xlabel="Month", width=0.5)
        bess_monthly_graph.set_xticklabels(month_names)
        y_min, y_max = bess_monthly_graph.get_ylim()
        max_limit = max(abs(y_min), abs(y_max))
        bess_monthly_graph.set_ylim(-max_limit, max_limit)
        bess_monthly_graph.set_yticks(np.linspace(-max_limit, max_limit, num=5))
        formatter = FuncFormatter(lambda x, _: f'{x:.1f}')
        bess_monthly_graph.yaxis.set_major_formatter(formatter)
        plt.tight_layout()
        plt.legend().set_visible(False)
        plt.savefig(os.path.join(output_dir, f'bess_monthly_bar_graph{suffix}.png'), format='png',  dpi=300, bbox_inches='tight')

    storage_hydrogen = storage_units.copy()
    if 'battery_storage' in storage_hydrogen:
        storage_hydrogen = storage_hydrogen.drop('battery_storage', axis=1)
    storage_hydrogen /= ((33.33 * 1000) / 1000) #scale from MWh to tons H
    storage_hydrogen_split = pd.DataFrame(index=storage_hydrogen.index)
    storage_hydrogen_split['retrieved'] = storage_hydrogen[storage_hydrogen.columns[0]].clip(lower=0)
    storage_hydrogen_split['stored'] = storage_hydrogen[storage_hydrogen.columns[0]].clip(upper=0)
    storage_hydrogen_split['month'] = (storage_hydrogen_split.index // 730).astype(int)
    storage_hydrogen_monthly = storage_hydrogen_split.groupby('month').sum()
    hydrogen_monthly_graph = storage_hydrogen_monthly.plot(kind='bar', stacked=True, figsize=(6, 3), ylabel="H_2 in tons", linewidth=0.4, xlabel="Month", width=0.5)
    hydrogen_monthly_graph.set_xticklabels(month_names)
    y_min, y_max = hydrogen_monthly_graph.get_ylim()
    max_limit = max(abs(y_min), abs(y_max))
    hydrogen_monthly_graph.set_ylim(-max_limit, max_limit) 
    hydrogen_monthly_graph.set_yticks(np.linspace(-max_limit, max_limit, num=5))
    formatter2 = FuncFormatter(lambda x, _: f'{x:.1f}')
    hydrogen_monthly_graph.yaxis.set_major_formatter(formatter2)
    plt.legend().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hydrogen_storage_monthly_bar_graph{suffix}.png'), format='png',  dpi=300, bbox_inches='tight')
        
    
    return 



def combine_optimal_capacity_statistics(scenario_ids):
    combined_table = pd.DataFrame()

    for i, scenario_id in enumerate(scenario_ids):
        output_dir = f"results/scenario_{scenario_id}"
        suffix = f"_scenario_{scenario_id}"
        file_path = os.path.join(output_dir, f"Statistics{suffix}.csv")
        statistics_df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0)
        optimal_capacity_series = statistics_df['Optimal Capacity']
        combined_table = combined_table.join(optimal_capacity_series.rename(f"Scenario {scenario_id}"), how='outer')

    desired_order = [
    'Onshore Wind',
    'PV',
    'PEM',
    'Compressor Power',
    'battery storage',
    'Hydrogen Storage',
    'Grid'
    ]

    existing_order = [comp for comp in desired_order if comp in combined_table.index]
    combined_table = combined_table.reindex(existing_order)

    conversion_factor = 33.33  # MWh to ton H2 
    hydrogen_row = pd.to_numeric(combined_table.loc['Hydrogen Storage'], errors='coerce')
    combined_table.loc['Hydrogen Storage'] = (hydrogen_row * 7/ conversion_factor).round(2)

    
    os.makedirs("results/combined", exist_ok=True)
    combined_path = os.path.join("results/combined", "Combined_Statistics.csv")
    combined_table.to_csv(combined_path, sep=";", decimal=",")
    
    return combined_path

def plot_combined_graphs_png(scenario_ids, graph_name, output_filename, rows=3, cols=2, dpi=600, color_map=None):
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2.7))
    axs = axs.flatten()

    for i, scenario_id in enumerate(scenario_ids):
        image_path = f"results/scenario_{scenario_id}/{graph_name}_scenario_{scenario_id}.png"
        if os.path.exists(image_path):
            img = Image.open(image_path)
            axs[i].imshow(img)
            axs[i].axis('off')
            letter = chr(65 + i)
            axs[i].set_title(f"{letter}.", fontsize=10, loc='left', x=0.15)
        else:
            axs[i].text(0.5, 0.5, f"No image for Scenario {scenario_id}", ha='center', va='center')
            axs[i].axis('off')

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')


    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    combined_path = os.path.join("results/combined", f"{output_filename}.png")
    plt.savefig(combined_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_combined_bess_hydrogen(scenario_ids, graph_bess_name, graph_hydrogen_name, output_filename, dpi=600):
    rows = 6
    cols = 2  
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2.7))
    axs = axs.reshape(rows, cols)  # Ensure 2D shape: (row, col)

    for row_idx, scenario_id in enumerate(scenario_ids):
        bess_path = f"results/scenario_{scenario_id}/{graph_bess_name}_scenario_{scenario_id}.png"
        hydrogen_path = f"results/scenario_{scenario_id}/{graph_hydrogen_name}_scenario_{scenario_id}.png"

         # Load and plot BESS image
        img_bess = Image.open(bess_path)
        axs[row_idx, 0].imshow(img_bess)
        axs[row_idx, 0].axis('off')
        axs[row_idx, 0].set_title(f"Scenario {scenario_id} - BESS", fontsize=10, loc='left', x=0.13)

        # Load and plot Hydrogen image
        img_hydrogen = Image.open(hydrogen_path)
        axs[row_idx, 1].imshow(img_hydrogen)
        axs[row_idx, 1].axis('off')
        axs[row_idx, 1].set_title(f"Scenario {scenario_id} - Pressurized H2 Tank", fontsize=10, loc='left', x=0.15)


    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    os.makedirs("results/combined", exist_ok=True)
    combined_path = os.path.join("results/combined", f"{output_filename}.png")
    plt.savefig(combined_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def lcoh_results(networks, scenario_ids):
    kpi_table = pd.DataFrame()    
    component_cost_table = pd.DataFrame()  

    for key, n in networks.items():
        scenario_id = key.replace("n", "")  


        lcoh_value = n.lcoh.round(1)
        objective = (n.objective / 1e6).round(1)
        capex_total = (n.statistics.capex().sum() / 1e6).round(1)
        opex_total = (n.statistics.opex().sum() / 1e6).round(1)
        load = (n.load_kg / 1000).round(1)

     
        kpi_table[f"Scenario {scenario_id}"] = pd.Series({
            "Capex (Mio EUR)": capex_total,
            "Opex (Mio EUR)": opex_total,
            "Annualized Total Costs (Mio EUR)": objective,
            "Load (t_h2)": load,
            "LCOH [EUR/kg]": lcoh_value
        })




    for scenario_id in scenario_ids:
        output_dir = f"results/scenario_{scenario_id}"
        suffix = f"_scenario_{scenario_id}"
        file_path = os.path.join(output_dir, f"Statistics{suffix}.csv")

        statistics_df = pd.read_csv(file_path, sep=";", decimal=".", index_col=0)

     
        total_costs = (statistics_df['Capital Expenditure'] + statistics_df['Operational Expenditure'])/1000000

    
        component_cost_table = component_cost_table.join(
        total_costs.rename(f"Scenario {scenario_id}"),
        how='outer'
    )

   
    full_table = pd.concat([component_cost_table, kpi_table])

    
    os.makedirs("results/combined", exist_ok=True)
    full_table.round(1).to_csv("results/combined/full_cost_summary.csv", sep=";", decimal=".")

    return full_table

if __name__ == "__main__":
    scenario_ids = [1, 2, 3, 4, 5, 6]
    networks = {}

    for scenario_id in scenario_ids:
        print(f"\n=== Analyzing Scenario {scenario_id} ===")
        n = load_network_from_netcdf(scenario_id)
        analyze_results(n, scenario_id)
        networks[f"n{scenario_id}"] = n

    full_table = lcoh_results(networks, scenario_ids)
    print(full_table)

    combine_optimal_capacity_statistics(scenario_ids)
    
    plot_combined_graphs_png(scenario_ids, graph_name="dispatch_hourly_graph", output_filename="combined_dispatch_hourly")
    plot_combined_graphs_png(scenario_ids, graph_name="dispatch_daily_area_graph", output_filename="combined_dispatch_daily")
    plot_combined_graphs_png(scenario_ids, graph_name="dispatch_monthly_bar_graph", output_filename="combined_dispatch_monthly")
    plot_combined_graphs_png(scenario_ids, graph_name="ldc_graph", output_filename="combined_ldc")
    plot_combined_graphs_png(scenario_ids=[1, 2, 3, 5], graph_name="bess_monthly_bar_graph", output_filename="combined_bess_monthly")
    plot_combined_graphs_png(scenario_ids, graph_name="hydrogen_storage_monthly_bar_graph", output_filename="combined_hydrogen_storage")
  #  plot_combined_bess_hydrogen(scenario_ids=[1, 2, 3, 4, 5, 6], graph_bess_name="bess_monthly_bar_graph", graph_hydrogen_name="hydrogen_storage_monthly_bar_graph" ,output_filename="combined_hydrogen_bess_storage")

