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


def load_network_from_netcdf(scenario_id, sensibility, value):
    filename = f"results_sensibilities/scenario_{scenario_id}/network_model_scenario_{scenario_id}_{sensibility}_{value}.nc"
    return pypsa.Network(filename)

def build_and_export_lcoh_tables(networks, scenario_ids, sensibility_cases, sensibility_values):
    sensibility_value_labels = {
        0.6: "-40%",
        0.8: "-20%",
        1.0: "Reference",
        1.2: "+20%",
        1.4: "+40%"
    }
    sensibility_case_labels = {
        1: "Wind Investment",
        2: "PV Investment",
        3: "BESS Investment",
        4: "Grid PPA Price",
        5: "PEM Investment"
    }

    tables = {scenario_id: pd.DataFrame(index=[sensibility_value_labels[v] for v in sensibility_values],
                                        columns=[sensibility_case_labels[c] for c in sensibility_cases])
              for scenario_id in scenario_ids}


    for key, n in networks.items():

        parts = key.lstrip('n').split('_')
        scenario_id = int(parts[0])
        sensibility = int(parts[1])
        value = float(parts[2])

        lcoh_value = n.lcoh.round(2)
        row_label = sensibility_value_labels[value]
        column_label = sensibility_case_labels[sensibility]

        tables[scenario_id].loc[row_label, column_label] = lcoh_value

    os.makedirs("results_sensibilities/combined", exist_ok=True)
    for scenario_id, table in tables.items():
        table.to_excel(f"results_sensibilities/combined/scenario_{scenario_id}_table.xlsx")

    return 

def plot_lcoh_tables(scenarios_ids):
    import matplotlib.ticker as ticker

    color_map = {
        'Wind Investment': '#4169E1', # light blue
        'PV Investment': '#ffdd57',  # yellow
        'Grid PPA Price': '#9467bd', # purple
        'BESS Investment': '#e377c2',# pink
        'PEM Investment': '#2ca02c',# green
    }


    for scenario_id in scenario_ids:
        table_path = os.path.join("results_sensibilities/combined", f"scenario_{scenario_id}_table.xlsx")
        df = pd.read_excel(table_path, index_col=0)

        plt.figure(figsize=(6, 3))
       
        for column in df.columns:
            color = color_map.get(column, None)
            plt.plot(df.index, df[column], marker='o', label=column, markersize=4, linewidth=1.0, color=color)

       # plt.title(f"Scenario {scenario_id}")
        plt.xlabel("Investment cost relative change in %")
        plt.ylabel("LCOH (EUR/kg)")
        plt.grid(True)
        plt.legend().set_visible(False)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        plt.tight_layout()
        image_path = f"results_sensibilities/combined/scenario_{scenario_id}_lcoh_plot.png"
        plt.savefig(image_path, dpi=300)
        plt.close()


def plot_all (scenarios_ids):

    fig, axs = plt.subplots(3, 2, figsize=(2 * 4.5, 3 * 2.7))
    axs = axs.flatten()

    for i, scenario_id in enumerate(scenario_ids):
        image_path = f"results_sensibilities/combined/scenario_{scenario_id}_lcoh_plot.png"
        if os.path.exists(image_path):
            img = Image.open(image_path)
            axs[i].imshow(img)
            axs[i].axis('off')
            letter = chr(65 + i)
            axs[i].set_title(f"{letter}.", fontsize=10, loc='left', x=0.15)

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    combined_path = os.path.join("results_sensibilities/combined/sensibilities_combined.png")
    plt.savefig(combined_path, dpi=600, bbox_inches='tight')
    plt.close()
    



if __name__ == "__main__":
    sensibility_values = [0.6, 0.8, 1, 1.2, 1.4]
    sensibility_cases = [1, 2, 3, 4, 5]
    scenario_ids = [1, 2, 3, 4, 5, 6]

    networks = {}
    for scenario_id in scenario_ids:
        for sensibility in sensibility_cases:
            for value in sensibility_values:
                print(f"\n=== Loading Scenario {scenario_id}_{sensibility}_{value} ===")
                n = load_network_from_netcdf(scenario_id, sensibility, value)
                networks[f"n{scenario_id}_{sensibility}_{value}"] = n

  #  build_and_export_lcoh_tables(networks, scenario_ids, sensibility_cases, sensibility_values)
    plot_lcoh_tables(scenario_ids)
    plot_all(scenario_ids)

