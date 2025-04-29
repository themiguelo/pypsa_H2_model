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

####################-----------1.0 Import Data -----------------------######################
def data_frame ():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    
    ###--- 1.1 Loading time series data ---###
    time_series_cf_path = os.path.join(base_dir, "Data", "time_series_cf.xlsx")
    time_series_cf = pd.read_excel(time_series_cf_path)
    print(time_series_cf)

    ###--- 1.2 Build Technology Parameters Data Frame ---###
    technology_parameters_path = os.path.join(base_dir, "Data", "technology_parameters.xlsx")
    technology_parameters = pd.read_excel(technology_parameters_path, index_col=[0, 1])
    print(technology_parameters)

    defaults = {
        "FOM": 0,
        "VOM": 0,
        "efficiency": 1,
        "capex": 0,
        "lifetime": 25,
        "CO2 intensity": 0,
        "discount_rate": 0.07,
    }
    print(technology_parameters.columns)

    define_year="value_2030"
    technology_parameters = technology_parameters[define_year].unstack().fillna(defaults)

  
    technology_parameters.to_excel(os.path.join("pypsa_H2_model/my_pypsa_model/Data", "technology_parameters_data_frame.xlsx"))

    


    def annuity(r, n):
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    annuity(0.07, 25)

    # Based on this, we can calculate the annualised capital costs (in PyPSA terms) €/MW/a, 
    # and we add them to our data frame of technical parameters

    annuity = technology_parameters.apply(lambda x: annuity(x["discount_rate"], x["lifetime"]), axis=1)
    technology_parameters["capital_cost"] = (annuity * technology_parameters["capex"]) + technology_parameters["FOM"]
    

    technology_parameters.to_excel(os.path.join("pypsa_H2_model/my_pypsa_model/Data", "technology_parameters_annualized.xlsx"))


    #######-------------1.3 Demand Series-----------#######
    demand_operation_planner_path = os.path.join(base_dir, "Data", "demand_operation_planner.xlsx")
    #Get demand schedule:
    demand_series = pd.read_excel(demand_operation_planner_path, index_col=0)
    load=demand_series["demand_kg"]*(33.33/1000) #load in MWh
    total_load_kg=demand_series["demand_kg"].sum()
    #Get operation schedule:
    operation_series_max = pd.read_excel(demand_operation_planner_path, index_col=0)["operation_series_max"]
    operation_series_min = pd.read_excel(demand_operation_planner_path, index_col=0)["operation_series_min"]

    return time_series_cf, technology_parameters, load, total_load_kg, operation_series_max, operation_series_min

####################------2.0 Model Setup-----------######################

def compressor_ratio():

    #Calculate the ratio needed to get power of compressor shaft attached to electricity bus

    #Parameters:
    N = 1 #Number of stages
    k = 1.4 #Isentropic index for hydrogen
    R = 4124 #Specific hydrogen constant (J/kg·K)
    Z = 1.10814 #Compressibility Factor, depends on pressure and temperature, see respective table
    T1 = 298.25  #Inlet temperature of hydrogen (K)
    P1 = 1 #Initial pressure (Pa or bar)
    P2 = 200 #Final pressure (Pa or bar)
    efficiency_motor= 0.95 #Efficiency of the shaft motor of the compressor (between 0 and 1)
    efficiency_isentropic = 0.7
    pressure_ratio = P2/P1
    x = ((N*k * R * T1* Z) / (k - 1)) * ((pressure_ratio ** ((k - 1) /(k*N))) - 1) / (efficiency_motor*efficiency_isentropic)
    return x
   

def setup_model(time_series_cf, technology_parameters, load, operation_series_max, operation_series_min, x):
    
    n = pypsa.Network()
    resolution = 1 #hourly resolution of model
    n.add("Bus", "electricity")
    n.add("Bus", "hydrogen")
    n.add("Bus", "compressed_hydrogen")

    #…and tell the pypsa.Network object n what the snapshots of the model will be using the utility function n.set_snapshots().
    n.set_snapshots(time_series_cf.index)#use the index (first column) of the time series ts
    n.snapshots
    n.snapshot_weightings.loc[:, :] = resolution #hourly resolution of model, the weighting of the snapshots (e.g. how many hours they represent)

    carriers = [
    "on_wind",
    "off_wind",
    "RE",
    "grid",
    "pv",
    "hydrogen_storage",
    "battery_storage"
    "hydrogen"
    ]

    n.add(
    "Carrier",
    carriers
    )


    if scenario_id in [2,3,5]:
     n.add(
        "Generator",
        "pv",
        bus="electricity",
        carrier="pv",
        p_max_pu=time_series_cf["pv"],
        capital_cost=technology_parameters.at["pv", "capital_cost"],
        marginal_cost=technology_parameters.at["pv", "VOM"],
        efficiency=technology_parameters.at["pv", "efficiency"],
        p_nom_extendable=True,
        )

    if scenario_id in [1,3,4]:
     n.add(
        "Generator",
        "on_wind",
        bus="electricity",
        carrier="on_wind",
        p_max_pu=time_series_cf["on_wind"],
        capital_cost=technology_parameters.at["on_wind", "capital_cost"],
        marginal_cost=technology_parameters.at["on_wind", "VOM"],
        efficiency=technology_parameters.at["on_wind", "efficiency"],
        p_nom_extendable=True,
        )

    #make sure the capacity factors are read-in correctly.
    #Since the power flow and dispatch are generally time-varying quantities, these are stored in a different location than e.g. n.generators. They are stored in n.generators_t.

    if scenario_id in [4,5,6]:
     n.add("Generator",
          "grid",
          bus="electricity",
          carrier="grid",
          capital_cost=technology_parameters.at["grid", "capital_cost"],
          marginal_cost=technology_parameters.at["grid", "VOM"],
          efficiency=1,
          p_nom_extendable=True,
        )

    if scenario_id in [1,2,3,5]:
     n.add("StorageUnit",
        "battery_storage",
        bus="electricity",
        carrier="battery storage",
        max_hours=4, #Maximum state of charge capacity in terms of hours at full output capacity p_nom, e.g  fully charged, the battery can discharge at full capacity for 6 hours
        capital_cost=technology_parameters.at["battery", "capital_cost"],
        marginal_cost=technology_parameters.at["battery", "VOM"],
        efficiency_store=technology_parameters.at["battery", "efficiency"],
        efficiency_dispatch=technology_parameters.at["battery", "efficiency"],
        p_nom_extendable=True,
        cyclic_state_of_charge=True, 
        )

    n.add("Link",
            "Electrolyzer",
            bus0="electricity",
            bus1="hydrogen",
            #bus2="electricity",
            carrier="hydrogen",
            capital_cost=technology_parameters.at["electrolyser_PEM", "capital_cost"],
            marginal_cost=technology_parameters.at["electrolyser_PEM", "VOM"],
            efficiency=technology_parameters.at["electrolyser_PEM", "efficiency"],
          #  efficiency2=-0.1, 
            p_min_pu=operation_series_min,  # 20% minimum load when working
            p_max_pu=operation_series_max, #0% when off, 90% maximum load when working
            p_nom_extendable=True,  # Allow the model to optimize capacity
    )
     
    n.add("Link",
           "Compressor",
            bus0="hydrogen", #input
            bus1="compressed_hydrogen", #output
            bus2="electricity", #input
            capital_cost=technology_parameters.at["compressor", "capital_cost"],
            efficiency=1, #conservation of hydrogen flow
            efficiency2=-(x/((39.39/1000)*3600))/1000000, #The power arriving at bus_n is just efficiency_n*p0, if negative power being withdrawn
            p_nom_extendable=True,  # Allow the model to optimize capacity
    )
  
    n.add(
        "StorageUnit",
        "hydrogen_storage",
        bus="compressed_hydrogen",
        carrier="hydrogen_storage",
        max_hours=7, 
        capital_cost=technology_parameters.at["hydrogen_storage", "capital_cost"],
        efficiency_store=technology_parameters.at["hydrogen_storage", "efficiency"],
        p_nom_extendable=True,
    )

    n.add(
        "Load",
        "demand",
        bus="compressed_hydrogen",
        carrier="hydrogen",
        p_set=load,
    )
    
    return n

#####################---------------3.0 Optimize Model ----------------##################
def optimize(n):
    n.optimize(solver_name="highs")
   
    return n

####################--------------- 4.0 Save & Load Model---------------###################
def save_network_to_netcdf(n, total_load_kg, scenario_id):
    #saves the PyPSA network to a NetCDF file. 
    output_dir = f"results/scenario_{scenario_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Total demand: {total_load_kg:.2f} kg")
    print(f"Objective function value (total costs) scenario {scenario_id}: {n.objective:.2f} EUR")
    n.load_kg=total_load_kg
    n.lcoh=n.objective/total_load_kg
    print(f"LCOH scenario {scenario_id}: {n.lcoh:.2f} EUR/kg")

    filename = os.path.join(output_dir, f"network_model_scenario_{scenario_id}.nc")
    n.export_to_netcdf(filename)

if __name__ == "__main__":
    scenario_ids = [1, 2, 3, 4, 5, 6]

    for scenario_id in scenario_ids:
        print(f"\n================= Running Scenario {scenario_id} =================")
        time_series_cf, tech_params, load, total_load_kg, op_max, op_min = data_frame()
        x = compressor_ratio()
        n = setup_model(time_series_cf, tech_params, load, op_max, op_min, x)
        optimize(n)
        save_network_to_netcdf(n, total_load_kg, scenario_id)
