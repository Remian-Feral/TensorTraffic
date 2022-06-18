import os, sys, optparse, xlsxwriter

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import traci  # TraCI functionality
from sumolib import checkBinary  # checkBinary help locate the Sumo binaries
import time

WrkBk = xlsxwriter.Workbook(f"TrafficEpisode.xlsx")
WrkSht = WrkBk.add_worksheet(f'Traffic Data')
WrkSht.write('A1', 'At Timestep')
WrkSht.write('B1', 'North Density Lane')
WrkSht.write('C1', 'South Density Lane')
WrkSht.write('D1', 'East Density Lane')
WrkSht.write('E1', 'West Density Lane')
WrkSht.write('F1', 'Total Congestion')
WrkSht.write('G1', 'Halting Time')
WrkSht.write('H1', 'Phase')
WrkSht.write('I1', 'Phase Duration')
WrkSht.write('J1', 'Reward')
WrkSht.write('K1', 'Total Loaded Cars')
WrkSht.write('L1', 'Total Arrived Cars')
WrkSht.set_column(0, 12, width=20)

TimeStep = []
NorthList = []
SouthList = []
EastList = []
WestList = []
Congestion = []
HaltingList = []
Phase = []
PhaseDur = []
RewardList = []
Total_LoadedCars = []
Total_ArrivedCars = []

LoadedCars = 0
ArrivedCars = 0

# traci.connect()
traci.start(['sumo-gui', '-c', 'Neural Traffic.sumocfg', "--start", "--quit-on-end"])

step = 0  # variable saves the number of current simulation steps
while step < 7000:

    North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
             traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner"))
    South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
             traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner"))
    East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
    West = traci.lanearea.getLastStepVehicleNumber("Pineda")

    SUM_HALTING_TIME = (traci.lanearea.getLastStepHaltingNumber("Nation_Highway_North_Outer") +
                        traci.lanearea.getLastStepHaltingNumber("Nation_Highway_North_Inner") +
                        traci.lanearea.getLastStepHaltingNumber("Nation_Highway_North_Outer_Forward") +
                        traci.lanearea.getLastStepHaltingNumber("Nation_Highway_North_Inner_Forward")) + \
 \
                       (traci.lanearea.getLastStepHaltingNumber("Nation_Highway_South_Outer") +
                        traci.lanearea.getLastStepHaltingNumber("Nation_Highway_South_Inner") +
                        traci.lanearea.getLastStepHaltingNumber("Nation_Highway_South_Outer_Forward") +
                        traci.lanearea.getLastStepHaltingNumber("Nation_Highway_South_Inner_Forward")) + \
 \
                       traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis") + \
                       traci.lanearea.getLastStepVehicleNumber("Pineda")

    HALTING_TIME = SUM_HALTING_TIME

    LoadedCars += traci.simulation.getLoadedNumber()
    ArrivedCars += traci.simulation.getArrivedNumber()

    traci.simulationStep()

    TimeStep.append(step)
    NorthList.append(North)
    SouthList.append(South)
    EastList.append(East)
    WestList.append(West)
    Congestion.append(North+South+East+West)
    HaltingList.append(HALTING_TIME)
    Total_LoadedCars.append(LoadedCars)
    Total_ArrivedCars.append(ArrivedCars)



    step += 1
    time.sleep(0.05)

    if (step % 50) == 0:
        North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner"))
        South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner"))
        East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
        West = traci.lanearea.getLastStepVehicleNumber("Pineda")

        NeuralTrf = "Neural Traffic Demands"
        print(NeuralTrf.center(60, '-'))

        print(f"North National Highway: {North} || South National Highway: {South}")
        print(f"F. Ponce de Leon Road: {East} || Pineda Road: {West}")

for _ in range(len(TimeStep)):
    x = _ + 2
    TimeStep[_] = TimeStep[_]/86400
    WrkSht.write('A' + str(x), TimeStep[_])
    WrkSht.write('B' + str(x), NorthList[_])
    WrkSht.write('C' + str(x), SouthList[_])
    WrkSht.write('D' + str(x), EastList[_])
    WrkSht.write('E' + str(x), WestList[_])
    WrkSht.write('F' + str(x), Congestion[_])
    WrkSht.write('G' + str(x), HaltingList[_])
    WrkSht.write('H' + str(x), Total_LoadedCars[_])
    WrkSht.write('I' + str(x), Total_ArrivedCars[_])
WrkBk.close()
traci.close()
