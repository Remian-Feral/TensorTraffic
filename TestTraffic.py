import os, sys, optparse

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


# traci.connect()
traci.start(['sumo-gui', '-c', 'Neural Traffic.sumocfg', "--start", "--quit-on-end"])

step = 0  # variable saves the number of current simulation steps
while step < 3600:
    traci.simulationStep()
    step += 1
    time.sleep(0.05)
    North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer_Forward") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner_Forward")
                 )
    South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer_Forward") +
                 traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner_Forward")
                 )
    East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
    West = traci.lanearea.getLastStepVehicleNumber("Pineda")


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

traci.close()
