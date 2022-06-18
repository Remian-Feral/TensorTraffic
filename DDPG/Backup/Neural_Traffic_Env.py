# region Import Dependencies
import os, sys, xlsxwriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Removing CudaNN Warnings

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

if 'SUMO_HOME' in os.environ:  # Checks for Environment Variable named 'SUMO_HOME'
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci  # TraCI functionality

# from sumolib import checkBinary  # checkBinary help locate the Sumo binaries
import time  # For delay functionality


# endregion

class NeuralTraffic(gym.Env):  # The environment defined

    def __init__(self):  # Initialize the Action and Observation Space
        self.North = 0
        self.South = 0
        self.East = 0
        self.West = 0
        self.LoadedCars = 0
        self.ArrivedCars = 0
        self.Dedicated_Left = False
        self.Step = 0
        self.Done = False
        self.SUM_HALTING_TIME = 0
        self.RepeatingPhase = 0
        self.Prev_Phase = None
        self.HALTING_TIME = 0
        self.Prev_HALTING_TIME = 0

        self.Reward = None
        self.Episode_No = 1
        self.verbose = False
        self.viewer = None

        self.PrevCongestion = 0

        self.Episode_No = 1

        # Temp Data Collection per Episode
        self.TimeStep = []
        self.NorthList = []
        self.SouthList = []
        self.EastList = []
        self.WestList = []
        self.Congestion = []
        self.HaltingList = []
        self.Phase = []
        self.PhaseDur = []
        self.RewardList = []
        self.Total_LoadedCars = []
        self.Total_ArrivedCars = []

        self.WrkBk = xlsxwriter.Workbook(f"./Episodic/TrafficEpisodeNo{self.Episode_No}.xlsx")
        self.WrkSht = self.WrkBk.add_worksheet(f'Traffic Data')
        WrkSht = self.WrkSht
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
        print('WrkBk Activated at init')

        self.action_space = spaces.Box(low=np.array([0, 1], dtype=np.int32), high=np.array([2, 120], dtype=np.int32))

        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
                                            high=np.array([207, 211, 79, 87, np.inf], dtype=np.float32))
        self.state = [self.North, self.South, self.East, self.West,  # Declaration of variables
                      self.HALTING_TIME]

        self.seed()

    def new_episode(self):  # Inital new Episodes

        self.North = 0
        self.South = 0
        self.East = 0
        self.West = 0
        self.LoadedCars = 0
        self.ArrivedCars = 0
        self.Dedicated_Left = False
        self.Step = 0
        self.Done = False
        self.SUM_HALTING_TIME = 0
        self.RepeatingPhase = 0
        self.Prev_Phase = None
        self.HALTING_TIME = 0
        self.Prev_HALTING_TIME = 0

        self.Reward = 0

        self.PrevCongestion = 0


        # Temp Data Collection per Episode
        self.TimeStep = []
        self.NorthList = []
        self.SouthList = []
        self.EastList = []
        self.WestList = []
        self.HaltingList = []
        self.Total_LoadedCars = []
        self.Total_ArrivedCars = []
        self.Congestion = []
        self.Phase = []
        self.PhaseDur = []

        self.WrkBk = xlsxwriter.Workbook(f"./Episodic/TrafficEpisodeNo{self.Episode_No}.xlsx")
        self.WrkSht = self.WrkBk.add_worksheet(f'Traffic Data')
        WrkSht = self.WrkSht
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
        print('WrkBk Activated at NewEp')

        info = {}
        self.state = [self.North, self.South, self.East, self.West, self.HALTING_TIME]

        return [self.state, self.Reward, self.Done, info]

    def seed(self, seed=None):  # Seed No. for reproducibility
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        Reward = 0
        info = {}
        if (self.Step < 7000 or not self.isEmpty()) and self.Done is False:
            # 000 or not self.isEmpty()) and self.Done is False
            # region The traffic part
            # ----------------------------------Traffic Control -----------------------------------------------------

            North, South, East, West, HALTING_TIME = self.Detector()

            self.North, self.South, self.East, self.West, self.HALTING_TIME = self.Detector()  # for Data Collection

            if HALTING_TIME < 0 or HALTING_TIME > 5000:
                HALTING_TIME = 0

            if self.HALTING_TIME < 0 or self.HALTING_TIME > 5000:
                self.HALTING_TIME = 0

            self.TimeStep.append(self.Step)
            self.NorthList.append(self.North)
            self.SouthList.append(self.South)
            self.EastList.append(self.East)
            self.WestList.append(self.West)
            self.Congestion.append(self.total_congestion())
            self.HaltingList.append(self.HALTING_TIME)
            self.Total_LoadedCars.append(self.LoadedCars)
            self.Total_ArrivedCars.append(self.ArrivedCars)

            self.state = np.array([North / 207, South / 211, East / 79, West / 87,
                                   HALTING_TIME], dtype=np.int32)  # Catches Halting Time bug at 0 timestep

            self.Traffic_Demands(North, South, East, West)

            self.Phase_Selector(action)

            self.Reward = self.Reward_Function(HALTING_TIME)

            North, South, East, West, HALTING_TIME = self.Detector()

            self.state = np.array([North / 207, South / 211, East / 79, West / 87,
                                   0 if HALTING_TIME > 5000 else HALTING_TIME],
                                  dtype=np.int32)  # Catches Halting Time bug at 0 timestep

            self.HALTING_TIME = HALTING_TIME
            # self.Render
            self.Phase_Repetition(action)
            self.RewardList.append(self.Reward)
            if self.Step >= 2400 and self.isEmpty():
                self.Step = 7000
                self.Done = True
            _ = lambda x: x.center(60, '-')
            print(_('-'))

            # ---------------------------------------------------------------------------------------------------------
            # endregion

        return [self.state, self.Reward, self.Done, info]

    def Detector(self):
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

        HALTING_TIME = SUM_HALTING_TIME / 4

        return North, South, East, West, HALTING_TIME

    def Phase_Selector(self, action):

        Phase, Duration = action
        Phase = int(round(Phase))
        Duration = int(Duration)

        self.Phase.append(Phase) # Data Collection
        self.PhaseDur.append(Duration)

        print('Phase value: {:.0f} || Duration Value: {}  || HaltingTime Value: {:.2f}'.format(Phase, Duration,
                                                                                               self.HALTING_TIME))
        if Phase == 0:
            for i in range(0, Duration):
                traci.trafficlight.setPhase('320811091', 0)
                traci.trafficlight.setPhase('469173108', 0)
                traci.simulationStep()
                self.LoadedCars += traci.simulation.getLoadedNumber()
                self.ArrivedCars += traci.simulation.getArrivedNumber()
                time.sleep(0.05)
                self.Step += 1
            for i in range(0, 20):
                traci.trafficlight.setPhase('320811091', 1)
                traci.trafficlight.setPhase('469173108', 1)
                traci.simulationStep()
                self.LoadedCars += traci.simulation.getLoadedNumber()
                self.ArrivedCars += traci.simulation.getArrivedNumber()
                time.sleep(0.05)
                self.Step += 1
            self.Dedicated_Left = False  # Ensures that green phase is always follow by a yellow phase
            # before transitioning to red phase

        elif Phase == 1:
            for i in range(0, Duration):
                traci.trafficlight.setPhase('320811091', 2)
                traci.trafficlight.setPhase('469173108', 2)
                traci.simulationStep()
                self.LoadedCars += traci.simulation.getLoadedNumber()
                self.ArrivedCars += traci.simulation.getArrivedNumber()
                time.sleep(0.05)
                self.Step += 1
            for i in range(0, 20):
                traci.trafficlight.setPhase('320811091', 3)
                traci.trafficlight.setPhase('469173108', 3)
                traci.simulationStep()
                self.LoadedCars += traci.simulation.getLoadedNumber()
                self.ArrivedCars += traci.simulation.getArrivedNumber()
                time.sleep(0.05)
                self.Step += 1
            self.Dedicated_Left = True  # Ensures that green phase is always follow by a yellow phase
            # before transitioning to red phase

        elif Phase == 2:

            if self.Dedicated_Left:  # Left Turn Compatibility True
                for i in range(0, Duration):
                    traci.trafficlight.setPhase('320811091', 4)
                    traci.trafficlight.setPhase('469173108', 4)
                    traci.simulationStep()
                    self.LoadedCars += traci.simulation.getLoadedNumber()
                    self.ArrivedCars += traci.simulation.getArrivedNumber()
                    time.sleep(0.05)
                    self.Step += 1

                for i in range(0, 4):
                    traci.trafficlight.setPhase('320811091', 5)
                    traci.trafficlight.setPhase('469173108', 5)
                    traci.simulationStep()
                    self.LoadedCars += traci.simulation.getLoadedNumber()
                    self.ArrivedCars += traci.simulation.getArrivedNumber()
                    time.sleep(0.05)
                    self.Step += 1

                for i in range(0, 2):
                    traci.trafficlight.setPhase('320811091', 6)
                    traci.trafficlight.setPhase('469173108', 6)
                    traci.simulationStep()
                    self.LoadedCars += traci.simulation.getLoadedNumber()
                    self.ArrivedCars += traci.simulation.getArrivedNumber()
                    time.sleep(0.05)
                    self.Step += 1

            else:  # Left Turn Compatibility False
                for i in range(0, 4):
                    traci.trafficlight.setPhase('320811091', 6)
                    traci.trafficlight.setPhase('469173108', 6)
                    traci.simulationStep()
                    self.LoadedCars += traci.simulation.getLoadedNumber()
                    self.ArrivedCars += traci.simulation.getArrivedNumber()
                    time.sleep(0.05)
                    self.Step += 1

                for i in range(0, Duration):
                    traci.trafficlight.setPhase('320811091', 4)
                    traci.trafficlight.setPhase('469173108', 4)
                    traci.simulationStep()
                    self.LoadedCars += traci.simulation.getLoadedNumber()
                    self.ArrivedCars += traci.simulation.getArrivedNumber()
                    time.sleep(0.05)
                    self.Step += 1

                for i in range(0, 4):
                    traci.trafficlight.setPhase('320811091', 5)
                    traci.trafficlight.setPhase('469173108', 5)
                    traci.simulationStep()
                    self.LoadedCars += traci.simulation.getLoadedNumber()
                    self.ArrivedCars += traci.simulation.getArrivedNumber()
                    time.sleep(0.05)
                    self.Step += 1

                for i in range(0, 2):
                    traci.trafficlight.setPhase('320811091', 6)
                    traci.trafficlight.setPhase('469173108', 6)
                    traci.simulationStep()
                    self.LoadedCars += traci.simulation.getLoadedNumber()
                    self.ArrivedCars += traci.simulation.getArrivedNumber()
                    time.sleep(0.05)
                    self.Step += 1

    def Phase_Repetition(self, action):  # Checks for repetition

        Phase, _ = action

        Phase = int(round(Phase))

        if self.Prev_Phase is None:

            self.Prev_Phase = Phase
            print('Previous Phase value: {:.0f} || Repetition Value: {}'.format(self.Prev_Phase, self.RepeatingPhase))

        elif self.Prev_Phase != Phase and self.RepeatingPhase <= 3:
            self.Reward += 100 * 3
            self.RepeatingPhase = 1
            self.Prev_Phase = Phase
            print('Phase Change is Rewarded and Repetition Ended')

        elif self.Prev_Phase == Phase and self.RepeatingPhase != 10:
            self.RepeatingPhase += 1
            print('Previous Phase value: {:.0f} || Repetition Value: {}'.format(self.Prev_Phase, self.RepeatingPhase))

        elif self.RepeatingPhase == 10:
            _ = self.total_congestion() ** (self.LoadedCars - self.ArrivedCars)
            self.Reward -= 7000 * (_ * round(self.HALTING_TIME))
            self.Done = True
            self.Step = 7000
            print('Repeated Phase Ending Episode')

        else:
            self.Prev_Phase = None
            self.RepeatingPhase = 1
            print('Repetition Ended')

    def Reward_Function(self, HALTING_TIME):

        Reward = 0

        if self.Prev_HALTING_TIME == 0 and HALTING_TIME <= 5000:
            self.Prev_HALTING_TIME = HALTING_TIME
            Reward = 0
            print('The Reward:{} || Vehicles Remaining: {} \n'
                  'PrevHaltingTime:{} || HaltingTime:{}'
                  .format(Reward, self.total_congestion(), self.Prev_HALTING_TIME, HALTING_TIME))

        elif self.Prev_HALTING_TIME != 0 and HALTING_TIME <= 5000:
            if self.Prev_HALTING_TIME == HALTING_TIME and self.isEmpty():
                Reward -= HALTING_TIME
            else:
                Reward += self.Prev_HALTING_TIME - HALTING_TIME
                Reward -= self.total_congestion()
            print('The Reward:{} || Vehicles Remaining: {} \n'
                  'PrevHaltingTime:{} || HaltingTime:{}'
                  .format(Reward, self.total_congestion(), self.Prev_HALTING_TIME, HALTING_TIME))
            self.Prev_HALTING_TIME = HALTING_TIME

        return Reward

    def isEmpty(self, Status=False, Detector=None):  # Checks for Empty Intersections

        if Detector is not None:
            if Detector == 'North':
                North = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer") +
                         traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner") +
                         traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Outer_Forward") +
                         traci.lanearea.getLastStepVehicleNumber("Nation_Highway_North_Inner_Forward")
                         )
                if North:
                    return False
                else:
                    return True

            if Detector == 'South':
                South = (traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer") +
                         traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner") +
                         traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Outer_Forward") +
                         traci.lanearea.getLastStepVehicleNumber("Nation_Highway_South_Inner_Forward")
                         )

                if South:
                    return False
                else:
                    return True

            if Detector == 'East':
                East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
                if East:
                    return False
                else:
                    return True

            if Detector == 'West':
                East = traci.lanearea.getLastStepVehicleNumber("San_Pedro_Libis")
                if East:
                    return False
                else:
                    return True

        elif Status is False and Detector is None:
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

            NEWS = North + South + East + West

            if NEWS:
                return False
            else:
                return True
        else:
            return True

    def total_congestion(self):
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

        _ = North + South + East + West
        return _

    def Traffic_Demands(self, North, South, East, West):
        NeuralTrf = "Traffic Demands"

        print(NeuralTrf.center(60, '-'))

        print(f"North National Highway: {North} || South National Highway: {South}")
        print(f"F. Ponce de Leon Road: {East} || Pineda Road: {West}")

    def Render(self):

        # WrkBk = xlsxwriter.Workbook(f"./Episodic/TrafficEpisodeNo{self.Episode_No}.xlsx")
        # WrkSht = WrkBk.add_worksheet('Traffic Data')
        # WrkSht = WrkSht
        # WrkSht.write('A1', 'At Timestep')
        # WrkSht.write('B1', 'North Density Lane')
        # WrkSht.write('C1', 'South Density Lane')
        # WrkSht.write('D1', 'East Density Lane')
        # WrkSht.write('E1', 'West Density Lane')
        # WrkSht.write('F1', 'Total Congestion')
        # WrkSht.write('G1', 'Halting Time')
        # WrkSht.write('H1', 'Phase')
        # WrkSht.write('I1', 'Phase Duration')
        # WrkSht.write('J1', 'Reward')
        # WrkSht.write('K1', 'Total Loaded Cars')
        # WrkSht.write('L1', 'Total Arrived Cars')
        # WrkSht.set_column(0, 12, width=30)

        WrkSht = self.WrkSht

        for _ in range(len(self.TimeStep)):
            x = _ + 2
            self.TimeStep[_] = self.TimeStep[_]/86400
            WrkSht.write('A' + str(x), self.TimeStep[_])
            WrkSht.write('B' + str(x), self.NorthList[_])
            WrkSht.write('C' + str(x), self.SouthList[_])
            WrkSht.write('D' + str(x), self.EastList[_])
            WrkSht.write('E' + str(x), self.WestList[_])
            WrkSht.write('F' + str(x), self.Congestion[_])
            WrkSht.write('G' + str(x), self.HaltingList[_])
            WrkSht.write('H' + str(x), self.Phase[_])
            WrkSht.write('I' + str(x), self.PhaseDur[_])
            WrkSht.write('J' + str(x), self.RewardList[_])
            WrkSht.write('K' + str(x), self.Total_LoadedCars[_])
            WrkSht.write('L' + str(x), self.Total_ArrivedCars[_])

        self.WrkBk.close()
        self.Episode_No += 1

        pass

    def ArrivedReward(self):
        self.Reward += (self.ArrivedCars - (self.LoadedCars - self.ArrivedCars)) * 10

    def reset(self):
        try:
            traci.start(['sumo-gui', '-c', 'Neural Traffic.sumocfg', "--start", "--quit-on-end"])
        except traci.TraCIException:
            self.Render()
            self.ArrivedReward()
            print("ENDING EPISODE ELSE WAS ACTIVATED")

            traci.close()
            traci.start(['sumo-gui', '-c', 'Neural Traffic.sumocfg', "--start", "--quit-on-end"])
        self.new_episode()

        return np.int32(np.array([self.North, self.South, self.East, self.West, self.HALTING_TIME]))


register(
    id='NeuralTraffic-v1',
    entry_point=f'{__name__}:NeuralTraffic',
    max_episode_steps=7000
)
