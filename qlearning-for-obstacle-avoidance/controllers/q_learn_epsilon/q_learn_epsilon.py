import random
import time
import numpy as np
import os
import csv

from collections import deque
from controller import Robot


qfile = open('qlog.txt','w')
q_saturation_pointfile = open('q_saturation_point.txt','w')
crewardfile = open('cummulative_rewardfile.txt','w') 
qsumfile = open('qsumfile.txt','w') 
epsilonfile = open('epsilon.txt','w') 
gradientfile = open('R_OLD.txt','w') 
rewardfile = open('rewardfile.txt','w') 
rewardquefile = open('rewardque.txt','w') 


TIME_STEP = 64
MAX_SPEED = 6.28
ACTION_TAKEN = False
FLAG=0
ALPHA=0.1 #LEARNING RATE
GAMMA=0.5 #DISCOUNT RATE
EPSILON= 0.90#EXPLORATION FACTOR
EPISODES=2
NEXT_STATE=0
ACTION=0#(0:FORWARD,1:BACKWARD,2:stop,3:LEFT)
ACTION_TAKEN=False
STATES=10
STATE = 0		#Added this
REWARD=0
ACTIONS=[1,2,3,4]
NO_OF_ACTIONS=4
Q = [[0.0] * 4 for _ in range(10)]
REWARDS = [[-10,-2,-1,10] for _ in range(10)]
CREWARD=0
qsum = None

# Resource calculation
RESOURCE_MAX = 100
R_NEW = 0
R_OLD = 60
SIZE_OF_REWARD_QUEUE = 50 # Window size
rewards_queue = deque(maxlen= SIZE_OF_REWARD_QUEUE)
resource_flag = False
difference_sum = 0 
old_gradient = 0
epsilon_start = True
saturation_queue_size = 100
saturation_queue = deque(maxlen= saturation_queue_size)
saturation_point = False



count = 0
leftSpeed=0
rightSpeed=0
gradient = 0

robot = Robot()

prox_sensors = []
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')    
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)



def forward():
    leftSpeed = 0.5*MAX_SPEED
    rightSpeed = 0.5*MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
        
def backward():
    leftSpeed = -0.5 * MAX_SPEED
    rightSpeed = -0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
        
def left():
    leftSpeed=0
    rightSpeed=0
    leftSpeed -= 0.5 * MAX_SPEED
    rightSpeed += 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)


def right():
    leftSpeed=0
    rightSpeed=0
    leftSpeed += 0.5 * MAX_SPEED
    rightSpeed -= 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

def stop():
    leftSpeed = 0
    rightSpeed = 0
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

def Obstacle_Avoider(): # should return True or False
    psValues = []
    for i in range(8):
        psValues.append(prox_sensors[i].getValue())
    obstacle = psValues[0] > 80.0 and psValues[7] > 80.0
    return True if obstacle else False

def DECAY(PARAMETER):
	PARAMETER=float(PARAMETER)*0.98
	return PARAMETER

def GET_STATE():
	STATE_NO=random.randint(0, 9)
	return STATE_NO

def UPDATE(S,NEXT_S,A,R,LEARNING_RATE,DISCOUNT_FACTOR):
	Q_OLD=Q[S][A]
	Q_MAX = max(Q[NEXT_S])
	Q_NEW = (1-LEARNING_RATE)*Q_OLD + LEARNING_RATE*(R + DISCOUNT_FACTOR*Q_MAX)
	Q[S][A]=Q_NEW
     
def RESOURCE_CALC(reward):
    global old_gradient
    rewards_queue.append(reward)
    diff_sum = old_gradient + (rewards_queue[-1] - rewards_queue[-2])
    old_gradient =  diff_sum - (rewards_queue[1] - rewards_queue[0])
    squeezing_factor =  diff_sum/SIZE_OF_REWARD_QUEUE
    R_NEW = (R_OLD + ((R_OLD/RESOURCE_MAX)* squeezing_factor))

    return R_NEW
'''
This function is for ensuring that the queue is full for the first time and then

difference_sum : Sum of the gradients of rewards within the queue
old_gradient: eases the calculation for subsequent difference sum

NOTE: 
Here we have not calculated the resource even though the queue is full
This could be done if reqd.
'''
def REWARD_ACCUMULATION(reward):
    global resource_flag, old_gradient, diff_sum
    rewards_queue.append(reward)
    if(len(rewards_queue) == rewards_queue.maxlen):
        diff_sum = 0 
        for i in range(len(rewards_queue)-1):
            diff_sum += (rewards_queue[i+1] - rewards_queue[i])
        old_gradient =  diff_sum - (rewards_queue[1] - rewards_queue[0])
        resource_flag = True


def find_saturation_point(qvalue, count, Q, threshold=0.5):
    """
    Iteratively adds values to a queue and checks for saturation.
    """
    global saturation_point
    saturation_queue.append(qvalue)
    if (len(saturation_queue) == saturation_queue.maxlen):
        print("saturation point queue is full")
        print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
        std_dev = np.std(saturation_queue)
        if std_dev < threshold:
            saturation_point = True
            q_saturation_pointfile.write(f"Iteration number: {str(count)}, Q-table: {list(Q)}")
            print(f"reached saturation point: {i}, standard_deviation: {std_dev} saturation point: {str(saturation_queue[i])} ")
            print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

def create_csv_files(directory="state_files", num_files=10):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(1, num_files + 1):
        file_name = f"state_{i}.csv"
        file_path = os.path.join(directory, file_name)
        
        data = [
            ["Iteration number", "0:FORWARD", "1:BACKWARD", "2:stop", "3:LEFT"]
        ]

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

def add_statewise_actions_to_files(q, count):
    directory = "state_files"
    
    for i in range(10):
        file_name = f"state_{i+1}.csv"
        file_path = os.path.join(directory, file_name)
        
        # Format the state values
        q_values_state = [f"{count}", f"{q[i][0]}", f"{q[i][1]}", f"{q[i][2]}", f"{q[i][3]}"]

        # Check if the file exists and append the data
        if os.path.exists(file_path):
            # Open the file in append mode and write the row
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(q_values_state)  # Use writerow for a single row

        else:
            print(f"File {file_path} does not exist.")


create_csv_files()

while robot.step(TIME_STEP) != -1:
    # Initialize sensors
    for i in range(8):
        sensor_name = 'ps' + str(i)
        sensor = robot.getDevice(sensor_name)
        sensor.enable(TIME_STEP)
        prox_sensors.append(sensor)

    for _ in range(EPISODES):
        ACTION_TAKEN = False
        obstacle = Obstacle_Avoider()

        if obstacle:
            flag = Obstacle_Avoider()
            count += 1

            if flag == 1:
                NEXT_STATE = (STATE + 1) % 10
                if NEXT_STATE < 0:
                    NEXT_STATE = 0
                print(f"STATE: {STATE}")

        if not obstacle:
            forward()
            flag = 0
        
        if flag == 1:
            PROB = np.random.uniform(0, 1)
            flag = 2
            ACTION = random.randint(0, 3) if PROB <= EPSILON else np.argmax(Q[STATE])
        
        if flag == 2:
            actions = [forward, backward, stop, left]
            actions[ACTION]()
            REWARD = REWARDS[STATE][ACTION]
            ACTION_TAKEN = True
            print(f'Action is {ACTION}')
            time.sleep(0.5)
        
        if ACTION_TAKEN:
            UPDATE(STATE, NEXT_STATE, ACTION, REWARD, ALPHA, GAMMA)
            STATE = NEXT_STATE
            CREWARD += REWARD
            
            if not resource_flag:
                print(f"reward accumulation: {list(rewards_queue)}")
                REWARD_ACCUMULATION(REWARD)
            else:
                R_OLD = RESOURCE_CALC(REWARD)
                print(f"resource calc: {R_OLD}")

            # gradientfile.write(f"{str(R_OLD)}\n")
            qsum = sum(sum(ql) for ql in Q)
            qsumfile.write(f"{str(count)}\t{str(qsum)}\n")

            if not saturation_point:
                EPSILON = DECAY(EPSILON)   
                print(f"EPSILON DECAY:{EPSILON}")             
                find_saturation_point(qsum, count, Q)
            else:
                print(f"EPSILON NO MORE DECAY:{EPSILON}")
                print("\n#########################################\n")             
                rewardquefile.write(f"Added reward: {REWARD}, Queue: {list(rewards_queue)}, R_NEW: {R_OLD}")
                print(f"epsilon entered: {R_OLD}")
                rewardquefile.write("\n**********************\n")
                EPSILON = R_OLD * 0.01
            
            gradientfile.write(f"{str(count)}\t{str(R_OLD)}\n")
            epsilonfile.write(f"{str(count)}\t{str(EPSILON)}\n")
            rewardfile.write(f"{str(count)}\t{str(REWARD)}\n")
            # Write Q matrix to qfile and compute qsum
            qfile.write(str(Q))
            qfile.write("\n******************\n")
            crewardfile.write(f"{str(count)}\t{str(CREWARD)}\n")
            
            add_statewise_actions_to_files(Q, count)  

    # Calculate the sum of all Q values

    # Write cumulative reward to rewardf
        
print(Q)
           
                
