import random
import time
import numpy as np

from queue import Queue
from controller import Robot


qfile = open('qlog.txt','w')
crewardfile = open('cummulative_rewardfile.txt','w') 
rewardfile = open('rewardfile.txt','w') 
qsumfile = open('qsumfile.txt','w') 
epsilonfile = open('epsilon.txt','w') 

TIME_STEP = 64
MAX_SPEED = 6.28
ACTION_TAKEN = False
FLAG=0
robot = Robot()

prox_sensors = []
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')    
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

leftSpeed=0
rightSpeed=0

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
Q = [[0.0] * 4 for _ in range(6)]
REWARDS = [[-10,-2,-1,10] for _ in range(6)]
CREWARD=0
count = 0
state_flag = False


def forward():
    print("forward")
    leftSpeed = 0.5*MAX_SPEED
    rightSpeed = 0.5*MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
        
def backward():
    print("backward")
    leftSpeed = -0.5 * MAX_SPEED
    rightSpeed = -0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
        
def left():
    print("left")
    leftSpeed=0
    rightSpeed=0
    leftSpeed -= 0.5 * MAX_SPEED
    rightSpeed += 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)


def right():
    print("right")
    leftSpeed=0
    rightSpeed=0
    leftSpeed += 0.5 * MAX_SPEED
    rightSpeed -= 0.5 * MAX_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

def stop():
    print("stop")
    leftSpeed = 0
    rightSpeed = 0
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

def Obstacle_Present_With_State(left, front, right):    
    # Create a mapping of conditions to states
    state_mapping = {
        (True, True, True): 0,
        (True, True, False): 1,
        (False, True, True): 2,
        (True, False, False): 3,
        (False, True, False): 4,
        (False, False, True): 5,
    }
    
    # Get obstacle statuses
    # left, front, right =  Obstacle_Avoider()

    
    # Determine the state based on the mapping
    state = state_mapping.get((left, front, right), None)
    print(f"state {state} and {left=},{front=},{right=}")
    
    # print(f"Obstacle_Present_With_State state:{state}")

    return state
    
def Obstacle_Avoider(): # should return True or False
    psValues = []
    for i in range(8):
        psValues.append(prox_sensors[i].getValue())
    
    # print(list(psValues))
    left_obstacle = True if (psValues[5] > 80.0 and psValues[6] > 80.0) else False
    front_obstacle = True if (psValues[0] > 80.0 and psValues[7] > 80.0) else False
    right_obstacle = True if (psValues[2] > 80.0 and psValues[1] > 80.0) else False

    return left_obstacle, front_obstacle, right_obstacle

def recursive_next_state_fn():
    global state_flag
    while True:
        left_obstacle, front_obstacle, right_obstacle = Obstacle_Avoider()
        obstacle = left_obstacle or front_obstacle or right_obstacle
        
        if not obstacle:
            print("go forward")
            forward()  # Move forward
        else:
            state = Obstacle_Present_With_State(left_obstacle, front_obstacle, right_obstacle)
            print(f"found next {state=}")
            if not ACTION_TAKEN:
                stop()
            state_flag = True
            return state
        
        if robot.step(TIME_STEP) == -1:  # Exit simulation if the user closes Webots
            break
        
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

while robot.step(TIME_STEP) != -1:
    # Initialize sensors
    
    for i in range(8):
        sensor_name = 'ps' + str(i)
        sensor = robot.getDevice(sensor_name)
        sensor.enable(TIME_STEP)
        prox_sensors.append(sensor)
    
    
    if not state_flag:
        STATE = recursive_next_state_fn()
        flag = 1
        print(f"\n OLD_STATE first loop {STATE}\n")
        print("^^^^^^^^^^^^^^^^\n")

    else:
        for _ in range(EPISODES):

            if flag == 1:
                PROB = np.random.uniform(0, 1)
                flag = 2
                ACTION = random.randint(0, 3) if PROB <= EPSILON else np.argmax(Q[STATE])
            
            if flag == 2:
                actions = [forward, backward, stop, left]
                actions[ACTION]()
                REWARD = REWARDS[STATE][ACTION]
                ACTION_TAKEN = True
                print(f'Action is {actions[ACTION]}')
                time.sleep(0.7)
            
            left_obstacle, front_obstacle, right_obstacle = Obstacle_Avoider()
            obstacle = left_obstacle or front_obstacle or right_obstacle

            if obstacle:
                count += 1
                flag = 1

                if flag == 1:
                    print("\n*******************\n")
                    print(f"OLD STATE {STATE}")

            if not obstacle:
                forward()
                flag = 0
            
            
            if ACTION_TAKEN:
                print("NEW_STATE")
                NEXT_STATE = recursive_next_state_fn()
                UPDATE(STATE, NEXT_STATE, ACTION, REWARD, ALPHA, GAMMA)
                STATE = NEXT_STATE
                EPSILON = DECAY(EPSILON)
                flag = 0
                print("\n*******************\n")
                # Calculate the sum of all Q values
                qsum = sum(sum(ql) for ql in Q)
                # print(f"Cummulative q: {qsum}")
                qsumfile.write(f"{str(count)}\t{str(qsum)}\n")
                epsilonfile.write(f"{str(count)}\t{str(EPSILON)}\n")
                
                CREWARD += REWARD
                
                rewardfile.write(f"{str(count)}\t{str(REWARD)}\n")
                # print(f"Cummulative Reward: {CREWARD}")
                # Write Q matrix to qfile and compute qsum
                qfile.write(str(Q))
                qfile.write("\n******************\n")

                # Calculate the sum of all Q values

                # Write cumulative reward to rewardf
                crewardfile.write(f"{str(count)}\t{str(CREWARD)}\n")
        
        
print(Q)
           
                
