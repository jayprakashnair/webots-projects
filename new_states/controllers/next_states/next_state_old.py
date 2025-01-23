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
log = open('log_file.txt', 'w')

TIME_STEP = 64
MAX_SPEED = 6.28
FLAG=0
EPISODES = 10000
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
NEXT_STATE=0
ACTION=0#(0:FORWARD,1:BACKWARD,2:stop,3:LEFT)
ACTION_TAKEN=False
STATE = None		#Added this
REWARD=0
ACTIONS=[1,2,3,4]
NO_OF_ACTIONS=4
Q = [[0.0] * 4 for _ in range(6)]
REWARDS = [[-10,-2,-1,10] for _ in range(6)]
CREWARD=0
count = 0
flag = 0



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

def Obstacle_Present_With_State():    
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
    left, front, right =  Obstacle_Avoider()
    
    # Determine the state based on the mapping
    state = state_mapping.get((left, front, right), None)
    
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

    ACTION_TAKEN = False
    left_obstacle, front_obstacle, right_obstacle = Obstacle_Avoider()

    obstacle = left_obstacle or front_obstacle or right_obstacle

    # print(f"obstacle: {obstacle} left_obstacle: {left_obstacle} or front_obstacle: {front_obstacle} or right_obstacle: {right_obstacle}")

    if obstacle: # if true 
        log.write("come to if part")
        log.write("\n ^^^^^^^^^^^^^^^^^^^^ \n")
        STATE = Obstacle_Present_With_State()
        log.write(f"STATE OUTSIDE: {STATE} count: {count} \n")
        flag = 1
        for count in range(EPISODES):
            log.write(f"\n inside for {count=} \n")
            log.write("\n $$$$$$$$$$ \n")
            log.write(f"{flag=}")
            if flag == 1:
                log.write(f"STATE INSIDE: {STATE} \n")
                PROB = np.random.uniform(0, 1)
                ACTION = random.randint(0, 3) if PROB <= EPSILON else np.argmax(Q[STATE])
                actions = [forward, backward, stop, left]
                actions[ACTION]()
                ACTION_TAKEN = True
                REWARD = REWARDS[STATE][ACTION]
                log.write(f"{flag=}\n")
                log.write(f'Action is {ACTION}\n')
                log.write(f"reward is {REWARD}\n")
                log.write(f'EPSILON is {EPSILON}\n')
                time.sleep(0.7)
                flag = 0
                continue
            log.write(f"\n OUTSIDE FLAG 1 if condition {count=} \n")
            left_obstacle, front_obstacle, right_obstacle = Obstacle_Avoider()
            obstacle = left_obstacle or front_obstacle or right_obstacle

            if not obstacle:
                forward()
                continue
            else:
                log.write(f"{flag=}")
                flag = 2
            
            if flag == 2:
                log.write(f"\n inside flag=2 {count=} \n")
                log.write(f"{flag=}")
                NEXT_STATE = Obstacle_Present_With_State()
                # time.sleep(0.5)
                log.write(f"NEXT_STATE: {NEXT_STATE}")
                UPDATE(STATE, NEXT_STATE, ACTION, REWARD, ALPHA, GAMMA)
                log.write(f"NEXT_STATE: {NEXT_STATE}")
                # time.sleep(0.5)
                STATE = NEXT_STATE
                EPSILON = DECAY(EPSILON) 
                ACTION_TAKEN = False  
     
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
                # Write cumulative reward to rewardf
                crewardfile.write(f"{str(count)}\t{str(CREWARD)}\n")    
    else:
        log.write("go forward \n")
        forward()   
        
print(Q)
           
                
