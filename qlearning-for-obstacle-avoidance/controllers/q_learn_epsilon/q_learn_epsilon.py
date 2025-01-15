import random
import time
import numpy as np

from collections import deque
from controller import Robot


qfile = open('qlog.txt','w+')
rewardfile = open('rewardfile.txt','w+') 
qsumfile = open('qsumfile.txt','w+') 
epsilonfile = open('epsilon.txt','w+') 
epsilonnewfile = open('epsilonnew.txt','w+') 
gradientfile = open('gradient.txt','w+') 


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

# Resource calculation
RESOURCE_MAX = 100
R_NEW = 0
R_OLD = 60
SIZE_OF_REWARD_QUEUE = 50 # Window size
rewards_queue = deque(maxlen= SIZE_OF_REWARD_QUEUE)
resource_flag = False
difference_sum = 0 
old_gradient = 0



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
    diff_sum = old_gradient + (rewards_queue[-2] - rewards_queue[-1])
    old_gradient =  diff_sum - (rewards_queue[0] - rewards_queue[1])
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
            diff_sum += (rewards_queue[i] - rewards_queue[i+1])
        old_gradient =  diff_sum - (rewards_queue[0] - rewards_queue[1])
        resource_flag = True


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
            EPSILON = DECAY(EPSILON)
            epsilonfile.write(f"{str(EPSILON)}\n")
            
            if resource_flag == False:
                REWARD_ACCUMULATION(REWARD)
            else:
                R_OLD = RESOURCE_CALC(REWARD)

            gradientfile.write(f"{str(R_OLD)}\n")

            if EPSILON < 0.35:
                EPSILON = R_OLD * 0.01
                epsilonnewfile.write(f"{str(EPSILON)}\n")
                time.sleep(7)

        
    CREWARD += REWARD
    count += 1

    print(f"Cummulative Reward: {CREWARD}")
    # Write Q matrix to qfile and compute qsum
    qfile.write(str(Q))
    qfile.write("\n******************\n")

    # Calculate the sum of all Q values
    qsum = sum(sum(ql) for ql in Q)
    print(f"Cummulative q: {qsum}")
    qsumfile.write(f"{str(count)}\t{str(qsum)}\n")

    # Write cumulative reward to rewardf
    rewardfile.write(f"{str(count)}\t{str(CREWARD)}\n")
        
print(Q)
           
                
