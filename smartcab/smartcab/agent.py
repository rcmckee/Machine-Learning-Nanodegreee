import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right'] 
        self.q_table = {}  
        
        # Epsilon, alpha, gamma
        self.epsilon_explore_vs_exploit = 0.05
        self.alpha_learning_rate = 0.9
        self.gamma_future_reward_discount = 0.05

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.get_State(inputs)
        
        # TODO: Select action according to your policy
        # action = None     
        if random.random() < self.epsilon_explore_vs_exploit:
            action = random.choice(self.actions) 
        else:
            action = self.get_Action(self.state) 
               
        # Execute action and get reward
        reward = self.env.act(self, action)
             
        # TODO: Learn policy based on state, action, reward
        thisQ = self.get_Q(self.state, action) 
        nextState = self.get_State(self.env.sense(self)) 
        nextQ = max([self.get_Q(nextState, nextAction) for nextAction in self.actions])  

        self.q_table[(self.state, action)] = self.alpha_learning_rate * (reward + self.gamma_future_reward_discount * nextQ) + (1 - self.alpha_learning_rate) * thisQ

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  
        
                
    def get_State(self, inputs):
        return tuple([tuple(inputs.values()), self.next_waypoint])

    def get_Action(self, state):
        qs = [self.get_Q(state, action) for action in self.actions]
        maxQ = max(qs)
        idx = random.choice([i for i in range(len(self.actions)) if qs[i] == maxQ])
        return self.actions[idx]
              
    def get_Q(self, state, action):
        if (state, action) not in self.q_table:
            return 0
        return self.q_table[(state, action)]
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
