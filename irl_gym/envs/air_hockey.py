import numpy as np
from gymnasium import Env,spaces
from itertools import combinations
import pygame
import cv2

class AirHockeyEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"]}
    def __init__(self,*,
                 params: dict,
                 render_mode : str = "rgb_array",
                 obs_type : str = "state") -> None:

        super(AirHockeyEnv,self).__init__() 
        self.time = 0 
        self._params = params 
        self.Ts = 1/params["freq"]
        self.bounds = params["screenSize"]

        self.goal_bounds = (params["goalPose"],params["goalHigh"])
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1,1, shape=(2,)) 
        self.obs_type = obs_type

        if obs_type == "screen": 
            self.observation_space = spaces.Box(0,1, shape=(*params["screenSize"],3))
        else:
            self.observation_space = spaces.Box(0,1, shape=(4+4*params["numPucks"],))

        self._state = spaces.Box(0,1,shape=(4+4*params["numPucks"],))
        
        if render_mode == "human":    
            self._screen = pygame.display.set_mode(params["screenSize"],vsync=True)

    def reset(self, seed = None, options = None):
        np.random.seed(seed) 
        self.hitter = Puck((np.random.rand(2)*(np.array(self.bounds) - 1)),
                        self._params["hitterRadius"],
                        self._params["hitterMass"])
        self.pucks = []
        for _ in range(self._params["numPucks"]):
            in_goal = True
            while(in_goal):
                pose = np.random.rand(2)*(np.array(self.bounds) - 1)
                in_goal = self.in_goal(pose)
            self.pucks.append(Puck((np.random.rand(2)*(np.array(self.bounds) - 1)),
                                    self._params["puckRadius"],
                                    self._params["puckMass"]))
        self._terminated = False
        if self.render_mode == "human" or self.obs_type == "screen":
            self.render()
        return self._get_obs(), self._get_info()
     
    def render(self):
        self._canvas = pygame.Surface(self.bounds)
        self._canvas.fill(color=(255,255,255))
        pygame.draw.rect(self._canvas,(255,0,0),(0,0,*self.bounds),width= 6)
        pygame.draw.rect(self._canvas,(0,0,255),self.goal_bounds)
        pygame.draw.circle(self._canvas,(255,0,0),self.hitter.pose,self.hitter.radius)
        for p in self.pucks:
            pygame.draw.circle(self._canvas,(0,0,0),p.pose,p.radius) 
        
        if self.render_mode == "human":
            self._screen.blit(self._canvas,self._canvas.get_rect())
            pygame.display.flip()
            pygame.time.Clock().tick(self._params["freq"])
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self._canvas)

    def close(self):
        pygame.display.quit()
        pygame.quit()
        return

    def step(self, action):
        reward = -.01
        state_init = self._state
        self.move_hitter(action)
        self.update_physics()
        obs = self._get_obs()
        state_final = self._state
        reward = self.reward(state_init,action,state_final)
        if self.render_mode == "human" or self.obs_type == "screen":
            self.render()
        return obs, reward, self._terminated, False, self._get_info()

    def update_physics(self):
        # Bounce off Hitter
        for puck in self.pucks:
            self.update_puck_pose(puck)
            if np.linalg.norm(self.hitter.pose-puck.pose) < self.hitter.radius + puck.radius:
                total_radius = self.hitter.radius + puck.radius
                d = self.hitter.pose - puck.pose
                d_norm = np.linalg.norm(d)
                overlap = total_radius - d_norm
                direction = d / d_norm
                correction = 0.5 * overlap * direction

                puck.pose -= correction
                self.hitter.pose += correction

                total_mass = (puck.mass + self.hitter.mass)
                puck.vel = ((puck.mass - self.hitter.mass) * puck.vel + 2 * self.hitter.mass * self.hitter.vel) / total_mass

        #Puck Collisions
        for puck1, puck2 in combinations(self.pucks, 2):
            if np.linalg.norm(puck1.pose - puck2.pose) < puck1.radius + puck2.radius:
                d = puck2.pose - puck1.pose
                d_norm = np.linalg.norm(d)
                overlap = puck1.radius + puck2.radius - d_norm
                direction = d / d_norm
                correction = 0.5 * overlap * direction

                puck1.pose -= correction
                puck2.pose += correction

                relative_velocity = puck2.vel - puck1.vel
                impulse = 2 * np.dot(relative_velocity, direction) / (puck1.mass + puck2.mass) * direction

                puck1.vel += impulse * puck2.mass
                puck2.vel -= impulse * puck1.mass

        # Bounce off of wall
        for p in self.pucks:
            bounce = ~self.in_bound(p)
            if bounce.any():
                p_fixed = np.clip(p.pose, p.radius, self.bounds - p.radius)
                p_diff = p_fixed - p.pose
                p.pose =  p_fixed
                p.vel = np.where(np.abs(p_diff) > 0, -p.vel, p.vel)*self._params["energyLoss"]
            p.vel *= self._params["friction"]

    def move_hitter(self,action_vel):
        self.hitter.pose = self.hitter.pose + self.Ts*self._params["maxVel"]*action_vel
        self.hitter.vel = action_vel*self.in_bound(self.hitter)
        self.hitter.pose = np.clip(self.hitter.pose,
                                  a_min= self.hitter.radius,
                                  a_max = self.bounds - self.hitter.radius)
    def in_goal(self, poses):
        return np.logical_and((self.goal_bounds[0][0] <= poses[::2]*self.bounds[0]) &
                              (poses[::2]*self.bounds[0] <= self.goal_bounds[0][0] + self.goal_bounds[0][1]),
                              (self.goal_bounds[1][0] <= poses[1::2]*self.bounds[1]) &
                              (poses[1::2]*self.bounds[1] <= self.goal_bounds[1][0] + self.goal_bounds[1][1]))

    def in_bound(self,puck):
        pose_in_bound = np.ones(shape=(2,),dtype = "bool")
        for xy, bound in enumerate(self.bounds):
            if puck.pose[xy] >= bound - puck.radius or puck.pose[xy] < puck.radius:
                pose_in_bound[xy] = False
        return pose_in_bound

    def _get_obs(self):
        puck_poses = [pose for p in self.pucks for pose in (p.pose/self.bounds) ]
        puck_vels  = [vel for p in self.pucks for vel in (p.vel)]
        self._state = np.array([*(self.hitter.pose/self.bounds), 
                                *puck_poses,
                                *(self.hitter.vel),
                                *puck_vels],dtype = "float32")

        if  self.obs_type == "screen":
            return pygame.surfarray.array3d(self._canvas)
        else:
            return self._state
        
    def reward(self,s0, action, s1):
        reward = -.05 

        #Reward for pucks being in the goal
        in_goal = self.in_goal(s1[2:(2+2*self._params["numPucks"])])
        reward += np.sum(in_goal)/self._params["numPucks"]

        #Punishment for moving inside goal 
        puck_vels = s1[4 + 2*self._params["numPucks"]:].reshape((-1,2))
        reward -= np.sum(in_goal*np.linalg.norm(puck_vels,axis=1)**(.5))/(2*self._params["numPucks"]) 

        self._terminated = (reward > .95 - 1e-1)
        return reward

    def _get_info(self):
        return {}

    def get_mouse_action(self):
        pygame.event.get()
        mouse_position = pygame.mouse.get_pos()
        dist = (mouse_position-self.hitter.pose)
        vel = dist/self.Ts
        vel_norm = np.linalg.norm(vel)
        if  vel_norm > self._params["maxVel"]:
            vel = vel/vel_norm 
        else:
            vel = vel/self._params["maxVel"]
        return vel

    def update_puck_pose(self, puck):
        puck.pose = puck.pose + self.Ts*self._params["maxVel"]*np.clip(puck.vel,-1,1)

class Puck:
    def __init__(self, pose, radius, mass):
        self.pose = pose
        self.vel = np.zeros(shape = (2,))
        self.radius = np.array(radius)
        self.mass = np.array(mass)
