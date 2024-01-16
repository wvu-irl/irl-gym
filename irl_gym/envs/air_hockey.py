import numpy as np
from gymnasium import Env,spaces
from itertools import combinations
import pygame

class AirHockeyEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"]}
    def __init__(self,*,render_mode : str = "", params: dict) -> None:
        super(AirHockeyEnv,self).__init__() 
        self.time = 0 
        self._params = params 
        self.Ts = 1/params["freq"]
        self.bounds = params["screenSize"]
        self.timeout = params["timeout"]
        self.hitter_radius = params["hitterRadius"]
        self.hitter_mass = params["hitterRadius"]
        self.puck_radius = params["puckRadius"]
        self.puck_mass = params["puckMass"]
        self.max_vel =  params["maxVel"]
        self.pucks = []
        self.goal_bounds = (params["goalPose"],params["goalHigh"])

        self.render_mode = render_mode
        self.action_space = spaces.Box(-1,1, shape=(2,)) 

        if params["obs_type"] == "rgb_array": 
            self.observation_space = spaces.Box(0,1, shape=(*params["screenSize"],3))
        elif params["obs_type"] == "dict":
            self.observation_space = spaces.Box(0,1, shape=(4+4*params["numPucks"],))
        else:
            self.observation_space = spaces.Box(0,1, shape=(4+4*params["numPucks"],))

        self._state = spaces.Box(0,1,shape=(4+4*params["numPucks"],))
        
        if render_mode == "human":    
            self._screen = pygame.display.set_mode(params["screenSize"],vsync=True)

    def reset(self, seed = None, options = None):
        np.random.seed(seed) 
        self.hitter = Puck((np.random.rand(2)*(np.array(self.bounds) - 1)),
                        self.hitter_radius,
                        self.hitter_mass)
        self.pucks = []
        for _ in range(self._params["numPucks"]):
            in_goal = True
            while(in_goal):
                pose = np.random.rand(2)*(np.array(self.bounds) - 1)
                in_goal = self.in_goal(pose)
            self.pucks.append(Puck((np.random.rand(2)*(np.array(self.bounds) - 1)),
                                    self.puck_radius, self.puck_mass))
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
        terminated = False
        state_init = self._state
        self.move_hitter(action)
        self.update_physics()
        obs = self._get_obs()
        state_final = self._state
        reward = self.reward(state_init,action,state_final)
        if self.time > self.timeout:
            terminated = True
        return obs, reward, terminated, False, self._get_info()

    def update_physics(self):
        # Bounce off Hitter
        for puck in self.pucks:
            self.update_puck_pose(puck)
            if np.linalg.norm(self.hitter.pose-puck.pose) < self.hitter.radius + puck.radius:
                total_mass = (puck.mass+self.hitter.mass)
                puck.vel = ((puck.mass-self.hitter.mass)*puck.vel + 2*self.hitter.mass*self.hitter.vel)/total_mass
                self.update_puck_pose(puck)
                d = self.hitter.pose - puck.pose
                d_norm= np.linalg.norm(puck.pose-self.hitter.pose) 
                if d_norm < self.hitter.radius + puck.radius:
                    self.hitter.pose = self.hitter.pose + (self.hitter.radius+puck.radius-d_norm)*(d/d_norm)

        # Bounce off pucks
        for puck1, puck2 in combinations(self.pucks, 2):
            if np.linalg.norm(puck1.pose-puck2.pose) < puck1.radius + puck2.radius:
               total_mass = (puck1.mass+puck2.mass)
               puck1.vel = ((puck1.mass-puck2.mass)*puck1.vel + 2*puck2.mass*puck2.vel)/total_mass
               puck2.vel = ((puck2.mass-puck1.mass)*puck2.vel + 2*puck1.mass*puck1.vel)/total_mass

        # Bounce off of wall
        for p in self.pucks:
            bounce = ~self.in_bound(p)
            p.vel = p.vel*(2*~bounce-1)
            p.pose = np.clip(p.pose,p.radius,self.bounds-p.radius)
            dvBounce = np.sqrt(np.abs(p.vel)) * self._params["energyLoss"]
            if bounce.any():
                p.vel = np.where(p.vel > 0, p.vel - dvBounce ,p.vel + dvBounce)
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
                              (poses[1::2]*self.bounds[1] <= self.goal_bounds[1][0] + self.goal_bounds[1][1])).all()
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

        if self._params["obs_type"] == "rgb_array":
            return pygame.surfarray.array3d(self._canvas)
        else:
            return self._state
        
    def reward(self,s0, action, s1):
        if self.in_goal(s1[2:(2+2*self._params["numPucks"])]):
            return 1
        else:
            return -.01

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
        puck.pose = puck.pose + self.Ts*self.max_vel*puck.vel

class Puck:
    def __init__(self, pose, radius, mass):
        self.init_pose = np.array(pose)
        self.pose = self.init_pose
        self.vel = np.zeros(shape = (2,))
        self.radius = np.array(radius)
        self.mass = np.array(mass)
