import carla
import math
import time

import numpy as np
from scipy.optimize import minimize
import copy
import casadi as ca

# kf = -128916  # N/rad
# kr = -85944  # N/rad
# lf = 1.06  # m
# lr = 1.85  # m
# m = 1412  # kg
# Iz = 1536.7  # kg*m2s
# dt = None  # s
# Lk = lf*kf - lr*kr  # = 22345.439999999973

# # estimated parameters by SLSQP without steering transformation
# kf = -42475.86273837026
# kr = -95479.90923490941
# lf = 1.3394037928828246
# lr = 1.5680617480584245
# m = 1614.1710915728436
# Iz = 1537.650952579222
# estimated parameters by SLSQP with steering transformation of 70 deg
kf = -102129.8307648713
kr = -89999.99208226385
lf = 1.2868463998073212
lr = 1.6031536001974758
m = 1699.9998612883219
Iz = 2699.993738447328

dt = None
Lk = lf*kf - lr*kr

n_input = 2  # acc, steering


class Vehicle:
    def __init__(self, state=None, actor=None, horizon=10, target_v=18, carla=True, delta_t=0.05, max_iter=200):
        '''
        state: x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0
        '''
        self.carla = carla
        global dt
        dt = delta_t
        if carla:
            self.actor = actor
            self.c_loc = self.actor.get_location()
            self.transform = self.actor.get_transform()
            yaw = self.transform.rotation.yaw * np.pi / 180
            self.x = self.c_loc.x
            self.y = self.c_loc.y
            self.yaw = yaw
            self.vx = 0
            self.vy = 0
            self.direct = 1
        else:
            assert not type(state) is None
            assert len(state) == 4
            self.x, self.y, self.yaw, v = state
            self.vx = v * math.cos(self.yaw)
            self.vy = v * math.sin(self.yaw)
            self.direct = 1

        # param of MPC
        self.horizon = horizon
        self.maxiter = max_iter
        # param of car state_dot
        self.omega = 0
        # in parameter identification, we use steer_bound = 1.22
        self.steer_bound = 0.7
        # in real control process, we use steer_bound = 0.8
        # self.steer_bound = 0.8
        self.acc_lbound = -6.0
        self.acc_ubound = 3.0
        self.target_v = target_v / 3.6

    def get_state_carla(self):
        self.transform = self.actor.get_transform()
        self.x = self.transform.location.x
        self.y = self.transform.location.y
        self.z = self.transform.location.z
        yaw = self.transform.rotation.yaw
        self.yaw = yaw

        self.velocity = self.actor.get_velocity()
        self.vx = np.sqrt(self.velocity.x**2 + self.velocity.y**2)
        # self.vy = self.velocity.y
        self.vy = 0
        self.omega = self.actor.get_angular_velocity().z

        return [self.x, self.y, self.yaw, self.vx, self.vy, self.omega], self.z

    def get_v(self):
        # return self.vx/math.cos(self.yaw)
        if self.carla:
            self.get_state_carla()
            return self.get_direction() * np.sqrt(self.vx**2+self.vy**2)
        else:
            return np.sqrt(self.vx**2+self.vy**2)

    def get_direction(self):
        '''
        Get the direction of the vehicle's velocity
        '''
        yaw = np.radians(self.yaw)
        v_yaw = math.atan2(self.velocity.y, self.velocity.x)
        error = v_yaw - yaw
        if error < -math.pi:
            error += 2*math.pi
        elif error > math.pi:
            error -= 2*math.pi
        error = abs(error)
        if error > math.pi/2:
            return -1
        return 1

    def set_state(self, next_state):
        _, _, _, self.vx, self.vy, self.omega = next_state
        self.vy = -self.vy
        self.omega = -self.omega

    def set_target_velocity(self, target_v):
        self.target_v = target_v / 3.6

    def set_location_carla(self, location):
        if len(location) == 6:
            x, y, z, roll, pitch, yaw = location
        elif len(location) == 3:
            x, y, yaw = location
            z = roll = pitch = 0
        self.actor.set_transform(carla.Transform(carla.Location(x=x, y=y, z=z),
                                                 carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)))

    def update(self, acc, steer):
        self.x += (self.vx*math.cos(self.yaw)-self.vy*math.sin(self.yaw))*dt
        self.y += (self.vy*math.cos(self.yaw)+self.vx*math.sin(self.yaw))*dt
        self.yaw += self.omega*dt
        self.vx += dt*acc
        self.vy = (m*self.vx*self.vy+dt*Lk*self.omega-dt*kf*steer *
                   self.vx-dt*m*(self.vx**2)*self.omega)/(m*self.vx-dt*(kf+kr))
        self.omega = (Iz*self.vx*self.omega+dt*Lk*self.vy-dt*lf *
                      kf*steer*self.vx)/(Iz*self.vx-dt*(lf**2*kf+lr**2*kr))

    def predict(self, prev_state, u):
        x, y, yaw, vx, vy, omega = prev_state
        acc = u[0]
        steer = u[1]

        x += (vx*math.cos(yaw)-vy*math.sin(yaw))*dt
        y += (vy*math.cos(yaw)+vx*math.sin(yaw))*dt
        yaw += omega*dt
        vx += dt*acc
        vy = (m*vx*vy+dt*Lk*omega-dt*kf*steer*vx -
              dt*m*(vx**2)*omega)/(m*vx-dt*(kf+kr))
        omega = (Iz*vx*omega+dt*Lk*vy-dt*lf*kf*steer*vx) / \
            (Iz*vx-dt*(lf**2*kf+lr**2*kr))

        return (x, y, yaw, vx, vy, omega)

    def solver_basis(self, Q=np.diag([3, 3, 1, 1, 0]), R=np.diag([1, 1]), Rd=np.diag([1, 10.0])):
        cx = ca.SX.sym('cx')
        cy = ca.SX.sym('cy')
        cyaw = ca.SX.sym('cyaw')
        cvx = ca.SX.sym('cvx')
        cvy = ca.SX.sym('cvy')
        comega = ca.SX.sym('comega')

        states = ca.vertcat(cx, cy)
        states = ca.vertcat(states, cyaw)
        states = ca.vertcat(states, cvx)
        states = ca.vertcat(states, cvy)
        states = ca.vertcat(states, comega)

        n_states = states.size()[0]
        self.n_states = n_states

        cacc = ca.SX.sym('cacc')
        csteer = ca.SX.sym('csteer')
        controls = ca.vertcat(cacc, csteer)
        n_controls = controls.size()[0]
        self.n_controls = n_controls

        rhs = ca.vertcat((cx + (cvx*ca.cos(cyaw)-cvy*ca.sin(cyaw))*dt),
                         (cy + (cvy*ca.cos(cyaw)+cvx*ca.sin(cyaw))*dt))
        rhs = ca.vertcat(rhs, cyaw + comega*dt)
        if self.carla:
            rhs = ca.vertcat(rhs, cvx + dt*cacc)
            rhs = ca.vertcat(rhs, (m*cvx*cvy+dt*Lk*comega-dt*kf*csteer*cvx -
                                   dt*m*(cvx*cvx)*comega)/(m*cvx-dt*(kf+kr)))
            rhs = ca.vertcat(rhs, (Iz*cvx*comega+dt*Lk*cvy-dt *
                             lf*kf*csteer*cvx)/(Iz*cvx-dt*(lf*lf*kf+lr*lr*kr)))
        else:
            rhs = ca.vertcat(rhs, cvx + dt*cacc)
            rhs = ca.vertcat(rhs, (m*cvx*cvy+dt*Lk*comega-dt*kf*csteer*cvx -
                                   dt*m*(cvx*cvx)*comega)/(m*cvx-dt*(kf+kr)))
            rhs = ca.vertcat(rhs, (Iz*cvx*comega+dt*Lk*cvy-dt *
                             lf*kf*csteer*cvx)/(Iz*cvx-dt*(lf*lf*kf+lr*lr*kr)))

        self.f = ca.Function('f', [states, controls], [rhs], [
            'input_state', 'control_input'], ['rhs'])

        # MPC
        self.U = ca.SX.sym('U', n_controls, self.horizon)
        self.X = ca.SX.sym('X', n_states, self.horizon+1)  # state
        self.P = ca.SX.sym('P', n_states, self.horizon+1)  # reference

        self.Q = Q
        self.Q_o = copy.deepcopy(Q)
        self.R = R
        self.Rd = Rd

        self.opt_variables = ca.vertcat(ca.reshape(
            self.U, -1, 1), ca.reshape(self.X, -1, 1))
        
        self.Da = 1
        self.ac_r = 10
        self.anc_r = 100

    def reset_Q(self):
        self.Q = self.Q_o
        # print('Q is reset to ', self.Q)

    def solver_add_cost(self):
        self.obj = 0
        self.g = []  # equal constrains for multi-shooting
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []

        self.g.append(self.X[:, 0]-self.P[:, 0]) # initial equal state condition
        # add objective function and equality constraints
        for i in range(self.horizon):
            state_error = (self.X[0:5, i]-self.P[0:5, i])

            self.obj = self.obj + ca.mtimes([state_error.T, self.Q, state_error]) \
                       + ca.mtimes([self.U[:, i].T, self.R, self.U[:, i]]) # state tracking cost + control cost
            if i < (self.horizon-1):
                control_diff = self.U[:, i]-self.U[:, i+1]
                self.obj = self.obj + \
                    ca.mtimes([control_diff.T, self.Rd, control_diff]) # control smoothness cost
            # x_next_ = self.predict(X[:,i], U[:,i])
            x_next_ = self.f(self.X[:, i], self.U[:, i]) # predict the next state using bicycle dynamic model
            self.g.append(self.X[:, i+1]-x_next_) # dynamics constraint
        # self.obj += ca.mtimes([(self.X[0:5, self.horizon]-self.P[0:5, self.horizon]).T, self.Q*4, (self.X[0:5, self.horizon]-self.P[0:5, self.horizon])])

    def get_obs_centers(self, ob_, radius=1.68, carla=False):
        '''
        [Right Coordinate System]
        Get the centers of the two circles that represent the obstacle
        Parameters:
            ob_ : [x, y, yaw]
            radius : radius of the obstacle
        Returns:
            obc1 : [x, y]
            obc2 : [x, y]
        '''
        if carla:
            obc1 = [ob_[0]+radius*np.cos(ob_[2]), ob_[1]+radius*np.sin(ob_[2])]
            obc2 = [ob_[0]-radius*np.cos(ob_[2]), ob_[1]-radius*np.sin(ob_[2])]
        else:
            obc1 = [ob_[0]+radius*np.cos(ob_[2]), ob_[1]+radius*np.sin(ob_[2])]
            obc2 = [ob_[0], ob_[1]]

        return obc1, obc2
    
    # the version 1 of solver_add_soft_obs
    # def solver_add_soft_obs(self, obs_infer, ratio=400, expn = 1, carla=True):
    #     # self.obj = 0
    #     for ob_ in obs_infer: # for each obstacle
    #         for i in range(self.horizon): # for each time step
    #             obc = self.get_obs_centers(ob_[i], carla=carla)
    #             for obc_ in obc: # for each circle of an obstacle
    #                 for j in range(i): # for each time step before i (newly added to let the EV stop when it is close to the obstacle)
    #                     for selfc in self.get_obs_centers(self.X[:, j], carla=carla): # for each circle of the EV
    #                         # dist = ca.sqrt((selfc[0]-obc_[0])**2+\
    #                         #     (selfc[1]-obc_[1])**2)
    #                         dist = (selfc[0]-obc_[0])**2+(selfc[1]-obc_[1])**2
    #                         # decrease the weight when i is getting larger
    #                         self.obj += ((1/dist)**expn*ratio)

    # the version 2 of solver_add_soft_obs [double-circle obstacle]
    # def solver_add_soft_obs(self, obs_infer, ratio=500, expn = 1, carla=True, pede = False):
    #     # self.obj = 0
    #     for ob_ in obs_infer: # for each obstacle
    #         for i in range(self.horizon): # for each time step
    #             if pede == False:
    #                 obc = self.get_obs_centers(ob_[i], carla=carla)
    #             else:
    #                 obc = [ob_[i]]
    #             for obc_ in obc: # for each circle of an obstacle
    #                 for selfc in self.get_obs_centers(self.X[:, i], carla=carla): # for each circle of the EV
    #                     dist = (selfc[0]-obc_[0])**2+(selfc[1]-obc_[1])**2
    #                     # decrease the weight when i is getting larger
    #                     self.obj += 10*((1/dist)**expn)*ratio/(i+1)

    # the version 3 of solver_add_soft_obs [ellipsoid obstacle]
    def solver_add_soft_obs(self, obs_infer, sizes=None, ratio=500, expn=1, carla=True, pede=False):
    # self.obj = 0
        if len(obs_infer[0]) < self.horizon:
            for ob_ in obs_infer:
                for i in range(self.horizon - len(ob_)):
                    ob_.append(ob_[-1])
        for ob_,order in zip(obs_infer,range(len(obs_infer))):  # for each obstacle
            for i in range(self.horizon):  # for each time step
                ob_center = self.get_obs_center(ob_[i], carla=carla)
                for self_center in self.get_obs_centers(self.X[:, i], carla=carla):  # for each circle of the EV
                    if pede == False:
                        if sizes is not None and sizes[order][1] > 2.4:
                            dist = self.dist_point_to_ellipsoid(self_center, ob_center, ob_[i][2], a=sizes[order][0], b=sizes[order][1])
                        else:
                            dist = self.dist_point_to_ellipsoid(self_center, ob_center, ob_[i][2])
                    else:
                        dist = self.dist_point_to_ellipsoid(self_center, ob_center, ob_[i][2], a=0.5, b=0.5)
                    # decrease the weight when i is getting larger
                    self.obj += 12 * ((1 / dist-0.5) ** expn) * ratio / (i + 1)

    def soft_obs_apf(self, obs, ref_traj, ratio=500, expn=1, carla=False, pede=False):
        obs_apf = 0
        for ob_ in obs:
            # version for double-circle obstacle
            obc = self.get_obs_centers(ob_, carla)
            for obc_ in obc:
                for selfc in self.get_obs_centers(ref_traj[:, 0], carla):
                    # dist = ca.sqrt((selfc[0]-obc_[0])**2+\
                    #     (selfc[1]-obc_[1])**2)
                    dist = (selfc[0]-obc_[0])**2+(selfc[1]-obc_[1])**2
                    obs_apf += (1/dist)**expn*ratio

            # version for ellipsoid obstacle
            # ob_center = self.get_obs_center(ob_, carla=carla)
            # for self_center in self.get_obs_centers(ref_traj[:, 0], carla):  # for each circle of the EV
            #     if pede == False:
            #         dist = self.dist_point_to_ellipsoid(self_center, ob_center, ob_[2])
            #     else:
            #         dist = self.dist_point_to_ellipsoid(self_center, ob_center, ob_[2], a=0.5, b=0.5)
            #     # decrease the weight when i is getting larger
            #     obs_apf += 12 * ((1 / dist-0.5) ** expn) * ratio
            # obs_apf = np.abs(obs_apf)
            # print('the apf for obs is ', obs_apf)
        
        return obs_apf
    
    def solver_add_soft_ttc(self, lv, ratio=5000, expn=1, carla=True, pede=False):
        v_lv = np.sqrt(lv.get_velocity().x**2 + lv.get_velocity().y**2)
        yaw_lv = lv.get_transform().rotation.yaw
        lv = [lv.get_location().x, -lv.get_location().y, v_lv] # Add a negative sign to the y axis value to align with the right coordinate system
        for i in range(self.horizon):
            selfc = self.get_av_center(self.X[:, i], carla=carla)
            dist2 = (selfc[0]-lv[0])**2+(selfc[1]-lv[1])**2
            self.obj += ca.if_else(
                selfc[2] > lv[2],
                ratio*(np.exp(((-dist2/((selfc[2]-lv[2])**2))+6)**expn)-1),
                0
            )
            for self_center in self.get_obs_centers(self.X[:, i], carla=carla):  # for each circle of the EV
                    if pede == False:
                        dist = self.dist_point_to_ellipsoid(self_center, lv[:2], yaw_lv)
                    else:
                        dist = self.dist_point_to_ellipsoid(self_center, lv[:2], yaw_lv, a=0.5, b=0.5)
                    # decrease the weight when i is getting larger
                    self.obj += 12 * ((1 / dist-0.5) ** expn) * ratio / (2*(i + 1))
        # print('---------------the apf for ttc is attached-----------------')

    def soft_ttc_apf(self, lv, ref_traj, ratio=400, expn=1, carla=False, pede=False):
        v_lv = np.sqrt(lv.get_velocity().x**2 + lv.get_velocity().y**2)
        lv = [lv.get_location().x, -lv.get_location().y, v_lv] # Add a negative sign to the y axis value to align with the right coordinate system
        ttc_apf = 0
        selfc = self.get_av_center(ref_traj[:, 0], carla=carla)
        dist2 = (selfc[0]-lv[0])**2+(selfc[1]-lv[1])**2
        ttc_apf = ratio*np.exp(((-dist2/((selfc[2]-lv[2])**2))+6)**expn)-1
        return ttc_apf
    
    def get_av_center(self, av_, carla=False):
        av_center = [av_[0], av_[1], av_[3]] # [x, y, v_x]
        return av_center
    def get_obs_center(self, ob_, carla=False):
        '''
        [Right Coordinate System]
        Get the center of the obstacle (ellipsoid)
        Parameters:
            ob_ : [x, y, yaw]
        Returns:
            ob_center : [x, y]
        '''
        ob_center = [ob_[0], ob_[1]]
        return ob_center
    
    def dist_point_to_ellipsoid(self, point, center, yaw, a=2.4, b=1.0):
        '''
        Calculate the distance between a point and an ellipsoid
        Parameters:
            point : [x, y]
            center : [x, y]
            yaw : rotation angle of the ellipsoid
            a : semi-major axis of the ellipsoid
            b : semi-minor axis of the ellipsoid
        Returns:
            dist : distance between the point and the ellipsoid
        '''
        x, y = point
        xc, yc = center

        # Rotate the point coordinates to align with the ellipsoid
        x_rot = xc + (x - xc) * np.cos(-yaw) - (y - yc) * np.sin(-yaw)
        y_rot = yc + (x - xc) * np.sin(-yaw) + (y - yc) * np.cos(-yaw)

        # Calculate the distance using the formula for a point and an ellipse
        dist = ((x_rot - xc) ** 2 / a ** 2 + (y_rot - yc) ** 2 / b ** 2)
        return dist

    

    # MPC Soler construction
    def solver_add_hard_obs(self, obs, carla=False):
        for i in range(self.horizon):
            for ob_ in obs:
                obc = self.get_obs_centers(ob_, carla)
                for obc_ in obc:
                    for selfc in self.get_obs_centers(self.X[:, i], carla):
                        dist = ca.sqrt((selfc[0]-obc_[0])**2 +
                                       (selfc[1]-obc_[1])**2)
                        self.g.append(dist)
                        self.ubg.append(np.inf)
                        self.lbg.append(0.5)


    def solver_add_c_road_pf(self, roads_pos, yaw=0, carla=False):
        '''
        [Right Coordinate System]
        Add the cost function for the crossable road lane
        Parameters:
            roads_pos : [y1, y2, y3, ...]
            yaw : yaw of the vehicle
        Returns:
            None
        '''
        for i in range(self.horizon):
            for road_pos in roads_pos:
                for selfc in self.get_obs_centers(self.X[:, i], carla):
                    selfc = [0, ca.sin(-yaw)*selfc[0]+ca.cos(-yaw)*selfc[1]]
                    dist = ca.fabs(selfc[1] - road_pos)

                # Standard Road Lane PF
                self.obj = self.obj + ca.if_else(
                    dist < 0.5,
                    self.ac_r * (dist-1)**2,
                    0)
        # print(' Crossable road lane PF added, with a road position of ', roads_pos)
                

    def c_road_pf(self, roads_pos, ref_traj, yaw=0, carla=False):
        road_pf = 0
        for road_pos in roads_pos:
            for selfc in self.get_obs_centers(ref_traj[:, 0], carla):
                selfc = [0, np.sin(-yaw)*selfc[0]+np.cos(-yaw)*selfc[1]]
                dist = np.fabs(selfc[1] - road_pos)

            # Standard Road Lane PF
            if dist < 1.5:
                road_pf += self.ac_r * (dist-1)**2

        return road_pf


    def solver_add_nc_road_pf(self, roads_pos, yaw=0, carla=False):
        '''
        [Right Coordinate System]
        Add the cost function for the noncrossable road lane
        Parameters:
            roads_pos : [y1, y2, y3, ...]
            yaw : yaw of the vehicle
        Returns:
            None
        '''
        for i in range(self.horizon):
            for road_pos, dir in roads_pos:
                for selfc in self.get_obs_centers(self.X[:, i], carla):
                    selfc = [0, (ca.sin(-yaw)*selfc[0]+ca.cos(-yaw)*selfc[1])]
                    dist = (selfc[1] - road_pos)**2
                    # self.Q[3, 3] = self.Q_o[3,3]*(1+dist)
                    ## Use reciprocal PF
                    self.obj = self.obj + \
                        ca.if_else(
                            dist < 2.25,
                            ca.if_else(dir == 1,
                                ca.if_else(selfc[1] > road_pos-0.2, 
                                        2501, 
                                        self.anc_r * (1/(selfc[1] - road_pos))**2),
                                ca.if_else(selfc[1] < road_pos+0.2, 
                                        2501, 
                                        self.anc_r * (1/(selfc[1] - road_pos))**2)
                                ),
                            0
                        )
        # print(' Non-crossable road lane PF added, with a road position of ', roads_pos)
        

    def nc_road_pf(self, roads_pos, ref_traj, yaw=0, carla=False):
        nc_road_pf = 0
  
        for road_pos, dir in roads_pos:
            for selfc in self.get_obs_centers(ref_traj[:, 0], carla):
                selfc = [0, (np.sin(-yaw)*selfc[0]+np.cos(-yaw)*selfc[1])]
                dist = np.fabs(selfc[1] - road_pos)

                ## Use reciprocal PF
                if dist < 1.5:
                    if dir == 1:
                        if selfc[1] > road_pos-0.2:
                            nc_road_pf += 1000
                        else:
                            nc_road_pf += self.anc_r * (1/np.fabs(selfc[1] - road_pos))**2
                    else:
                        if selfc[1] < road_pos+0.2:
                            nc_road_pf += 1000
                        else:
                            nc_road_pf += self.anc_r * (1/np.fabs(selfc[1] - road_pos))**2
        
        return nc_road_pf
                    

    def solver_add_single_tr_lgt_pf(self, light_pos):
        '''
        Add traffic light cost function
        Parameters:
            light_pos : [x] 
        '''
        for i in range(self.horizon):
            dist = -(self.X[:, i][0] - light_pos)
            dist_l = 1.5 - self.X[:, i][1]
            dist_r = self.X[:, i][1] + 1.5
            
            self.obj = self.obj + 200*(1/(dist)) + 1000*(1/(dist_l))**2 + 1000*(1/(dist_r))**2 
    

    def solver_add_single_tr_lgt_pf_carla(self, lane_center_y, yaw, tl_x):
        '''
        Add traffic light cost function in Carla
        Parameters: 
            lane_center_y : y coordinate of the lane center
            yaw : yaw of the lane center point
        Returns:
            None
        '''
        for i in range(self.horizon):
            selfc = [ca.cos(-yaw)*self.X[:, i][0] - ca.sin(-yaw)*self.X[:, i][1],
                     ca.sin(-yaw)*self.X[:, i][0] + ca.cos(-yaw)*self.X[:, i][1]]
            dist = -(selfc[0] - tl_x) + 1.5
            dist_l = 1.5 - (selfc[1]-lane_center_y)
            dist_r = (selfc[1]-lane_center_y) + 1.5

            self.obj += 200*(1/(dist)) + 1000*(1/(dist_l))**2 + 1000*(1/(dist_r))**2

    def traffic_pf(self, ref_traj, lane_center_y, yaw, tl_x):
        traffic_pf = 0
 
        selfc = [np.cos(-yaw)*ref_traj[0, 0] - np.sin(-yaw)*ref_traj[1, 0],
                    np.sin(-yaw)*ref_traj[0, 0] + np.cos(-yaw)*ref_traj[1, 0]]
        dist = -(selfc[0] - tl_x) + 2
        dist_l = 1.5 - (selfc[1]-lane_center_y)
        dist_r = (selfc[1]-lane_center_y) + 1.5

            # traffic_pf += 200*(1/(dist)) + 1000*(1/(dist_l))**2 + 1000*(1/(dist_r))**2
        traffic_pf = 200*(1/(dist))
        return traffic_pf
        

    def solver_add_bounds(self, obs=np.array([]), u00 = None):
        # add the jerk bounds to self.g
        if u00 is not None:
            self.g.append(self.U[:, 0]-u00)
        for i in range(self.horizon-1):
            self.g.append((self.U[0, i+1]-self.U[0, i])) # acc
            self.g.append((self.U[1, i+1]-self.U[1, i])) # steer
            
        nlp_prob = {'f': self.obj, 'x': self.opt_variables, 'p': self.P,
                    'g': ca.vertcat(*self.g)}

        opts_setting = {'ipopt.max_iter': self.maxiter, 'ipopt.print_level': 0, 'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        # boundary for initial state and dynamic constraints
        for _ in range(self.horizon+1):
            for _ in range(self.n_states):
                self.lbg.append(0.0)
                self.ubg.append(0.0)

        # for hard obstacle constraints (normally don't use)
        for _ in range((self.horizon+1)*obs.shape[0]*2):
            self.lbg.append(0)
            self.lbg.append(0)

            self.ubg.append(np.inf)
            self.ubg.append(np.inf)
        
        # for first step of control input smoothness
        if u00 is not None:
            self.lbg.append(-3) # control rate is 20 Hz
            self.lbg.append(-0.7)
            self.ubg.append(3)
            self.ubg.append(0.7)
        # for jerk constraints
        for _ in range(self.horizon-1):
            self.lbg.append(-0.9) # control rate is 20 Hz
            self.lbg.append(-0.2) 
            self.ubg.append(0.9)
            self.ubg.append(0.2)

        # for control input constraints
        for _ in range(self.horizon):
            self.lbx.append(self.acc_lbound)
            self.lbx.append(-self.steer_bound)

            self.ubx.append(self.acc_ubound)
            self.ubx.append(self.steer_bound)

        # for state constraints
        for _ in range(self.horizon+1):
            self.lbx.append(-np.inf)
            self.lbx.append(-np.inf)
            self.lbx.append(-np.inf)
            self.lbx.append(-self.target_v)
            self.lbx.append(-np.inf)
            self.lbx.append(-np.inf)

            self.ubx.append(np.inf)
            self.ubx.append(np.inf)
            self.ubx.append(np.inf)
            self.ubx.append(self.target_v)
            self.ubx.append(np.inf)
            self.ubx.append(np.inf)

    '''
    MPC solver without initialized u_opt from last iteration of MPC
    '''
    def solve_MPC_wo(self, z_ref, z0, a_opt, delta_opt):
        xs = z_ref
        x0 = z0
        u0 = np.array([a_opt, delta_opt] *
                      self.horizon).reshape(-1, self.n_controls).T
        x_m = np.array(x0*(self.horizon+1)).reshape(-1, self.n_states).T

        c_p = np.vstack((x0, xs)).T
        init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))
        # unconstraint
        # res = self.solver(x0=init_control, p=c_p)

        # constraint
        res = self.solver(x0=init_control, p=c_p, lbg=self.lbg, lbx=self.lbx,
                          ubg=self.ubg, ubx=self.ubx)
        estimated_opt = res['x'].full()

        size_u0 = self.horizon*self.n_controls
        u0 = estimated_opt[:size_u0].reshape(self.horizon, self.n_controls)
        x_m = estimated_opt[size_u0:].reshape(self.horizon+1, self.n_states)

        if self.carla:
            return u0[0, 0], u0[0, 1], x_m
        else:
            return u0, x_m

    '''
    MPC solver with initialized u_opt from last iteration of MPC
    '''
    def solve_MPC(self, z_ref, z0, n_states, u0):
        xs = z_ref
        x_m = n_states

        c_p = np.vstack((z0.T, xs)).T
        init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))

        # unconstraint
        # res = self.solver(x0=init_control, p=c_p)

        # constraint
        start_time = time.time()
        res = self.solver(x0=init_control, p=c_p, lbg=self.lbg, lbx=self.lbx,
                          ubg=self.ubg, ubx=self.ubx)
        cost_time = time.time()-start_time
        estimated_opt = res['x'].full()

        size_u0 = self.horizon*self.n_controls
        u0 = estimated_opt[:size_u0].reshape(self.horizon, self.n_controls)
        x_m = estimated_opt[size_u0:].reshape(self.horizon+1, self.n_states)

        if self.carla:
            return u0[:, 0], u0[:, 1], x_m, cost_time
        else:
            return u0, x_m, cost_time
