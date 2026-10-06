from solver import *

from utils import *
from constants import *

from os.path import exists

import copy

DAYMONTHS = [31,28,31,30,31,30,31,31,30,31,30,31]
HOURMONTHS = [sum(DAYMONTHS[:i]*24) for i in range(len(DAYMONTHS)+1)]
hours_to_months = lambda hours: sum(i-1 + (hours-HOURMONTHS[i-1])/(HOURMONTHS[i]-HOURMONTHS[i-1])
                                    for i in range(1,len(DAYMONTHS)+1)
                                        if HOURMONTHS[i-1] < hours <= HOURMONTHS[i])

import numpy as np
import pandas

### Default Stub Predictor

"""
predictor(table,t,i0,imax)
            function for predicting t-th entry in the table,
            i0 being the prediction being made and imax maximum for t
"""

def truth_predictor(table, t, i0=None, imax=None):
    return table[t]

### 

class Optimizer():
    """
    CSV-based Optimizer class (Applied Energy '21 cost-optimization model):
        -Loads input data.
        -Solves non-battery-equipped cost-optimization.
        -Deleguates battery-equipped problem optimization to a LP-solver.
        -Can use different solver models depending on assumptions.
        -Uses a single price time-series for the entire set of households.
        -Can use different solar profiles, by default same one for all households.
        -Uses 100 consumption profiles by default.
    """
    # NEW DEFAULT OPTIMIZER: receives list/iterable/numpy_arrays as input
    # Another change: el_in/el_out (kWh) etc should be with 2 digits only; cost as well
    # ie that is the smallest unit of measure
    # Yet another change: do not save the solver within the optimizer use a one-time use object
    # Also once solved, remember either battery_difference and/or el_balance (rounded!)
    # create functions get_el_in(h), get_el_out(h), get_battery_levels(h)
    def __init__(self, pv_file=PV_FILE_DEFAULT,
                 battery_file=BAT_FILE_DEFAULT,
                 cons_file=CONS_FILE_DEFAULT,
                 sun_file=SUN_FILE_DEFAULT,
                 price_file=PRICE_FILE_DEFAULT,
                 tax=TAX_DEFAULT,
                 el_tax=EL_TAX_DEFAULT,
                 el_net=EL_NET_DEFAULT,
                 predictor=truth_predictor, # remove from the optimizer main-class
                 solver=Solver,
                 solverex_name=DEFAULT_LP_SOLVER,
                 single_sun=True
                 ):
         #converts data into numpy arrays, makes faster execution
        self.pv = np.array(load(pv_file))
        self.battery = np.array(load(battery_file))
        self.cons = np.array(loadcsv(cons_file))
        
        if single_sun:
            sun_profile = load(sun_file)
            self.sun = [sun_profile]*len(self.cons)
        else:
            self.sun = loadcsv(sun_file)

        self.sun = np.array(self.sun)

        # If only one solar profile is given, use it for all households
        if self.sun.ndim == 1:
            self.sun = np.tile(
                self.sun,
                (len(self.cons), 1)
            )

        self.price = np.array(load(price_file))

        # Alle vier Haushalte bekommen dasselbe Haushaltsstrompreisprofil.
        if self.price.ndim == 1:
            self.price = np.tile(
                self.price,
                (len(self.cons), 1)
            )

        # [extension] this should also be loaded. params.csv
        self.tax = tax
        self.el_tax = el_tax
        self.el_net = el_net
        self.snm = solverex_name
        
        self.predictor = predictor
        self.solver_class = solver
        self.last_solver = None
        self.el_in = [None] * len(self.cons)
        self.el_out = [None] * len(self.cons)

    ### utils ###

    def is_prosumer(self, i):
        return self.pv[i]+self.battery[i] > 0

    def is_consumer(self, i):
        return not self.is_prosumer(i)
    
    def get_prosumers(self):
        return [i for i in range(len(self.cons)) if self.is_prosumer(i)]

    def get_consumers(self):
        return [i for i in range(len(self.cons)) if self.is_consumer(i)]

    def get_all_users(self):
        return [i for i in range(len(self.cons))]
    
    ### main optimization functions ###

    # create solver depending on the model used

    def _solver_parameters(self):
        return [self.price, self.predictor, self.tax, self.el_tax, self.el_net, self.snm]

    def _solver_param_household(self, h):
        return [self.pv[h], self.battery[h], self.cons[h], self.sun[h]] + self._solver_parameters()

    def _solver_param_all(self):
        return [self.pv, self.battery, self.cons, self.sun] + self._solver_parameters()
    
    def _translosses_solver(self, losses, grid_fee=0):
        return SolverTransmissionLosses(losses, grid_fee, *self._solver_param_all())

    def _distlosses_solver(self, loss, gridfee=0, chgbatloss=0, disbatloss=0):
        return SolverGeneralModel(chgbatloss, disbatloss, loss, gridfee, *self._solver_param_all())

    # optimize yearly cost for individual houses

    def optimize(self, h=None, n=None, solver_class=None, solver=None):
        if h is None:
            return self.optimize_all()
        
        if self.battery[h] == 0:
            return self.bill_nobat(h)

        if solver:
            solverh = solver(h)
        else:
            solver_class = solver_class if solver_class else self.solver_class
            solverh = solver_class(*self._solver_param_household(h))

        # backup of last solver used, essentially for debugging reasons
        self.last_solver = solverh      
        self.yearcost = solverh.solve(n)
        
        # remember optimization choices -- recompute them from batlevel for forecast optimization
        self.bat = solverh.batlevel[1:]
        self.el = [round(self.cons[h][j] - self.sun[h][j]*self.pv[h]
                        +(solverh.batlevel[j+1]-solverh.batlevel[j]),5)
                        for j in range(len(self.price))]
        self.el_in = [el if el>0 else 0 for el in self.el]
        self.el_out = [-el if el<0 else 0 for el in self.el]

        # all quantities are rounded to 5 numbers (meaningless information beyond)
        return round(self.yearcost,5)

    # optimize community cost

    def optimize_single_community(self, n=None):
        return self.optimize_community(list(range(len(self.pv))),[],n)

    def optimize_batlosses(self, losses, dislosses=0, h=None, n=None):
        def _battery_solver(h):
            return SolverBatteryLosses(losses, dislosses, *self._solver_param_household(h))
        return self.optimize(h, n, _battery_solver)

    def optimize_translosses(self, losses, grid_fee=0, n=None):
        return self._translosses_solver(losses, grid_fee).solve(n)

    # expect n*n matrixes for losses and fees
    def optimize_distlosses(self, losses, grid_fee=None, charge_loss=0, discharge_loss=0, n=None):
        fee = grid_fee if grid_fee else [[0]*len(self.cons)]*len(self.cons)
        return self._distlosses_solver(losses, fee, charge_loss, discharge_loss).solve(n)

    # cooperative gain: individual cost - community cost
    
    def cooperative_gain(self):
        return self.optimize_all()-self.optimize_community()

    def avg_cooperative_gain(self):
        return self.cooperative_gain() / len(self.pv)

    ### subset optimizations ###

    # in-place optimization (no copy required here)
    def optimize_subset(self, equipped, unequipped=[]):
        return sum([self.optimize(h) for h in equipped]+[self.bill_nopvbat(h) for h in unequipped])

    def optimize_community(self, equipped, unequipped=[], n=None):
        return self.subset(equipped, unequipped).aggregate_all().optimize(0, n)

    def avg_cooperative_gain_subset(self, equipped_houses, unequipped=[]):
        return ((self.optimize_subset(equipped_houses, unequipped)
                -self.optimize_community_subset(equipped_houses, unequipped))
                /(len(equipped_houses)+len(unequipped)))

    ### cost functions ###
    
    def cost(self, t, elin, elout):
        bought = elin * (self.price[t]*self.tax + self.el_tax)
        sold = elout * (self.price[t] + self.el_net)
        return bought - sold

    def cost_nobat(self, h, t):
        el = self.cons[h][t] - self.sun[h][t]*self.pv[h]
        return self.cost(t, el if el > 0 else 0, -el if el < 0 else 0)
    
    ### other utils ###
    
    def avg_coop_gain(self):
        return self.cooperative_gain()/len(range(self.pv))

    # solar profiles stay untouched -- perhaps averaging them is best if different
    def aggregate_all(self):
        self.sun = [self.sun[0]]
        self.pv = [sum(self.pv)]
        self.battery = [sum(self.battery)]   
        
        self.cons = [[sum(house[t] for house in self.cons) for t in range(len(self.price))]]
        return self

    # short alias for aggregate_all()
    def all(self):
        return self.aggregate_all()

    def optimize_all(self):
        return sum(self.optimize(h) for h in range(len(self.cons)))

    # set same profile for all households
    def set_pv_profile(self, profile):
        self.sun = [profile]*len(self.pv)
        return self
            
    ### some more optimizer utils ###
        
    def remove_household(self, h):
        del self.pv[h]
        del self.battery[h]
        del self.cons[h]
        return self

    def remove_households(self, households):
        for h in households:
            self.remove_household(h)
        return self

    def keep_households(self, households):
        self.pv = [self.pv[h] for h in range(len(self.pv)) if h in households]
        self.battery = [self.battery[h] for h in range(len(self.pv)) if h in households]
        self.cons = [self.cons[h] for h in range(len(self.pv)) if h in households]
        return self

    def keep_n_households(self, n):
        self.remove_households(range(n,len(pv)))
        return self

    def subset(self, equipped, unequipped=[]):
        equipped, unequipped = list(equipped), list(unequipped)
        opt = copy.copy(self) # [extension] instead create a new object with the updated constructor
        opt.pv = [opt.pv[h] for h in equipped] + [0]*len(unequipped)
        opt.battery = [opt.battery[h] for h in equipped] + [0]*len(unequipped)
        opt.cons = [opt.cons[h] for h in equipped+unequipped]
        opt.sun = [opt.sun[h] for h in equipped+unequipped]
        return opt

    def copy(self):
        return copy.deepcopy(self)

    def new_household_parameters(self, pv, battery, consumptions):
        opt = copy.copy(self)
        opt.pv = pv
        opt.battery = battery
        opt.cons = consumptions
        return opt

    def hours(self, start, end):
        opt = copy.copy(self)
        opt.sun = [self.sun[h][start:end] for h in range(len(self.cons))]
        opt.price = self.price[start:end]
        opt.cons = [self.cons[h][start:end] for h in range(len(self.cons))]
        return opt

    def get_prosumers(self):
        return [p for p in range(len(self.pv)) if self.pv[p] > 0]

    def get_avg_loads(self):
        return [sum(user)/len(user) for user in self.cons]

    ### these "optimizations" don't require a solver ###
             
    def bill_nopvbat(self, h):
        """Computes the bill of user h (without taking into account any resources).

        PV/battery if any are ignored and el_in/el_out are set.

        Args:
            h (int): end-user h

        Returns:
            int: cost of user h over billing period ignoring any resources.
        """
        
        #self.el_in[h] = [0]*len(self.price)
        #self.el_out[h] = [0]*len(self.price)
        #return sum(self.cost(t,self.cons[h][t],0) for t in range(len(self.price)))
        
        #vectorized cost-computation
        self.el_in[h] = np.zeros(len(self.price))
        self.el_out[h] = np.zeros(len(self.price))
        rate = self.price * self.tax + self.el_tax
        return np.dot(self.cons[h], rate)     

            
    def bill_nobat(self, h=None):
        if h is not None:
            self.el = [round(self.cons[h][t]-self.sun[h][t]*self.pv[h],5)
                       for t in range(len(self.price))]
    def bill_nobat(self, h=None):
        if h is not None:
            self.el = [round(self.cons[h][t]-self.sun[h][t]*self.pv[h],5)
                       for t in range(len(self.price))]
            self.el_in = [(self.el[t] if self.el[t]>0 else 0) for t in range(len(self.price))]
            self.el_out = [(-self.el[t] if self.el[t]<0 else 0) for t in range(len(self.price))]
            return sum(self.cost_nobat(h, t) for t in range(len(self.price)))
        return sum(self.bill_nobat(h) for h in range(len(self.cons)))

            #vectorize bill_nobat
##            el = self.cons[h] - self.sun[h] * self.pv[h]
##            self.el_in = np.clip(el, 0, None)
##            self.el_out = np.clip(-el, 0, None)
##
##            return np.sum(self.el_in * (self.price*self.tax + self.el_tax)
##              - self.el_out * (self.price + self.el_net))

    def greedy_cost(self, h=0):
        cost = battery = 0
        max_battery = self.battery[h]
        self.bat = []
        
        for t in range(len(self.price)):
            production = self.sun[h][t]*self.pv[h]
            consumption = self.cons[h][t]       
        if production >= consumption :      # charge battery
                battery += production - consumption
                if battery > max_battery:
                    cost -= (battery-max_battery)*(self.price[t] + self.el_net)
                    battery = max_battery
        else:                               # discharge battery
                battery -= consumption - production
                if battery < 0:
                    cost += (-battery) * (self.price[t]*self.tax + self.el_tax)
                    battery = 0
        self.bat.append(battery)

        return cost

    def greedy_coalition_cost(self):
        total_cost = 0
        houses = range(len(self.pv))
        local_battery = [0]*len(houses)

        for t in range(len(self.sun[0])):
            common_pool = common_demand = 0

            cost_el_in = lambda el : el * (self.price[t]*self.tax + self.el_tax)
            cost_el_out = lambda el : el * (self.price[t] + self.el_net)

            local_demand = [self.cons[h][t] for h in houses]
            
            # (1) use your own installation to cover your own consumption
            # and discharge / charge your own battery first !
            for h in houses:
                production = self.sun[h][t]*self.pv[h]

                if production >= local_demand[h] :
                    local_demand[h] = 0
                    surplus = production-local_demand[h]
                    local_battery[h] += surplus
                    if local_battery[h] > self.battery[h]:
                        common_pool += (local_battery[h]-self.battery[h])
                        local_battery[h] = self.battery[h]
                else:
                    local_demand[h] -= production
                    local_battery[h] -= local_demand[h]
                    if local_battery[h] < 0:
                        common_demand += (-local_battery[h])
                        local_battery[h] = 0

            # (2) Use common energy pool to satisfy remote demands
            if common_pool > common_demand:
                common_pool -= common_demand
            else:
                common_pool = 0
                common_demand -= common_pool

            # (3) Charge remote batteries
            if common_pool > 0:
                for h in houses:
                    if local_battery[h]+common_pool < self.battery[h]:
                        local_battery[h] += common_pool
                        break
                    else:
                        maxcharge = self.battery[h]-local_battery[h]
                        common_pool -= maxcharge

            # (4) Buy or Sell to the grid any leftover generation or demand
            total_cost += cost_el_in(common_demand)
            total_cost -= cost_el_out(common_pool)

        return total_cost

    ### some more utils

    def el_sun(self,h):
        return sum(self.sun[h])*self.pv[h]

    def original_bill(self, el_in, el_out): # recompute cost post-optimization
        def default_cost(t):
            bought = el_in[t] * (self.price[t]*self.tax + self.el_tax)
            sold = el_out[t] * (self.price[t] + self.el_net)
            return bought - sold
        return round(sum(default_cost(t) for t in range(len(self.price))), 5)

    def recompute_bill(self): # recompute cost post-optimization
        return self.original_bill(self.el_in, self.el_out)
    
       
    def individual_savings(self): # post optimization savings, after agregation
        #for t in range(len(self.price)): 
        #avoided repeated range
        T = self.price.shape[0]
        for t in range(T):           
          if t == 0:
                pool_bat = self.bat[0]
          else:
                pool_bat = self.bat[t] - self.bat[t-1]
                pool_sun = self.sun[0][t]*self.pv[0]
                pool_grid = self.el_in[t]
                

    ### static functions -- shorthand to build / save / load an Optimizer
    
    def from_folder(foldername, price_and_sun_folder=DATA_FOLDER, *args, **kwargs):
        return Optimizer(pv_file=join(foldername,PV_FILE),
                         battery_file=join(foldername,BAT_FILE),
                         cons_file=join(foldername,CONS_FILE),
                         sun_file=join(price_and_sun_folder,SUN_FILE),
                         price_file=join(price_and_sun_folder,PRICE_FILE),
                         *args, **kwargs)

    def from_folder2(foldername, *args, **kwargs):
        return Optimizer.from_folder(foldername, foldername, *args, **kwargs)

    ### shorthand to load PV profiles
    # E20, E30, N20, N30, NE20, NE30, NV20, NV30, S20, S30, SE20, SE30, SV20, SV30, V20, V30
    # original solar profile is between S20 and S30 (just under S30) for coordinate 55.5 and 13.125.
    def load_solar_profile(lat=55.5, lon=13.125, orientation='S30', pvprofiles=PV_PROFILES_DEFAULT):
        profiles = loadcsv(pvprofiles, str)
        i = next(i for i in range(len(profiles)) if float(profiles[i][0]) == lat
                 and float(profiles[i][1]) == lon and profiles[i][2] == orientation)
        return [float(x) for x in profiles[i][3:]]

    # static functions -- serialization utils
    def generate_full_optimizer(opt, binfile, alr=6, bdr=15):
        n = len(opt.cons)
        avg_cons = [sum(opt.cons[h])/len(opt.cons[h]) for h in range(n)]
        for h in range(n):
            opt.pv[h] = round(alr*avg_cons[h],2)
            opt.battery[h] = round(bdr*avg_cons[h],2)
        return opt

    # serialization (needs to check with tax values etc!!)
    def opt2221():
        return load_or_precompute(OPT_2221_BIN, lambda:Optimizer.from_folder(DATA_FOLDER_2221))
    
    def save_optimizer(input_folder=DATA_FOLDER_2221, binfile=OPT_2221_BIN):
        savedump(Optimizer.from_folder(input_folder), binfile)

    def load_optimizer():
        return Optimizer.opt2221()

    def load_full_optimizer(binfile=join(DATA_FOLDER_2221,'opt_2221_alr6_bdr15.bin')):
        return load_or_precompute(binfile,
                                  lambda:Optimizer.generate_full_optimizer(
                                      Optimizer.from_folder(DATA_FOLDER_2221),binfile))

    def export2csv(self,all_solar_profiles=False,target_dir=""):
        write2csv(join(target_dir,"pv.csv"), self.pv)
        write2csv(join(target_dir,"battery.csv"), self.battery)
        write2dmatrix2csv(join(target_dir,"cons.csv"), self.cons)
        if all_solar_profiles:
            write2dmatrix2csv(join(target_dir,"sun.csv"), self.sun)
        else:
            write2csv(join(target_dir,"sun.csv"), self.sun[0])
        write2csv(join(target_dir,"price.csv"), self.price)

    def save(self, filename):
        savedump(self, filename)

    def __repr__(self):
        return f"Optimizer with {len(self.cons)} end-users over {len(self.price)} hours."

    @staticmethod
    def load(filename):
        opt = loaddump(filename)
        opt.pv = list(opt.pv)
        opt.battery = list(opt.battery)
        opt.cons = list(list(cons_h) for cons_h in opt.cons)
        opt.sun = list(opt.sun)
        opt.price = list(opt.price)
        return opt

    def static_sharing_solver(self, allocation_keys, prosumer, p_p2p, p_feed_in=0.08, residual_price=0.0):

        self.solver = SolverStaticSharing(
            allocation_keys,
            prosumer,
            p_p2p,
            residual_price,
            p_feed_in,
            0,                  # charge_loss
            0,                  # discharge_loss
            1,                  # selfdischarge
            0,                  # syst_losses
            0,                  # trans_losses
            0,                  # grid_fee
            self.pv,
            self.battery,
            self.cons,
            self.sun,
            self.price,
            self.predictor,
            self.tax,
            self.el_tax,
            self.el_net,
            self.snm
        )

        return self.solver


## this one requires a huge clean-up!
class OptimizerExt(Optimizer):
    """
    ISGT NA'23 Extended Optimizer.
    [WARNING: This Optimizer is configured by default to use SEK (Swedish Kr) for price values.]
    Loads both household-dependent solar profiles and price profiles:
        0. Uses 2221 consumption profiles by default
        1. Load solar profile based on household's location
            * Find closest precalculated profile from position
            * Use orientation / tilt
        2. Use PV19 distribution of solar orientation / tilts
            Add 2 orientations? or even 3?
            Sum them up.
        3. Use region-dependent prices
        4. Share all data? (except consumptions?)
    """

    def __init__(self,
                 pv_file=PV_2221_FILE,
                 battery_file=BAT_2221_FILE,
                 cons_file=CONS_2221_FILE,
                 sun_file=INDIVIDUAL_SUN_FILE,
                 price_file=INDIVIDUAL_PRICE_FILE,
                 tax=TAX_DEFAULT,
                 el_tax=EL_TAX_DEFAULT,
                 el_net=EL_NET_DEFAULT,
                 predictor=truth_predictor,
                 solver=SolverGeneralModel,
                 solverex_name=DEFAULT_LP_SOLVER):
        self.pv = load(pv_file)
        self.battery = load(battery_file)
        self.cons = loadcsv(cons_file)
        self.sun = loadcsv(sun_file)
        self.price = loadcsv(price_file)
        self.predictor = predictor
        self.default_parameters()
        self.el_in = [None]*len(self.cons)
        self.el_out = [None]*len(self.cons)
        

    def default_parameters(self):
        self.charge_loss = 0.025
        self.discharge_loss = 0.025
        self.selfdischarge = 0.99995
        self.syst_losses = 0.00             
        self.trans_losses = 0.03            
        self.tax = 1.25
        self.el_tax = 0.725                 
        self.el_net = 0.075
        self.grid_fee = (0.154*1.25)/4      
        self.fixedfee = 206.25

    def best_parameters(self):
        self.charge_loss = 0
        self.discharge_loss = 0
        self.selfdischarge = 1
        self.syst_losses = 0
        self.trans_losses = 0
        self.grid_fee = 0

    def subset(self, equipped, unequipped=[]):
        equipped, unequipped = list(equipped), list(unequipped)
        opt = copy.copy(self)
        opt.pv = [opt.pv[h] for h in equipped] + [0]*len(unequipped)
        opt.battery = [opt.battery[h] for h in equipped] + [0]*len(unequipped)
        opt.cons = [opt.cons[h] for h in equipped+unequipped]
        opt.sun = [opt.sun[h] for h in equipped+unequipped]
        opt.price = [opt.price[h] for h in equipped+unequipped]
        opt.el_in = [[0]*len(opt.price[0]) for h in equipped+unequipped]
        opt.el_out = [[0]*len(opt.price[0]) for h in equipped+unequipped]
        opt.imported = [[0]*len(opt.price[0]) for h in equipped+unequipped]
        opt.exported = [[0]*len(opt.price[0]) for h in equipped+unequipped]
        return opt

    def hours(self, start, end):
        opt = copy.copy(self)
        opt.sun = [self.sun[h][start:end] for h in range(len(self.cons))]
        opt.price = [self.price[h][start:end] for h in range(len(self.cons))]
        opt.cons = [self.cons[h][start:end] for h in range(len(self.cons))]
        return opt

    def aggregate_all(self):
        self.sun = [[avg(self.sun[h][t]
                         for h in range(len(self.pv))) for t in range(len(self.sun[0]))]]
        self.price = [[avg(self.price[h][t]
                           for h in range(len(self.pv))) for t in range(len(self.price[0]))]]
        self.pv = [sum(self.pv)]
        self.battery = [sum(self.battery)]
        self.cons = [[sum(house[t] for house in self.cons) for t in range(len(self.cons[0]))]]
        return self

    def new_cost(self, price_t, elin, elout):
        bought = elin * (price_t*self.tax + self.el_tax)
        sold = elout * (price_t + self.el_net)
        return bought - sold

    def cost_nobat(self, h, t):
        el = self.cons[h][t] - self.sun[h][t]*self.pv[h]
        return self.new_cost(self.price[h][t], el if el > 0 else 0, -el if el < 0 else 0)

    def bill_nopvbat(self, h):
        #self.el_in[h] = [0]*len(self.price[h])
        #self.el_out[h] = [0]*len(self.price[h])
        #return sum(self.cost(t,self.cons[h][t],0) for t in range(len(self.cons[0])))
        
        #vectorization
        cons = np.array(self.cons[h])
        price = np.array(self.price[h])
        self.el_in[h] = np.zeros(len(price))
        self.el_out[h] = np.zeros(len(price))
        return self._bill_core(cons, price)
    

    def bill_nobat(self, h=None):
        if h is not None:
            self.el = [round(self.cons[h][t]-self.sun[h][t]*self.pv[h],5)
                       for t in range(min(len(self.cons[h]),len(self.sun[h])))]
            self.el_in[h] = [(self.el[t] if self.el[t]>0 else 0) for t in range(len(self.cons[0]))]
            self.el_out[h] = [(-self.el[t] if self.el[t]<0 else 0) for t in range(len(self.cons[0]))]
            self.bat = [[0]*len(self.price[h]) for _ in range(len(self.cons))]
            self.imported = [[0]*len(self.price[h]) for _ in range(len(self.cons))]
            self.exported = [[0]*len(self.price[h]) for _ in range(len(self.cons))]
            return sum(self.cost_nobat(h, t) for t in range(len(self.cons[h])))
        return sum(self.bill_nobat(h) for h in range(len(self.cons)))

    def make_standard_solver(self, solverclass=Solver, h=0):
        return      solverclass(self.pv[h], self.battery[h],
                             self.cons[h], self.sun[h], self.price[h],
                             self.predictor, self.tax,
                             self.el_tax, self.el_net, 'cbc')
    
    def original_solver_optimize(self, h):
        self.solver = self.make_standard_solver(Solver, h)
        self.months = len(self.price[0])/(24*365/12)
        self.totalcost = self.fixedfee*self.months + self.solver.solve(len(self.price[h]))
        return round(self.totalcost,2)

    # completely equivalent to above for 1 house only
    def transloss_optimize(self):
        houses = range(len(self.pv))
        self.solver = SolverTransmissionLosses(self.trans_losses, self.grid_fee,
                                    [self.pv[h] for h in houses], [self.battery[h] for h in houses],
                                    [self.cons[h] for h in houses], [self.sun[h] for h in houses],
                        [avg(self.price[h][t] for h in houses) for t in range(len(self.price[0]))],
                                     self.predictor, self.tax,
                                    self.el_tax, self.el_net, 'cbc')
        self.yearcost = len(self.pv)*self.fixedfee*12 + self.solver.solve(len(self.price[0]))
        return round(self.yearcost,2)

    def general_solver(self, solverclass=SolverGeneralModel):
        self.solver = solverclass(self.charge_loss, self.discharge_loss, self.selfdischarge,
                                    self.syst_losses, self.trans_losses, self.grid_fee,
                                    self.pv, self.battery,
                                    self.cons, self.sun, self.price, self.predictor, self.tax,
                                    self.el_tax, self.el_net, 'cbc')

    def recompute_standard_exchanges(self, agg_bat):
        self.el_in = [[0]*len(self.price[0]) for _ in range(len(self.cons))]
        self.el_out = [[0]*len(self.price[0]) for _ in range(len(self.cons))]
        self.bat = [[0]*len(self.price[0]) for _ in range(len(self.cons))]
        self.imported = [[0]*len(self.price[0]) for _ in range(len(self.cons))]
        self.exported = [[0]*len(self.price[0]) for _ in range(len(self.cons))]

        p = next(i for i in range(len(self.cons)) if self.pv[i] > 0)
        
        # we assume 1 prosumer only
        for t in range(len(self.price[0])):

            # PROSUMMER
            self.bat[p][t] = agg_bat[t]
            el_gen = self.pv[p]*self.sun[p][t]
            bat_diff = agg_bat[t]-agg_bat[t-1] if t > 0 else agg_bat[t]
            prosumer_balance = round(el_gen - self.cons[p][t] - bat_diff, 5)
            
            if prosumer_balance > 0:
                electricity_pool = prosumer_balance
            else:
                self.el_in[p][t] = -prosumer_balance
                electricity_pool = 0
        
            # USUAL CONSUMMERS
            for i in range(len(self.cons)):
                if i != p: 
                    
                    if electricity_pool > 0:        # substract imported portion if any
                        self.imported[i][t] = min(electricity_pool, self.cons[i][t])
                        electricity_pool -= self.imported[i][t]
                        self.el_in[i][t] = self.cons[i][t]-self.imported[i][t]
                    else:                           # import everything from the grid

                        self.el_in[i][t] = self.cons[i][t]

            # Sell not-used electricity
            if electricity_pool > 0:
                self.el_out[p][t] = electricity_pool
            if prosumer_balance > 0:
                self.exported[p][t] = prosumer_balance - self.el_out[p][t]
       

    # only make sense when a single (aggregated) household is optimized
    def optimize(self, h=0,  solver=Solver):
        """
        previous name: optimize_standard_solver
        """
        if self.battery[h] == 0:
            return self.yearcost_nobat(h)
        
        self.solver = self.make_standard_solver(solver, h)
        self.months = len(self.price[h])/(24*365/12)
        self.totalcost = self.fixedfee*self.months
        
        self.totalcost += self.solver.solve(len(self.price[h]))

        # decision data is generated only for one single house
        self.el_in = [[self.solver.model.el_in[t].value for t in range(len(self.price[h]))]]
        self.el_out = [[self.solver.model.el_out[t].value for t in range(len(self.price[h]))]]
        self.bat = [[self.solver.model.bat[t].value for t in range(len(self.price[h]))]]
        
        return round(self.totalcost,2)
    

    def optimize_full_community(self, solverclass=None, solver=None):
        """
        previous name: optimize
        """
        
        if sum(self.battery) == 0 and len(self.cons) == 1: # ONLY for SINGLE households!
            return sum(self.yearcost_nobat(h) for h in range(len(self.cons)))

        if solver:
            self.solver = solver
        elif solverclass:
            self.general_solver(solverclass)
        else:
            self.general_solver()

        self.months = hours_to_months(len(self.price[0]))

        self.fixed_fee = len(self.pv)*self.fixedfee*self.months
        
        self.totalcost = self.fixed_fee + self.solver.solve(len(self.price[0]))

        self.el_in = [[round(self.solver.model.el_in[(h,t)].value,4)
                       for t in range(len(self.price[h]))] for h in range(len(self.cons))]
        self.el_out = [[round(self.solver.model.el_out[(h,t)].value,4)
                        for t in range(len(self.price[h]))] for h in range(len(self.cons))]
        self.bat = [[round(self.solver.model.bat[(h,t)].value,4)
                     for t in range(len(self.price[h]))] for h in range(len(self.cons))]
        self.imported = [[round(self.solver.model.imported[(h,t)].value,4)
                          for t in range(len(self.price[h]))] for h in range(len(self.cons))]
        self.exported = [[round(self.solver.model.exported[(h,t)].value,4)
                          for t in range(len(self.price[h]))] for h in range(len(self.cons))]

        return round(self.totalcost,2)

    def optimize_one(self, h):
        self.solver = self.original_solver_optimize(h)
        self.months = hours_to_months(len(self.price[h]))
        self.totalcost = self.fixedfee*self.months + round(self.solver.solve(),2)
        
        return round(self.totalcost,2)
    
    def yearcost_nobat(self, h=0, months=1):
        self.months = hours_to_months(len(self.price[h]))
        return round(self.bill_nobat(h) + self.fixedfee*months,2)

    def peak(self, h=0):
        return max(max(self.el_in[h][t],self.el_out[h][t]) for t in range(len(self.price[h])))

    def check(self, h=0): # only make sense for a single (aggregated) household
        return all((self.bat[h][t-1]-self.bat[h][t]+self.el_in[h][t]-self.el_out[h][t]
                    +self.sun[h][t]*self.pv[h]) == self.cons[h][t]
                   for t in range(1,len(price)))

    def peak_cost(self):
        daymonths = [31,28,31,30,31,30,31,31,30,31,30,31]
        # [extension] (TO-DO)

    def load_full_optimizer(binfile=join(DATA_FOLDER_2221,'opt2221_alr6_bdr15.bin')):
        return load_or_precompute(binfile,
                        lambda:OptimizerExt.generate_full_optimizer(OptimizerExt(),binfile))

    def load_default_optimizer(binfile=OPT_2221_BIN, input_folder=DATA_FOLDER_2221):
        if not exists(binfile):
            Optimizer.save_optimizer(input_folder, binfile)
        return loaddump(binfile)
        

### Generate PV and Battery sizes based on ALR/BDR from average values
class AlrBdr():
    def __init__(self,alr,bdr,avg_file=AVG_FILE_DEFAULT):
        self.alr, self.bdr = alr, bdr
        if avg_file:
            self.avg = load(avg_file)
            self._set_pv_battery()

    def _set_pv_battery(self):
        self.pv = [self.avg[h]*self.alr for h in range(len(self.avg))]
        self.battery = [self.avg[h]*self.bdr for h in range(len(self.avg))]

    def from_consumptions(alr,bdr,cons):
        alrbdr = AlrBdr(alr,bdr,None)
        alrbdr.avg = [sum(house)/len(house) for house in cons]
        alrbdr._set_pv_battery()
        return alrbdr
        
### Optimizer using ALR/BDR to generate PV and Battery sizes
class AlrBdrOptimizer(Optimizer):
    def __init__(self, alr, bdr, avg_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alrbdr = (AlrBdr(alr,bdr,avg_file) if avg_file
                        else AlrBdr.from_consumptions(alr,bdr,self.cons))

        self.pv = self.alrbdr.pv
        self.battery = self.alrbdr.battery

# Helper class that memoizes the costs associated with one optimizer
# Memoized costs are returned when available, otherwise an optimization is ran
class OptCosts(dict):
    def __init__(self, opt, *args, **kwargs ):
        self.opt = opt
        dict.__init__(self, *args, **kwargs )
        
    def __missing__(self, key):
        """
            Expects either an integer as a key (end-user ID to retrieve cost),
            or an iterable as a key (group of end-users to retrieve aggregated cost)
        """
        if type(key) is int:
            self[key] = self.opt.optimize(key)
        else:
            self[key] = self.opt.subset(list(key)).aggregate_all().optimize()
        return self[key]

    # sum the costs of a group of users
    def base_cost(self, group):
        return sum(self[user] for user in group)

    # "forget" the optimizer, useful to serialize only the costs
    def to_dict(self):
        return dict(self)

