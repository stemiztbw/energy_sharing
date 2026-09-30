from xml.parsers.expat import model

from pyomo.environ import *
from pyomo.opt import SolverFactory

"""
Solver classes.
Wrapper to instantiate the pyomo object.
Different solvers for the different studied models.
"""

"""
Base Solver Class: "Single Household" model (so-called "Aggregation model")
Assumes no tranmission losses, battery losses, etc.
"""
class Solver:
    def __init__(self, pv, maxbat, cons, sun, price, predictor, tax, el_tax, el_net, solver_name):
        self.pv = pv
        self.maxbat = maxbat
        self.sun = sun
        self.price = price
        self.cons = cons
        self.tax = tax
        self.el_tax = el_tax
        self.el_net = el_net
        self.predictor = predictor
        self.solver_name = solver_name

    ### main cost functions ###
        
    def cost(self, price_t, elin, elout):
        return elin * (price_t*self.tax + self.el_tax) - elout * (price_t + self.el_net)

    def _cost_model(self, t):
        price_t = self.predictor(self.price, t, self.i, self.j)
        return self.cost(price_t, self.model.el_in[t], self.model.el_out[t])

    ### default initializations ###
        
    """ predictions
    sun(t)    projected sun profile at starting time
    price(t)  projected electricity prices at starting time
    cons(t)   projected consumption profile for h at starting time
    """
    def _init_predictors(self, i, j):
        self.i = i
        self.j = j

    def _init_model(self, i, j):
        self._init_predictors(i,j)
        self.model = ConcreteModel()
        self.model.T = RangeSet(i,j)
        self.model.t0 = i

    def _init_optimization_variables(self):
        self.model.bat = Var(self.model.T, domain=NonNegativeReals, bounds=(0,self.maxbat))
        self.model.el_in = Var(self.model.T, domain=NonNegativeReals)
        self.model.el_out = Var(self.model.T, domain=NonNegativeReals)

    def _init_objective_function(self):
        self.model.OBJ = Objective(expr = sum(self._cost_model(t) for t in self.model.T))

    def _add_constraints(self):
        self.model.batlevel_cstr = Constraint(self.model.T, rule=self.battery_level)

    def _solve(self):
        SolverFactory(self.solver_name).solve(self.model)

    ### default constraints ###

    def battery_level(self, model, t):
        sun_t = self.predictor(self.sun, t, self.i, self.j)
        cons_t = self.predictor(self.cons, t, self.i, self.j)
        
        balance = sun_t*self.pv - cons_t + model.el_in[t] - model.el_out[t]
        if t == model.t0:
            return model.bat[t] == self.bat0 + balance
        return model.bat[t] == model.bat[t-1] + balance

    ### Main Optimizer function (Pyomo Model -- Forecast Optimizer) ###
    
    def forecast_optimize(self, i, j, bat0):
        """
        solve optimally the linear program over period [i,j] for 1 household
        i         starting time
        j         ending time
        bat0      battery level at starting time
        """ 
        self.bat0 = bat0
        self._init_model(i,j)
        self._init_optimization_variables()
        self._init_objective_function()
        self._add_constraints()
        self._solve()

    ### this is used to correct the decision at time t based on predictions ###

    def _cap_battery(self, model, i):
        el_balance = model.el_in[i].value - model.el_out[i].value
        local_balance = self.sun[i]*self.pv - self.cons[i]
        newbat = self.batlevel[i] + local_balance + el_balance
        return max(min(newbat,self.maxbat),0)

    def _elec_cost(self, i):
        el_balance = self.batlevel[i+1] - self.batlevel[i] - self.sun[i]*self.pv + self.cons[i]
        el_in = el_balance if el_balance > 0 else 0
        el_out = -el_balance if el_balance < 0 else 0
        return self.cost(self.price[i], el_in, el_out)

    ## Strategy 1: decision is followed about interaction with the grid ##
    def _iterative_solve1(self, i, windowsize):
        self.forecast_optimize(i, i+windowsize, self.batlevel[i])
        self.batlevel.append(self._cap_battery(self.model, i))  # Strategy 1
        return self._elec_cost(i)

    ## Strategy 2: target battery level is followed first ##
    def _iterative_solve(self, i, windowsize):
        self.forecast_optimize(i, i+windowsize, self.batlevel[i])
        self.batlevel.append(self.model.bat[i].value)           # Strategy 2
        return self._elec_cost(i)
    
    ### Iterative solver with window size of n ###
    """
    This should not be default, subclass with "solve_sliding_window()" method.
    Actually this is "research" code, a real system will solve the entire given period.
    (the data used can be forecast but a single period is solved at a time)
    """
    
    def solve(self, windowsize=None):
        """
        performs successive optimizations on truncated chunks of length windowsize
        windowsize      sliding window length (window slide = 1), None = use all data.
        """
        N = min(len(self.cons),len(self.sun),len(self.price))
        windowsize = windowsize if windowsize is not None else N-1
        self.batlevel = [0]
        total_bill = 0

        for i in range(0,N-windowsize-1):
            total_bill += self._iterative_solve(i, windowsize)

        t = N-windowsize-1
        model = self.forecast_optimize(t, N-1, self.batlevel[t])

        for i in range(t,N):
            self.batlevel.append(self.model.bat[i].value)
            total_bill += self._elec_cost(i)

        return total_bill

"""
Warning: fix code with serialization problem with lambda functions!
"""

# to reproduce the old behavior on average predictor in the e-energy '20 work.
class SolverNoPredictionOnCurrentHour(Solver):
    def _init_predictors(self, i, j):
        self.sunf = lambda t : self.predictor(self.sun, t, i, j)
        self.pricef = lambda t : self.predictor(self.price, t, i, j)
        self.consf = lambda t : self.predictor(self.cons, t, i, j)
        
    # sun and cons are considered included in bat0 -- price is still forecasted
    def battery_level(self, model, t):
        el = model.el_in[t] - model.el_out[t]
        if t == model.t0:
            return model.bat[t] == self.bat0 + el
        return model.bat[t] == model.bat[t-1] + self.sunf(t)*self.pv - self.consf(t) + el

    # add true values for sun and cons within "bat0"
    def _iterative_solve(self, i, wsize):
        self.forecast_optimize(i, i+wsize, self.batlevel[i] + self.sun[i]*self.pv - self.cons[i])
        self.batlevel.append(self.model.bat[i].value)           
        return self._elec_cost(i)

# use also current price in the cost function
class SolverNoPredictionAtAllCurrentHour(SolverNoPredictionOnCurrentHour):
    def _cost_model(self, t):
        if t == self.model.t0:
            return self.cost(self.price[t], self.model.el_in[t], self.model.el_out[t])
        return self.cost(self.pricef(t), self.model.el_in[t], self.model.el_out[t])
    
### Adding battery losses, which requires 2 extra optimization variables.
class SolverBatteryLosses(Solver):
    def __init__(self, charging_losses, discharging_losses=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chg_loss = charging_losses
        self.dischg_loss = discharging_losses

    def _init_optimization_variables(self):
        super()._init_optimization_variables()
        self.model.charge = Var(self.model.T, domain=NonNegativeReals)
        self.model.discharge = Var(self.model.T, domain=NonNegativeReals)

    def battery_level(self, model, t):
        balance = ((1-self.chg_loss)*model.charge[t]-(1+self.dischg_loss)*model.discharge[t])
        if t == model.t0:  
            return model.bat[t] == self.bat0 + balance
        return model.bat[t] == model.bat[t-1] + balance

    def demand(self, model, t):
        grid_balance = model.el_in[t] - model.el_out[t]
        bat_balance = model.discharge[t] - model.charge[t]
        return self.consf(t) == self.sunf(t)*self.pv + grid_balance + bat_balance 

    def _add_constraints(self):
        self.model.batlevel = Constraint(self.model.T, rule=self.battery_level)
        self.model.demand = Constraint(self.model.T, rule=self.demand)

### Transmission losses: the full community needs to get optimized synchronously here
class SolverTransmissionLosses(Solver):
    def __init__(self, losses, grid_fee, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transmission_losses = losses
        self.grid_fee = grid_fee

    def _init_predictors(self, i, j):
        self.sunf = lambda h, t : self.predictor(self.sun[h], t, i, j)
        self.pricef = lambda t : self.predictor(self.price, t, i, j)
        self.consf = lambda h, t : self.predictor(self.cons[h], t, i, j)

    def _init_model(self, i, j):
        super()._init_model(i,j)
        self.model.H = RangeSet(0,len(self.pv)-1)

    def _init_optimization_variables(self):
        self.model.bat = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.el_in = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.el_out = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.imported = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.exported = Var(self.model.H, self.model.T, domain=NonNegativeReals)

    def _cost_model(self, h, t):
        return self.cost(self.pricef(t), self.model.el_in[(h,t)], self.model.el_out[(h,t)])

    def _init_objective_function(self):
        self.model.OBJ = Objective(expr = sum(self._cost_model(h, t)
                        + self.model.exported[(h,t)] * self.grid_fee
                                              for h in self.model.H for t in self.model.T))

    def battery_limit(self, model, h, t):
        return model.bat[(h,t)] <= self.maxbat[h]

    def battery_level(self, bat0, i):
        def rule(model, h, t):
            el1 = model.el_in[(h,t)] - model.el_out[(h,t)]
            el2 = model.imported[(h,t)] - model.exported[(h,t)]
            if t == model.T[1]:
                return model.bat[(h,t)] == self.bat0[h] + el1 + el2
            balance = self.sunf(h,t)*self.pv[h] - self.consf(h,t) + el1 + el2
            return model.bat[(h,t)] == model.bat[(h,t-1)] + balance
        return rule

    def elec_balance(self, model, t):
        total_exported = sum(model.exported[(h,t)] for h in model.H)
        total_imported = sum(model.imported[(h,t)] for h in model.H)
        return (1-self.transmission_losses)*total_exported == total_imported

    def _add_constraints(self):
        self.model.batlimit = Constraint(self.model.H, self.model.T, rule=self.battery_limit)
        batlevel_constraint = self.battery_level(self.bat0, self.model.T[1])
        self.model.batlevel = Constraint(self.model.H, self.model.T, rule=batlevel_constraint)
        self.model.elbalance = Constraint(self.model.T, rule=self.elec_balance)

    ### Iterative solver with window size of n -- Maintain |H| battery levels ###
    ### Initial battery level constraint: bat0 is provided with real information ###
    def solve(self, windowsize=None):
        N = len(self.price)
        windowsize = windowsize if windowsize is not None else N-1
        
        def el_balance(h,t):
            return self.sun[h][t]*self.pv[h] - self.cons[h][t]

        def bill(model, t):
            return sum(self.cost(self.price[t], model.el_in[(h,t)].value, model.el_out[(h,t)].value)
                           for h in range(len(self.maxbat)))
        
        bat = [el_balance(h,0) for h in range(len(self.maxbat))]
        total_bill = 0

        for i in range(0,N-windowsize-1):
            self.forecast_optimize(i, i+windowsize, bat)
            bat = [self.model.bat[(h,i)].value + el_balance(h,i+1) for h in range(len(self.maxbat))]
            total_bill += bill(self.model, i)

        self.forecast_optimize(N-windowsize-1, N-1, bat)

        return total_bill + sum(bill(self.model, k) for k in range(N-windowsize-1,N))

"""
General LP-Solver: include all above sub-models.
Most general and slower solver.
"""
class SolverGeneralModel(Solver):
    def __init__(self, charge_loss=0, discharge_loss=0, selfdischarge=1,
                 syst_losses = 0, trans_losses=0, grid_fee=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batcharge_losses = charge_loss
        self.batdischarge_losses = discharge_loss
        self.selfdischarge = selfdischarge
        self.syslos = syst_losses
        self.translosses = trans_losses
        self.grid_fee = grid_fee

    # household's dependent price
    def _init_predictors(self, i, j):
        self.sunf = lambda h, t : self.predictor(self.sun[h], t, i, j)
        self.pricef = lambda h, t : self.predictor(self.price[h], t, i, j)
        self.consf = lambda h, t : self.predictor(self.cons[h], t, i, j)

    def new_cost(self, price_t, el_in, el_out, el_imp, el_exp):
        pbuy = (price_t*self.tax + self.el_tax)
        psell = (price_t + self.el_net)
        pcom = (pbuy+psell)/2
        pgrid = self.tax*self.grid_fee / 2
        return el_in*pbuy - el_out*psell + el_imp*(pcom+pgrid) - el_exp*(pcom-pgrid)

    def _cost_model(self, h, t):
        return self.new_cost(self.pricef(h,t), self.model.el_in[(h,t)], self.model.el_out[(h,t)],
                             self.model.imported[(h,t)], self.model.exported[(h,t)])

    def local_cost(self, model, h, t):
        return self.new_cost(self.pricef(h,t), model.el_in[(h,t)].value, model.el_out[(h,t)].value,
                             model.imported[(h,t)].value, model.exported[(h,t)].value)

    def community_cost(self, model, t):
        return sum(self.local_cost(model, h, t) for h in model.H)

    def battery_limit(self, model, h, t):
        return model.bat[(h,t)] <= self.maxbat[h]
        
    def battery_level(self, bat0, i):
        def rule(model, h, t):
            el_charged = (1-self.batcharge_losses)*model.charge[(h,t)]
            el_discharged = (1+self.batdischarge_losses)*model.discharge[(h,t)]
            bat_balance = el_charged - el_discharged
            if t == i:
                return model.bat[(h,t)] == self.selfdischarge*bat0[h] + bat_balance
            return model.bat[(h,t)] == self.selfdischarge*model.bat[(h,t-1)] + bat_balance
        return rule

    def demand(self, model, h, t):
        grid_balance = (1-self.syslos)*model.el_in[(h,t)] - (1+self.syslos)*model.el_out[(h,t)]
        el_balance = grid_balance + model.imported[(h,t)] - model.exported[(h,t)]
        battery_balance = model.discharge[(h,t)] - model.charge[(h,t)]
        return self.consf(h,t) == self.sunf(h,t)*self.pv[h] + el_balance + battery_balance

    def elec_correspondance(self, model, h, t):
        total_exported = sum(model.exported[(h,t)] for h in model.H)
        total_imported = sum(model.imported[(h,t)] for h in model.H)
        return (1-self.translosses)*total_exported == total_imported

    ### Single solver with window size of n -- Maintain |H| battery levels ###
    def solve(self, N):
        # Init
        bat0 = [0]*len(self.maxbat)
        i, j = 0, N-1
        self._init_predictors(i,j)

        # Build model
        self.model = ConcreteModel()
        self.model.T = RangeSet(i,j)
        self.model.H = RangeSet(0,len(self.pv)-1)
        
        ### variables: battery level, dis/charge, el bought, sold, locally imported and exported.
        self.model.bat = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.charge = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.discharge = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        
        self.model.el_in = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.el_out = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.imported = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.exported = Var(self.model.H, self.model.T, domain=NonNegativeReals)

        self.model.OBJ = Objective(expr = sum(self._cost_model(h, t)
                                              for h in self.model.H for t in self.model.T))
        
        self.model.batlimit = Constraint(self.model.H, self.model.T, rule=self.battery_limit)
        self.model.batlevel = Constraint(self.model.H, self.model.T, rule=self.battery_level(bat0, i))
        self.model.demand = Constraint(self.model.H, self.model.T, rule=self.demand)
        self.model.electrans = Constraint(self.model.H, self.model.T, rule=self.elec_correspondance)

        
        # Solve model
        SolverFactory(self.solver_name).solve(self.model)

        # Return cost
        return sum(self.community_cost(self.model, t) for t in range(N))

"""
ST 30.09.26
Static Sharing Solver (Aufteilungsschlüssel): introduction of allocation keys that defines sharing volume..
"""
class SolverStaticSharing(SolverGeneralModel):
    def __init__(self, allocation_keys, prosumer, residual_price=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if abs(sum(allocation_keys.values()) - 1.0) > 1e-9:
            raise ValueError("Allocation keys must sum to 1.")

        if prosumer in allocation_keys:
            raise ValueError(
                "Prosumer must not be included in allocation_keys."
            )

        if any(alpha < 0 for alpha in allocation_keys.values()):
            raise ValueError(
                "Allocation keys must be non-negative."
            )

        self.allocation_keys = allocation_keys
        self.prosumer = prosumer
        self.residual_price = residual_price

    def sharing_allocation_limit(self, model, c, t):  # E_sharing <= alpha * E_PVüberschuss

        if c not in self.allocation_keys:
            return Constraint.Skip

        alpha = self.allocation_keys[c]

    def pv_available(self, t):
        pv_generation = (
            self.sunf(self.prosumer, t)
            * self.pv[self.prosumer]
        )

        return max(
            pv_generation - self.consf(self.prosumer, t),
            0
        )

    def sharing_demand_limit(self, model, c, t):  # E_sharing <= E_cons
        if c not in self.allocation_keys:
            return Constraint.Skip

        return model.imported[c, t] <= self.consf(c, t)

    def only_prosumer_exports(self, model, h, t):  # nur der Prosumer darf Community-Energie exportieren.

        if h == self.prosumer:
            return Constraint.Skip

        return model.exported[h, t] == 0

    def prosumer_no_import(self, model, t):  # Consumer sollen nur importieren und nicht exportieren.
        return model.imported[self.prosumer, t] == 0

    def only_allocated_consumers_import(self, model, h, t):  # Haushalte, die keine allocation_key zugeteilt bekommen haben (weil sie kein ES machen wollen), können keine Sharing-Energie importieren. 

        if h in self.allocation_keys:
            return Constraint.Skip

        return model.imported[h, t] == 0

    def solve(self, N):
        # Init
        bat0 = [0]*len(self.maxbat)
        i, j = 0, N-1
        self._init_predictors(i,j)

        # Build model
        self.model = ConcreteModel()
        self.model.T = RangeSet(i,j)
        self.model.H = RangeSet(0,len(self.pv)-1)

        ### variables: battery level, dis/charge, el bought, sold, locally imported and exported.
        self.model.bat = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.charge = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.discharge = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        
        self.model.el_in = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.el_out = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.imported = Var(self.model.H, self.model.T, domain=NonNegativeReals)
        self.model.exported = Var(self.model.H, self.model.T, domain=NonNegativeReals)
    
        self.model.OBJ = Objective(expr = sum(self._cost_model(h, t)
                                            for h in self.model.H for t in self.model.T))
            
        self.model.batlimit = Constraint(self.model.H, self.model.T, rule=self.battery_limit)
        self.model.batlevel = Constraint(self.model.H, self.model.T, rule=self.battery_level(bat0, i))
        self.model.demand = Constraint(self.model.H, self.model.T, rule=self.demand)
        self.model.electrans = Constraint(self.model.H, self.model.T, rule=self.elec_correspondance)
        self.model.sharingAllocation = Constraint(self.model.H, self.model.T, rule=self.sharing_allocation_limit)
        self.model.sharingDemand = Constraint(self.model.H, self.model.T, rule=self.sharing_demand_limit)
        self.model.onlyProsumerExports = Constraint(self.model.H, self.model.T, rule=self.only_prosumer_exports)
        self.model.prosumerNoImport = Constraint(self.model.T, rule=self.prosumer_no_import)
        self.model.onlyAllocatedConsumersImport = Constraint(self.model.H, self.model.T, rule=self.only_allocated_consumers_import)
        
        # Solve model
        SolverFactory(self.solver_name).solve(self.model)

        # Return cost
        return sum(self.community_cost(self.model, t) for t in range(N))


"""
Peak-Solver: using a fake cost function to push for peak decrease
Uses a 'Non-Linear Solver' to find the solution (slow)
"""
class PeakSolver(Solver):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.solver_name = 'ipopt' # Non-Linear Solver
        self.alpha = 0.05

    def fake_cost(self, price_t, elin, elout):
        return (elin + self.alpha*(1+elin) * (1+elin)) * (price_t*self.tax + self.el_tax) - elout * (price_t + self.el_net)

    def _cost_model(self, t):
        return self.fake_cost(self.price[t], self.model.el_in[t], self.model.el_out[t])

    @classmethod
    def make(cls, alpha_value):
        class PeekSolverA(cls):
            def __init__(self, *kargs, **kwargs):
                super().__init__(*kargs, **kwargs)
                self.alpha = alpha_value
        return PeekSolverA

