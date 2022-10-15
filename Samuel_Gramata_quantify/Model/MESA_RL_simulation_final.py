#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 21:32:04 2022

@author: samuelgramata
"""

from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector

from scipy.stats import powerlaw, lognorm
import numpy as np
import pandas as pd
import powerlaw as pr
from numpy.random import normal
import random
from operator import attrgetter
from ddpg_11 import Agent, ReplayBuffer, OUActionNoise, CriticNetwork, ActorNetwork

import torch

#Initialize the reinforcement learning model 
agent = Agent(alpha = 0.001, beta = 0.001, input_dims = [12], tau = 0.001, env = 1, batch_size = 64, layer1_size = 400, layer2_size = 300, n_actions=1)
np.random.seed(0)
agent.load_models()

##VC Coefficients
#VC Coefficients - General
Number_of_VCs = 100 #approximate number of VCs in USA. In fact, it is 1000, but for computaional reasons we devide everything by 10
Fund_maturity = 40 #the number of time steps until the outcome of the fund is realised, 1 step=3months
VC_quality_alpha = 2.06 #alpha coefficient for power law distribution
VC_quality_x_min = 0 #x_min coefficient for power law distribution
VC_max_TVPI = 14 #we want to normalize VC_quality so that it lies between 0 and 1
Average_portfolio_size = 32 #Based on real world data

#VC Coefficients - Employees
Number_of_employees_sd = 1.3711 #standard deviation coefficinet for lognormal distribution of number of employees
Number_of_employees_loc = 0.8426 #loc coefficient for lognormal distribution of number of employees
Number_of_employees_scale = 9.5626 #scale coefficient for lognormal distribution of number of employees
VC_work_hours_per_week = 56 #Average numebr of hours worked by an analyst in VC
Work_weeks_per_month = 4 
Work_hours_per_month = VC_work_hours_per_week*Work_weeks_per_month #Work hours per months per employee in VC
Work_hours_per_3months = Work_hours_per_month*3 #1 time step = 3 months, thus we are interested in hours per 3 months
Percentage_of_time_spend_on_other_activities = 0.31 #time spend by VC employee on activitties not related to either screening or advising
Time_for_screening_and_monitroring_3months_per_emp = Work_hours_per_3months*(1-Percentage_of_time_spend_on_other_activities)
Number_of_funds_at_operation = 2 #at same time, VC takes care of multiple funds
Time_for_screening_and_monitroring_3months_per_emp_per_fund = Time_for_screening_and_monitroring_3months_per_emp/Number_of_funds_at_operation
Average_number_of_investemnt_analysts = 19 #Based on real-world data 

#VC Coefficients - Time needed
Screening_time = 60 #Time in hours needed to screen a startup
Advising_time = 27.5 #Time in hours needed per time step(i.e. 3 months) to advise to a startup in the portfolio 

#VC Coefficients - Returns
Early_returns_alpha = 2.3758 #alpha coefficient for power law distribution of early stage retruns
Early_returns_x_min = 4.5761 #X_min coefficeint for power law distribution of early stage returns
Early_startup_exit = 32 #number of time steps it takes early startup to exit
Late_returns_lognormal_sd = 0.98981 #standard deviation for log normal distribution of late stage returns
Late_returns_lognormal_loc = -0.133236 #location coefficient for log normal distribution of late stage returns
Late_returns_lognormal_scale = 1.79759 #scale coefficinet for log normal distribution of late stage returns
Late_startup_exit = 24 #number of time steps it takes a late stage startup to exit

##Startup Coefficients
#Startup Coefficients - General
Number_of_new_startups = 25600 #Number of business starts in USA every 3 months, In fact, it is 256000, but for computaional reasons we devide everything by 10
Number_of_late_stage_startups = 36 #It is in fact 355, but we divide everything by 10 for computational reasons
EPI_alpha = 0.29872 #alpha coefficient for power law distribution of EPI which is taken as a proxy for startup potential
EPI_loc = 0 #location coefficient for power law distribution of EPI
EPI_scale = 1 #scale coefficient for power law distribtuion of EPI
Percentage_of_startups_getting_to_later_stage = 0.2979 #Percentage of startups which recieve funding in 1st and 2nd round 

#Startup Coefficients - Time progression equation
#EPI = Alpha*EPI + Beat*VC + Industry_Shock + Macro_Shock + Idiosyncratic_shock
Alpha = 0.99 #alpha coefficient for time progression equation. Expresses weight of EPI
Beta = 0.01 #beta coefficient for time progression equation. Expresses the weight of VC 
Macro_shock_mean = 0 #mean for normal distribution of macro shock
Macro_shock_sd = 0.0681 #standard deviation for normal distribution of macro shock
Industry_shock_mean = 0 #mean for normal distribution of industry shock
Industry_shock_sd = 0.0432 #standard deviatio for normal distribution of industry shock
Idiosyncratic_shock_mean = 0 #mean for normal distribution for idiosyncratic shock
Idiosyncratic_shock_sd = 0.0775 #standard deviation for normal distribtion for idiosyncratic shock

#Startup Coefficeints - Industries
List_of_Industries = ["Business Productivity Software", "Drug Discovery", "Financial Software", "Media and Information Services", "Network Management Software", "Biotechnology", "Application Software", "Therapeutic Devices", "Surgical Devices", "Other Healthcare Tech Systems"]
Probability_Distribution_of_Industries = [0.3486, 0.1133, 0.10841, 0.1006, 0.0844, 0.0729, 0.0556, 0.0440, 0.0365, 0.0357]
Correlation_matrix = pd.read_excel("Correlation_matrix.xlsx")
Correlation_matrix.index = Correlation_matrix["Unnamed: 0"]
del Correlation_matrix["Unnamed: 0"]

#Startup Coefficients - Screening
Noise_mean_before_screening_ES = 0 #mean for normal distribution of noise before screening for early stage startups
Noise_standard_deviation_before_screening_ES = 0.75 #standard deviation for normal distribution of noise before screening for early stage startups
Noise_mean_after_screening_ES = 0 #mean for normal distribution of noise after screening for early stage startups
Noise_standard_deviation_after_screening_ES = 0.2 #standard deviation for normal distribution of noise after screening for early stage startups
Noise_mean_before_screnniing_LS = 0 #mean for normal distribution of noise before screening for late stage startups
Noise_standard_deviation_before_screening_LS = 0.53 #standard deviation for normal distribution of noise before screening for late stage startups
Noise_mean_after_screening_LS = 0 #mean for normal distribution of noise after screening for late stage startups
Noise_mean_before_screnniing_LS = 0.14 #standard deviation for normal distribution of noise after screening for late stage startups

#Startup Coefficients - Investors
Number_of_investors_per_round = 5 #max number of investors allowed to invest in startup
Number_of_due_diligence_investors = 10 #number of investors enagaged in due diligence

##A sample of early stage returns, used for potential -> return mapping
#powerlaw package does not have an inverse cdf function, hence we apporximate it with a sample
sample_size = 10000
theoretical_distribution_ER = pr.Power_Law(x_min = Early_returns_x_min, parameters = [Early_returns_alpha])
simulated_data_ER = theoretical_distribution_ER.generate_random(sample_size)
simulated_data_new_ER = []
for i in simulated_data_ER: #simulated data start with 1, but rerturns start with 0, hence we translate the smaple by -1
    i = i-1
    simulated_data_new_ER.append(i)
sample_data_ER = sorted(simulated_data_new_ER)

##A distribution for VC_quality
theoretical_distribution_VC = pr.Power_Law(x_min = VC_quality_x_min, parameters = [VC_quality_alpha])

##General model coefficents
Risk_free_rate = 1.521 #10 year return on us treasury bill
Estimate_of_early_stage_screenings = int((Number_of_VCs * Average_number_of_investemnt_analysts * (Time_for_screening_and_monitroring_3months_per_emp_per_fund/Screening_time)*(1-Percentage_of_startups_getting_to_later_stage))/(Number_of_due_diligence_investors))

##Here we define class for VC, Startup and Activation
#VC is assigend a unique id, VC quality and the number of investment analysts
class VC(Agent):
    def __init__(self, unique_id, VC_quality, Investment_analysts, Fund_life_stage, model):
        self.unique_id = unique_id
        self.model = model
        self.VC_quality = VC_quality
        self.Fund_life_stage = Fund_life_stage
        self.Investment_analysts = Investment_analysts
        self.Endowement = 1
        self.Screening_prospects = []
        self.Portfolio = []
        self.Portfolio_size = len(self.Portfolio)
        self.Effort_left_for_fund = self.Investment_analysts*Time_for_screening_and_monitroring_3months_per_emp_per_fund
        self.Effort_allocated_to_startups_in_portfolio = self.Portfolio_size*Advising_time
        self.Effort_left_for_screening = self.Effort_left_for_fund - self.Effort_allocated_to_startups_in_portfolio
        self.Number_of_available_screenings = self.Effort_left_for_screening/Screening_time 
        self.Time_till_end_of_investment_stage_E = max(0, (Fund_maturity - Early_startup_exit - self.Fund_life_stage))
        self.Time_till_end_of_investment_stage_L = max(0, (Fund_maturity - Late_startup_exit - self.Fund_life_stage))
        
    #This funciton enables us to map EPI (startup potnetial) into returns
    def EPI_to_returns(self, EPI, stage):
        #This gives us probability of observing EPI less or equal to observed vlaue
        probability = powerlaw.cdf(EPI, EPI_alpha, EPI_loc, EPI_scale)
        #If VC has invested in the startup in early stage then:
        if stage == "Early":
            return float(sample_data_ER[int(sample_size*probability)])
        #If VC has invested in the startup in later stage then:
        if stage == "Late":
            return float(lognorm.ppf(probability, Late_returns_lognormal_sd, Late_returns_lognormal_loc, Late_returns_lognormal_scale))        
    
    #Projects EPI fo startups into the future
    def time_progression_projected(self, EPI):
        EPI_ = Alpha*EPI + Beta*self.VC_quality + np.random.normal(Idiosyncratic_shock_mean, Idiosyncratic_shock_sd)\
        + np.random.normal(Industry_shock_mean, Industry_shock_sd) + np.random.normal(Macro_shock_mean, Macro_shock_sd)
        return EPI_

    #Calculates expecetd covariance between two startups
    #Since EPI_final boils down to EPI_final ≈ Alpha^32*EPI_0 + (Alpha^31 + Alpha^30 + ...+ 1) *Beta*Advising
    #+(Alpha^31 + Alpha^30 + ...+ 1) * Average_Idiosyncratic_shock + (Alpha^31 + Alpha^30 + ...+ 1) *Averga_Industry_shock +(Alpha^31 + Alpha^30 + ...+ 1) *Averga_Macro_shock
    # the Cov(Startup_1, Startup_2) = 
    def expected_covariance(self, startup_1, startup_2):
        time_together = Early_startup_exit - max(getattr(startup_1, "Life_stage"), getattr(startup_2, "Life_stage"))
        alpha_sumation_1 = ((1 - Alpha**time_together)/(1-Alpha))
        alpha_sumation_2 = ((1 - Alpha**time_together)/(1-Alpha))
        Industry_correlation =  float(Correlation_matrix.loc[getattr(startup_1, "Industry")[0], getattr(startup_2, "Industry")[0]]) 
        return float(alpha_sumation_1*alpha_sumation_2*Industry_correlation*Industry_shock_sd**2 + alpha_sumation_1*alpha_sumation_2*Macro_shock_sd**2)
    
    #Calculate variance for one startup
    def expected_variance(self, startup_1):
        alpha_sumation_1 = ((1 - Alpha**(Early_startup_exit - getattr(startup_1, "Life_stage")))/(1-Alpha))
        alpha_sumation_2 = ((1 - Alpha**(1 + Early_startup_exit - getattr(startup_1, "Life_stage")))/(1-Alpha))                   
        if getattr(startup_1, "Life_stage") == 0:
            noise = Noise_standard_deviation_after_screening_ES
        else:
            noise = Noise_standard_deviation_after_screening_ES/(getattr(startup_1, "Life_stage")**(1/2))
        return alpha_sumation_1**2*noise**2 + alpha_sumation_2**2*Idiosyncratic_shock_sd**2 + alpha_sumation_2**2*Industry_shock_sd**2 + alpha_sumation_2**2*Macro_shock_sd**2                                                                                        

    #Calculate expected variance for the whole portfolio 
    def expected_portfolio_variance(self, Portfolio):
        if len(Portfolio) == 1:
            return self.expected_variance(Portfolio[0][0])
        else:
            Total_variance = 0
            for i in Portfolio:
                for j in Portfolio:
                    if i != j:
                        Total_variance = (i[2]*j[2]*self.expected_covariance(i[0], j[0])) + Total_variance
                    if i == j:
                        Total_variance = (i[2]**2*self.expected_variance(i[0])) + Total_variance
            return Total_variance
                                    
    #Gets expected return on a Portfolio
    def expected_return(self, Portfolio):
        Return = 0
        if len(Portfolio) == 0:
            return 0
        else:
            for i in Portfolio:
                Projected_EPI = getattr(i[0], "EPI_after_screening")
                for j in range(0,(Early_startup_exit-getattr(i[0],"Life_stage"))):                    
                    Projected_EPI = self.time_progression_projected(Projected_EPI)
                    if Projected_EPI < 0:
                        Projected_EPI = 0
                    if Projected_EPI > 1:
                        Projected_EPI = 0.99
                Return = float((self.EPI_to_returns(Projected_EPI, i[1])*i[2]))+ Return
            return Return
    
    def expected_return_without_projection(self, Portfolio):
        Return = 0
        if len(Portfolio) == 0:
            return 0
        else:
            for i in Portfolio:
                Projected_EPI = getattr(i[0], "EPI_after_screening")
                Return = float((self.EPI_to_returns(Projected_EPI, i[1])*i[2]))+ Return
            return Return
        
    def final_return(self, Portfolio):
        Return = 0
        for i in Portfolio:
            Return = float((self.EPI_to_returns(getattr(i[0], "EPI"), i[1])*i[2]))+ Return
        return Return + self.Endowement
            
            
    #Expected Sharpe ratio
    def expected_portfolio_coefficient(self, Portfolio):  
        if len(Portfolio) == 0:
            return 0
        else:
            return float(self.expected_return_without_projection(Portfolio) - Risk_free_rate)/float(self.expected_portfolio_variance(Portfolio)**(1/2))
    
    #Is a startup in the portfolio or not
    def startup_in_portfolio(self, Prospect):
        for i in self.Portfolio:
            if Prospect[0] in i:
                return 1
            else:
                return 0
        
    #Gets reward after taking a particular action a 
    def get_reward(self, action, startup):
        #If less than 0.01 was invested, we assume that VC does not invest into a given startup
        if self.Fund_life_stage <= (Fund_maturity - Early_startup_exit):   
            if action < 0:
                return torch.tensor([-100*(-action[0])])
            if 0<action <0.005:
                return torch.tensor([0])
        #If action in this range then VC invests in startup
            if 0.005<=action<=1 and action <= self.Endowement:
                return torch.tensor([(self.expected_portfolio_coefficient((self.Portfolio + [list(startup) + list(action)])) - self.expected_portfolio_coefficient(self.Portfolio))])
            if 0.005<=action<=1 and action > self.Endowement:
                return torch.tensor([-100*(action[0]-self.Endowement)])
            if action>1:
                return torch.tensor([-100*action[0]])
        if (Fund_maturity - Early_startup_exit)<= self.Fund_life_stage <= (Fund_maturity-Late_startup_exit):
            if getattr(startup[0], "Life_stage")<8:
                return torch.tensor([-10])
            else:
                if action < 0:
                    return torch.tensor([-100*(-action[0])])
                if 0<action <0.005:
                    return torch.tensor([0])
        #If action in this range then VC invests in startup
                if 0.005<=action<=1 and action <= self.Endowement:
                    return torch.tensor([(self.expected_portfolio_coefficient((self.Portfolio + [list(startup) + list(action)])) - self.expected_portfolio_coefficient(self.Portfolio))])
                if 0.005<=action<=1 and action > self.Endowement:
                    return torch.tensor([-100*(action[0]-self.Endowement)])
                if action>1:
                    return torch.tensor([-100*action[0]])
        if (Fund_maturity-Late_startup_exit)< self.Fund_life_stage:
            return torch.tensor([-10])             
        
    #Gets state which is inputed into the RL model
    def get_state(self, Prospect): 
        ##Prospect attributes
        #Attribtue 1 
        Prospect_EPI = getattr(Prospect[0], "EPI_after_screening")
        #Attribute 2 
        Prospect_stage = 0 
        if Prospect[1] == "Late":
            Prospect_stage = 1
        #Attribute 3
        Industry_correlation = 0
        for i in self.Portfolio:    
            Industry_correlation =  float(Correlation_matrix.loc[getattr(Prospect[0], "Industry")[0], getattr(i[0], "Industry")[0]]) + Industry_correlation
        Average_correlation_with_portfolio = 0
        if self.Portfolio_size != 0:
            Average_correlation_with_portfolio = Industry_correlation/self.Portfolio_size
        
        ##Cohort attributes
        #Attibute 4 
        total_cohort = 0
        for i in self.Screening_prospects:
            total_cohort = getattr(i[0], "EPI_after_screening") + total_cohort
        Screenings_mean = total_cohort/len(self.Screening_prospects) 
        #Attribute 5
        Screenings = []
        for i in self.Screening_prospects:
            Screenings.append(getattr(i[0], "EPI_after_screening"))
        Screenings_sd = np.std(Screenings)
            
        ##Portfolio attributes
        #Attribute 6
        total = 0
        for i in self.Portfolio:
            total = getattr(i[0], "EPI_after_screening") + total
        Portfolio_mean = 0
        if self.Portfolio_size != 0:
            Portfolio_mean = total/self.Portfolio_size
        #Attribute 7
        EPIs = []
        Portfolio_sd = 0
        if self.Portfolio_size != 0:
            for i in self.Portfolio:
                EPIs.append(getattr(i[0], "EPI_after_screening"))
            Portfolio_sd = np.std(EPIs)
            
        #VC attributes
        #Attribute 8
        Percentage_screening_left = self.Effort_left_for_screening/(Time_for_screening_and_monitroring_3months_per_emp_per_fund*self.Investment_analysts)
        #Attribute 9
        VC_quality = self.VC_quality
        #Attribute 10
        Endowement = self.Endowement
        #Attribute 11
        Time_till_end_of_investment_stage_E = self.Time_till_end_of_investment_stage_E/(Fund_maturity - Early_startup_exit)
        #Attribute 12
        Time_till_end_of_investment_stage_L = self.Time_till_end_of_investment_stage_L/(Fund_maturity - Late_startup_exit)
        
        state_ = torch.tensor([Prospect_EPI, Prospect_stage, Average_correlation_with_portfolio, Screenings_mean, Screenings_sd, Portfolio_mean, Portfolio_sd, Percentage_screening_left, VC_quality, Endowement, Time_till_end_of_investment_stage_E, Time_till_end_of_investment_stage_L])
        return state_
    
    #Gets next state 
    def get_next_state(self, action, Prospect):
        #Startup_in_portfolio = 0
        #Attribute 1
        Prospect_EPI = 0
        #Attribute 2
        Prospect_stage = 0 
        #Attribute 3
        Average_correlation_with_portfolio = 0
        
        #Cohort attributes
        #Attribute 4
        total_cohort = 0
        for i in self.Screening_prospects:
            total_cohort = getattr(i[0], "EPI_after_screening") + total_cohort
        Screenings_mean = total_cohort/len(self.Screening_prospects) 
        #Attribute 5
        Screenings = []
        for i in self.Screening_prospects:
            Screenings.append(getattr(i[0], "EPI_after_screening"))
        Screenings_sd = np.std(Screenings)
            
        #Portfolio attributes
        #Attribute 6
        total = 0
        for i in self.Portfolio:
            total = getattr(i[0], "EPI_after_screening") + total
        Portfolio_mean = 0
        if self.Portfolio_size != 0:
            Portfolio_mean = total/self.Portfolio_size
        #Attribute 7
        EPIs = []
        Portfolio_sd = 0
        if self.Portfolio_size != 0:
            for i in self.Portfolio:
                EPIs.append(getattr(i[0], "EPI_after_screening"))
            Portfolio_sd = np.std(EPIs)
            
        #VC attributes
        #Attribute 8
        Percentage_screening_left = self.Effort_left_for_screening/(Time_for_screening_and_monitroring_3months_per_emp_per_fund*self.Investment_analysts)
        #Attribute 9
        VC_quality = self.VC_quality
        #Attribute 10
        Endowement = self.Endowement
        #Attribute 11
        Time_till_end_of_investment_stage_E = self.Time_till_end_of_investment_stage_E/(Fund_maturity - Early_startup_exit)
        #Attribute 12
        Time_till_end_of_investment_stage_L = self.Time_till_end_of_investment_stage_L/(Fund_maturity - Late_startup_exit)
        
        next_state_ = torch.tensor([Prospect_EPI, Prospect_stage, Average_correlation_with_portfolio, Screenings_mean, Screenings_sd, Portfolio_mean, Portfolio_sd, Percentage_screening_left, VC_quality, Endowement, Time_till_end_of_investment_stage_E, Time_till_end_of_investment_stage_L])
        return next_state_
    
    def step(self):
        #VC only participates in screening and matching early on in their Fund life cycle 
        if self.Fund_life_stage <= (Fund_maturity - Late_startup_exit):
            for i in self.Screening_prospects:
                if i[1] == "Early" and self.Fund_life_stage >= (Fund_maturity - Early_startup_exit):
                    pass
                else:     
                    done = 0
                    obs = self.get_state(i)
                    #act_1 = agent.choose_action(obs)
                    act = agent.choose_action(obs)
                    reward = self.get_reward(act, i)
                    if 0.005<=act<=1 and float(act) <= self.Endowement:
                        i[0].VC_investments.append(self)
                        self.Portfolio.append(i+[float(act)]+[float(getattr(i[0],"EPI_after_screening"))]+[float(self.Fund_life_stage)])
                        self.Endowement = self.Endowement - float(act)
                    new_state = self.get_next_state(act, obs)
                    agent.remember(obs, act, reward, new_state, int(done))
                    agent.learn()
                    #agent.load_models()
                    #print(agent.memory.state_memory)
                    #print(agent.memory.action_memory)
                    #print("This is a sample")
                    #print(agent.memory.sample_buffer(1))
                    #print("This is a group of parameters")
                    #print(dict(agent.actor.named_parameters()))
                    print("This is effort left for screening")
                    print(self.Effort_left_for_screening)
                    print("This is portfolio length")
                    print(self.Portfolio_size)
        self.Fund_life_stage += 1
        self.Portfolio_size = len(self.Portfolio)
        self.Effort_allocated_to_startups_in_portfolio = self.Portfolio_size*Advising_time
        self.Effort_left_for_screening = self.Effort_left_for_fund - self.Effort_allocated_to_startups_in_portfolio
        self.Number_of_available_screenings = self.Effort_left_for_screening/Screening_time 
        self.Time_till_end_of_investment_stage_E = max(0, (Fund_maturity - Early_startup_exit - self.Fund_life_stage))
        self.Time_till_end_of_investment_stage_L = max(0, (Fund_maturity - Late_startup_exit - self.Fund_life_stage))
        agent.save_models()

class Startup(Agent):
    def __init__(self, unique_id, EPI, Industry, Life_stage, model):
        self.unique_id = unique_id
        self.model = model
        self.Industry = Industry
        self.EPI = EPI
        self.Life_stage = Life_stage
        self.EPI_with_noise = 0
        self.EPI_after_screening = 0
        self.VC_potential_investments = []
        self.VC_investments = []
    
    def average_investor_quality(self):
        total = 0
        if len(self.VC_investments) != 0:
            for i in self.VC_investments:
                total = getattr(i, "VC_quality") + total
            return total/len(self.VC_investments)
        if len(self.VC_investments) == 0:
            return 0
    
    #This is a funciton which makes a startup to progress over time                                
    #The EPI(potential) in time t+1 depends on EPI in time t, advising from VC and random shocks
    def time_progression(self):
        self.EPI = Alpha*self.EPI + Beta*self.average_investor_quality() + np.random.normal(Idiosyncratic_shock_mean, Idiosyncratic_shock_sd)\
        + np.random.normal(Industry_shock_mean, Industry_shock_sd) + np.random.normal(Macro_shock_mean, Macro_shock_sd)
        self.Life_stage += 1
        if self.EPI > 1:
            self.EPI = 0.99
        if self.EPI < 0:
            self.EPI = 0
                                    
    def noise_before_screening(self):
        if self.Life_stage == 0:
            self.EPI_with_noise = self.EPI + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES)
            while self.EPI_with_noise>1 or self.EPI_with_noise<0:
                self.EPI_with_noise = self.EPI + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES)
        else:
            self.EPI_with_noise = self.EPI + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES/(self.Life_stage**(1/2)))
            while self.EPI_with_noise>1 or self.EPI_with_noise<0:
                self.EPI_with_noise = self.EPI + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES/(self.Life_stage**(1/2)))
    
    def noise_after_screening(self):
        if self.Life_stage == 0:
            self.EPI_after_screening = self.EPI + np.random.normal(Noise_mean_after_screening_ES, Noise_standard_deviation_after_screening_ES)
            while self.EPI_after_screening>1 or self.EPI_after_screening<0:
                self.EPI_after_screening = self.EPI + np.random.normal(Noise_mean_after_screening_ES, Noise_standard_deviation_after_screening_ES)
        else:
            self.EPI_after_screening = self.EPI + np.random.normal(Noise_mean_after_screening_ES, Noise_standard_deviation_after_screening_ES/(self.Life_stage**(1/2)))
            while self.EPI_after_screening>1 or self.EPI_after_screening<0:
                self.EPI_after_screening = self.EPI + np.random.normal(Noise_mean_after_screening_ES, Noise_standard_deviation_after_screening_ES)
    
    #Function step refers to range of updates that occur every time step 
    def step(self):
        #self.VC_potential_investments.sort(key = lambda x: x.VC_quality, reverse=True)
        #for i in self.VC_potential_investments[:5]:
            #self.VC_investments.append(i)
        #self.VC_potential_investments = []
        #Updating EPI with noise for startups
        self.noise_before_screening()
        self.noise_after_screening()
        #Collecting the prospects for this time step, 
        #0.450 and 0.570 correspond to levels of EPI that give return greater than 2
        if self.Life_stage == 0 and self.EPI_with_noise > 0.450:
            world.Early_stage_prospects.append(self)
        if self.Life_stage == 8 and self.EPI_with_noise > 0.570:
            world.Late_stage_prospects.append(self) 
        self.time_progression()  
        #We also make all the startups progress in time    
       
        
#Activation class, determines in which order agents are activated
class Activation_1(BaseScheduler):
    def step(self):
        #First, starups are activated
        for agent in self.agent_buffer(shuffled=True):
            if agent.unique_id >= world.VC_number:
                agent.step()
                
class Activation_2(BaseScheduler):
    def step(self):
        #Then, VCs are activated
        for agent in self.agent_buffer(shuffled=True):
            if agent.unique_id < world.VC_number:
                agent.step()
        #After agents are activated, we updated the step value

class World(Model):
    def __init__(self, VC_number, Startup_number):
        self.VC_number = VC_number
        self.Startup_number = Startup_number
        self.schedule_1 = Activation_1(self) 
        self.schedule_2 = Activation_2(self)
        self.Early_stage_prospects = []
        self.Late_stage_prospects = []
        self.VCs = []
        
        #Creating Agents - VC
        for i in range (Number_of_VCs):
            a = VC(i,float(theoretical_distribution_VC.generate_random(1)/VC_max_TVPI),int(lognorm.rvs(Number_of_employees_sd, Number_of_employees_loc, Number_of_employees_scale)),0,self)
            while a.VC_quality>1:
                a.VC_quality = float(theoretical_distribution_VC.generate_random(1)/VC_max_TVPI)
            self.schedule_2.add(a)
            self.VCs.append(a)
            
        self.VCs.sort(key = attrgetter('VC_quality'), reverse = True)                
        
        ##Collecting data
        self.datacollector = DataCollector(
          agent_reporters={"EPI": lambda a: getattr(a, "EPI", None)}
        )

    #Creating Agents - Startups_early stage
    def create_startups_early_1(self):        
        for j in range (Number_of_VCs + self.schedule_1.steps*Number_of_new_startups + self.schedule_1.steps*Number_of_late_stage_startups, Number_of_VCs +  (self.schedule_1.steps+1)*Number_of_new_startups + self.schedule_1.steps*Number_of_late_stage_startups):
            b = Startup(j, powerlaw.rvs(EPI_alpha, EPI_loc, EPI_scale),random.choices(List_of_Industries, Probability_Distribution_of_Industries),0, self)
            self.schedule_1.add(b)         
                
    def create_startups_early_2(self):        
        for j in range (Number_of_VCs + self.schedule_1.steps*Number_of_new_startups + 8*Number_of_late_stage_startups, Number_of_VCs +  (self.schedule_1.steps+1)*Number_of_new_startups + 8*Number_of_late_stage_startups):
            b = Startup(j, powerlaw.rvs(EPI_alpha, EPI_loc, EPI_scale),random.choices(List_of_Industries, Probability_Distribution_of_Industries),0, self)       
            self.schedule_1.add(b) 
       
    #Creating Agents - Startups_late stage
    def create_startups_late(self):
        #Creating list of late_stage EPIs 
        Data = []
        for i in range(Number_of_new_startups):
            epi = powerlaw.rvs(EPI_alpha, EPI_loc, EPI_scale)
            epi_with_noise = epi + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES)
            while epi_with_noise >1 or epi_with_noise<0:
                epi_with_noise = epi + np.random.normal(Noise_mean_before_screening_ES, Noise_standard_deviation_before_screening_ES)
            epi_after_screening = epi + np.random.normal(Noise_mean_after_screening_ES, Noise_standard_deviation_after_screening_ES)
            while epi_after_screening >1 or epi_after_screening<0:
                epi_after_screening = epi + np.random.normal(Noise_mean_after_screening_ES, Noise_standard_deviation_after_screening_ES)
            data_point = [epi, epi_with_noise, epi_after_screening]
            if data_point[1] > 0.450:
                Data.append(data_point)
        Data.sort(key = lambda x: x[1], reverse = True)
        Data_selection = Data[:Estimate_of_early_stage_screenings]
        Data_selection.sort(key = lambda x: x[2], reverse = True)
        Data_selection_final = Data_selection[:Number_of_late_stage_startups]
        EPI_list = []
        for i in Data_selection_final:
            EPI_list.append(i[0])    
        for (j, i) in zip(range (Number_of_VCs + (self.schedule_1.steps+1)*Number_of_new_startups + self.schedule_1.steps*Number_of_late_stage_startups, Number_of_VCs + (self.schedule_1.steps+1)*Number_of_new_startups + (self.schedule_1.steps+1)*Number_of_late_stage_startups), range(Number_of_late_stage_startups)):
            c = Startup(j, EPI_list[i],random.choices(List_of_Industries, Probability_Distribution_of_Industries),8, self)
            self.schedule_1.add(c)
                
    def startups_generation(self):
        if self.schedule_1.steps<8:
            self.create_startups_early_1()
            self.create_startups_late()
        else:
            self.create_startups_early_2()
            
    def matching_prospects_to_VCs(self):
        index = 0
        for i in world.Early_stage_prospects:
            for j in world.VCs:
                if getattr(j, "Number_of_available_screenings")*(1-Percentage_of_startups_getting_to_later_stage) > 1+ len(getattr(j, "Screening_prospects")):
                    j.Screening_prospects.append([i, "Early"])
                    i.VC_potential_investments.append([j])
                    index +=1
                if index == 10:
                    break
            index = 0
        index = 0 
        for i in world.Late_stage_prospects:
            for j in world.VCs:
                if getattr(j, "Number_of_available_screenings") > len(getattr(j, "Screening_prospects")):
                    j.Screening_prospects.append([i, "Late"])
                    i.VC_potential_investments.append([j])
                    index +=1
                if index == 10:
                    break
            index = 0
        
    def step(self):
        self.Early_stage_prospects = []
        self.Late_stage_prospects = []
        self.startups_generation()
        self.schedule_1.step()
        self.Early_stage_prospects.sort(key = attrgetter('EPI_with_noise'), reverse = True)
        self.Late_stage_prospects.sort(key = attrgetter('EPI_with_noise'), reverse = True)
        self.matching_prospects_to_VCs()
        self.schedule_2.step()
        self.schedule_1.steps += 1
        self.schedule_1.time += 1
        #self.datacollector.collect(self)

world = World(Number_of_VCs, Number_of_new_startups)
for i in range(40):
    world.step()
    
Statistics_on_VCs = []
for i in world.VCs:
    Statistics_on_VCs = [[i.unique_id, i.VC_quality, i.Portfolio_size, i.Endowement, i.Investment_analysts, i.final_return(i.Portfolio)]] + Statistics_on_VCs

df = pd.DataFrame(np.array(Statistics_on_VCs), columns = ["Unique_id", "VC_Quality","Portfolio_size", "Endowement", "Investment_analysts", "Final_return"])
df.to_excel("Statistics_on_VCs.xlsx")
print(df)


Portfolio_data = []    
for i in world.VCs:
    for j in i.Portfolio:
        Transit = [i.unique_id, j[0].unique_id, j[0].EPI, i.EPI_to_returns(j[0].EPI,j[1])]
        for z in j[1:]:
            Transit.append(z)
        Transit.append(j[0].Industry[0])
        Portfolio_data = [Transit] + Portfolio_data   
df2 = pd.DataFrame(np.array(Portfolio_data), columns = ["Unique_id_VC", "Unique_id_Startup", "EPI_final", "Return", "Stage","Amount_Invested", "EPI_after_screening", "Fund_life_stage", "Industry"])
df2.to_excel("Portfolio_data.xlsx")
print(df2) 
        
        