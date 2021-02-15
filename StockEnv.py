# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 00:36:36 2021

@author: Benst
"""
import pandas
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
import copy

class State:
    def __init__(self, date, price, volume, shares, balance):
        self.Date = date
        self.Price = price
        self.Volume = volume
        self.Shares = shares
        self.Balance = balance
    
    def to_array(self):
        array = []
        array.append(self.Date)
        array.append(self.Price)
        array.append(self.Volume)
        array.append(self.Shares)
        array.append(self.Balance)
       # print(array)
        return array
    
    
class StockEnv:
    
    def __init__(self, ticker, startBal, batchsize, timestep):
        data = self.LoadData(ticker)
        self.BatchSize = batchsize
        self.Finished = False
        self.Data, self.PriceScale, self.VolumeScale = self.FormatData(data)
        self.TimeStep = timestep
        self.BalScale = 2000
        self.RewardScale = 1
        self.ShareScale = 100
        firstrow = self.Data.iloc[timestep]
        self.State = State(firstrow['Datetime'], firstrow['Price'], firstrow['Volume'], 0, startBal/self.BalScale)
        self.States =pandas.DataFrame(np.zeros((self.BatchSize,5)))
        self.States.iloc[0] = self.State.to_array()
        
    def LoadData(self, ticker):
        yf.pdr_override() # <== that's all it takes :-)
        ftse = yf.Ticker(ticker)
        #data = ftse.history(period= "1wk", interval="1m")
        data = ftse.history(start="2021-01-20", end="2021-01-27", interval = "1m")
        data2 = ftse.history(start="2021-01-27", end="2021-02-03", interval = "1m")
        data3 = ftse.history(start="2021-02-03", end="2021-02-10", interval = "1m")
        #data4 = ftse.history(start="2021-01-10", end="2021-01-15", interval = "1m")
        #data5 = ftse.history(start="2021-01-01", end="2021-01-08", interval = "1m")
        data = pandas.concat([data,data2,data3], axis = 0)
        data = data.reset_index()
        return data
    
    def FormatData(self, data):    
        Price = data['Open']
        Volume = data['Volume']
        Time = (data['Datetime'].dt.day) + (data['Datetime'].dt.hour/24) + (data['Datetime'].dt.minute/1440)
        Time = Time/3000
        X = pandas.DataFrame(Time)
        X ['Price'] = Price/Price.max()
        
        X['Volume'] = Volume/Volume.max()
        return X, Price.max(), Volume.max()
    
    def Step(self, action):
        self.Penalty = 0
        self.PreviousState = copy.deepcopy(self.State)
        self.States = self.States.shift(1)
       # self.AllStates.append(self.State.to_array())
        if (action == 0):
            self.Hold()
        elif(action == 1):
            self.Buy()
        else:
            self.Sell()
        self.State.Shares = (round(self.State.Shares*self.ShareScale))/self.ShareScale
        reward = self.GetReward()
        self.States.iloc[0] = self.State.to_array()
        if(self.TimeStep > len(self.Data.index)-4):
            self.Finished = True
        return self.States.to_numpy(), reward
    
    def GetStateValue(self, state):
        return ((state.Balance * self.BalScale) + ((state.Shares*self.ShareScale) * (state.Price* self.PriceScale)))
    
    def GetReward(self):
        thisR = self.GetStateValue(self.State)
        lastR = self.GetStateValue(self.PreviousState)
        return (thisR - lastR)/self.RewardScale
    
    def AdvanceTime(self):
        self.TimeStep+=1
        row = self.Data.iloc[self.TimeStep]
        self.State.Date = row['Datetime']
        self.State.Price = row['Price']
        self.State.Volume = row['Volume']
    
    def Buy(self):
        self.LastAction = "Buy"
        if((self.State.Balance * self.BalScale) - (self.State.Price * self.PriceScale) < 0):
            self.Penalty = 1
            self.Hold()
        else:
            self.State.Balance -= (self.State.Price * self.PriceScale)/self.BalScale
            self.State.Shares += (1/ self.ShareScale)
            self.AdvanceTime()
    
    def Hold(self):
        self.LastAction = "Hold"
        self.AdvanceTime()
        
    def Sell(self):
        self.LastAction = "Sell"
        if(self.State.Shares * self.ShareScale < 1):
            self.Hold()
            self.Penalty = 1
        else:
            self.State.Balance += (self.State.Price * self.PriceScale)/self.BalScale
            self.State.Shares -= (1/ self.ShareScale)
            self.AdvanceTime()
    
            
