import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima_model import ARIMA

class paper_value:
    
    def __init__(self, paper = None, market_indicator = '^BVSP', type_of_period = None, len_period = 5, type_of_frequency = None, len_of_frequency = 5, cdi_anual = 4.25):
        self.set_paper(paper)
        self.set_market_indicator(market_indicator)
        self.type_period = type_of_period if type_of_period else share.PERIOD_TYPE_DAY
        self.type_frequency = type_of_frequency if type_of_frequency else share.FREQUENCY_TYPE_MINUTE
        self.len_period = len_period
        self.len_of_frequency = len_of_frequency
        self.cdi_annual = cdi_anual
        self.data = self._get_historical_data(self.paper)
        self.market_data = self._get_historical_data(self.market_indicator)
    
    def _get_paper_data(self, action):
        symbol_data = None
        try:
            symbol_data = action.get_historical(self.type_period,
                                                      self.len_period,
                                                      self.type_frequency,
                                                      self.len_of_frequency)
        except YahooFinanceError as e:
            print(e.message)
            sys.exit(1)
        return symbol_data
    
    def _get_historical_data(self, paper):
        symbol_paper = self._get_paper_data(paper)
        data = pd.DataFrame(data = symbol_paper)
        data['Data'] = self.modify_timestamp(data['timestamp'])
        data.set_index('Data', inplace = True)
        data.drop(columns = 'timestamp', inplace = True)
        data.dropna(inplace = True)
        self.log_returns(data)
        self.percent_returns(data)
        return data
        
    def get_paper_data(self):
        return self.data
    
    def get_market_data(self):
        return self.market_data
    
    def modify_timestamp(self, list_of_timestamp):
        timestamp_to_data = [datetime.fromtimestamp(int(str(x)[:10])) for x in list_of_timestamp]
        timestamp_to_data = pd.to_datetime(timestamp_to_data)
        return timestamp_to_data
    
    def get_finance_indicator(self, indicator_return = 'log_returns', start = None, end = None):
        start = start if start else self.data.index[0]
        end = end if end else self.data.index[-1]
        indice_beta = self.beta_indicator(self.data, self.market_data, indicator_return, start, end)
        performance_atual = self.total_return_action(self.data, start, end)
        performance_indicador = self.total_return_action(self.market_data, start, end)
        indice_sharpe = self.sharpe_indice(self.data, start, end)
        indice_sharpe_indicador = self.sharpe_indice(self.market_data, start, end)
        media_price_action = self.media_price(self.data)
        finance_indicator = {'Preço Médio do Ativo': media_price_action,'Indice Beta': indice_beta, 'Performance Atual do Ativo' : performance_atual, 
                             'Performance do Indicador' : performance_indicador, 'Índice Sharpe do Ativo' : indice_sharpe, 
                            'Índice Sharpe do Indicador' : indice_sharpe_indicador}
        return finance_indicator
    
    def log_returns(self, df):
        df['close_1'] = df['close'].shift(1)
        df['log_returns'] = np.log(df['close']/df['close_1'])
        df['log_returns'].fillna(0, inplace = True)
        df.drop(columns = 'close_1', inplace = True)
        
    def percent_returns(self, df):
        df['return_percent'] = np.exp(df['log_returns']) - 1
    
    def set_paper(self, paper_name):
        self.paper = share.Share(paper_name)
        # self.data = self._get_historical_data(self.paper)
        
    def set_market_indicator(self, market_indicator):
        self.market_indicator = share.Share(market_indicator)
        # self.market_data = self._get_historical_data(self.market_indicator)
        
    def beta_indicator(self, action, market, indicator_return, start, end):
        self.merged_data = pd.merge(action[[indicator_return]], market[[indicator_return]], how = 'inner', on = 'Data', suffixes = ('_action', '_market'))
        data_for_beta_computer = self.merged_data[((self.merged_data.index >= start) & (self.merged_data.index <= end))]
        covariance_matrix = data_for_beta_computer.cov()
        beta_indicator_number = covariance_matrix.iloc[0][indicator_return+'_action']/covariance_matrix.iloc[1][indicator_return+'_market']
        return beta_indicator_number
    
    def sharpe_indice(self, action, start = None, end = None):
        investiment_return = self.total_return_action(self.data, start, end)
        volatility = self.volatility_computer(action, start, end)
        len_days = (end - start).days
        cdi_to_period = self.cdi_annual*len_days/252
        shapiro = (investiment_return - cdi_to_period)/volatility
        return shapiro
    
    def total_return_action(self, action, start = None, end = None):
        investiment_return = (action[action.index == end]['close'][0] - action[action.index == start]['close'][0])*100/action[action.index == start]['close'][0]
        return investiment_return
    
    def volatility_computer(self, paper_value, start, end):
        prices = paper_value[((paper_value.index >= start) & (paper_value.index <= end))]['log_returns'].values
        standard_deviation = np.std(prices*100)
        return standard_deviation
    
    def media_price(self, action):
        media_price = np.sum((action['close'].values*action['volume'].values))/np.sum(action['volume'].values)
        return media_price
    
    def visualization_action(self):
        df_plot = self.data.copy()
        info = self.get_finance_indicator()
        df_plot.reset_index(inplace = True)
        cand = go.Candlestick(
                        x=df_plot['Data'],
                        open=df_plot['open'], high=df_plot['high'],
                        low=df_plot['low'], close=df_plot['close'],
                        increasing_line_color= '#66CC99', decreasing_line_color= '#CC6666', name = 'Stock'
                        )
        hist_action = go.Histogram(y=df_plot['close'], xaxis="x2", yaxis = 'y', name = 'Histogram Stock')
        
        log_returns = go.Scatter(x = df_plot['Data'], y = df_plot['log_returns']*100, xaxis = 'x', yaxis = 'y3', name = 'Log Returns')
        
        hist_log = go.Histogram(y=df_plot['log_returns']*100, xaxis = "x4", yaxis = 'y3', name = 'Histogram Log Returns')
        
        performance = go.Indicator(
            mode = "number+delta",
            value = info['Performance Atual do Ativo'],
            number = {'suffix': '%'},
            delta = {"reference": info['Performance do Indicador'], "valueformat": ".2f", 'relative': True},
            title = {"text": "<br><span style='font-size:0.6em;color:#9D5D21'>Comp. entre Ativo e Indicador</span>"},
            domain = {'x': [0.9, 1], 'y': [0.2, 0.8]})
        
        layout = go.Layout(
            xaxis=dict(
                domain=[0, 0.5]
            ),
            xaxis2=dict(
                domain=[0.55, 0.85]
            ),
            xaxis4=dict(
                domain=[0.55, 0.85],
                anchor="y3"
            ),
            yaxis=dict(
                domain=[0, 0.7]
            ),
            yaxis3 = dict(
                domain=[0.8, 1])
        )
        
        data = [cand, hist_action, log_returns, hist_log, performance]
        
        fig = go.Figure(data=data, layout = layout)
        
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=False)
        fig.update_layout(title_text='Prices of Stock by Time', xaxis_rangeslider_visible=False, plot_bgcolor='#204161', paper_bgcolor = "#143A5E", font={"color": "white"})

        return fig
    def visualization_cov_var(self):
        select_columns = [x for x in self.data.columns if 'log_returns' in x ]
        all_data = pd.merge(self.data[select_columns], self.market_data[['log_returns']], how = 'inner', suffixes = ('', '_market'), left_index=True, right_index=True)
        data = all_data.cov()['log_returns_market'][:-1]/all_data['log_returns_market'].var()
        fig = go.Figure(go.Heatmap(
                        z=[data.values],
                        colorscale='magma',
                        zmid=data.mean()))
        fig.update_xaxes(tickvals = list(range(len(data))), ticktext = data.index)
        fig.update_layout(title_text='Beta Indices with Market Indicator', xaxis_rangeslider_visible=False, plot_bgcolor='#204161', paper_bgcolor = "#143A5E", font={"color": "white"})
        return fig
