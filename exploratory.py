import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np



#Find out the nulls
df = pd.read_csv('ALL_PIT_DEC_2014_NOV_2015_raw.csv')


missing_count = df.isnull().sum()


missing_df = pd.DataFrame({'Column_Name':missing_count.index, 'No. of Missing rows':missing_count.values})

missing_df['missing_prop'] = (missing_df['No. of Missing rows'] * 100)/(df.shape[0])
missing_df = missing_df.sort_values(by= 'missing_prop', ascending=False)
missing_df = missing_df[missing_df['No. of Missing rows'] != 0]

data = [go.Bar(
            x=missing_df['Column_Name'],
            y=missing_df['missing_prop']
    )]



layout = go.Layout(title='A Simple Plot', width=1500, height=640, margin=go.Margin(
                                                                                  l=50,
                                                                                  r=50,
                                                                                  b=160,
                                                                                  t=100,
                                                                                  pad=4
                                                                              ), 
                  xaxis= dict(autotick=False, tickangle=45, ticks='outside',ticklen=8, tickfont=dict(
            family='Courier New, monospace, serif',
            size=10,
            color='black'
        )))
fig = go.Figure(data=data, layout=layout)

py.image.save_as(fig, filename='a-simple-plot.png')



# #Diff Between the crs and Actual Time
# our_df = pd.DataFrame()


# our_df['csr_time'] = df['CRS_ELAPSED_TIME']
# our_df['actual_time'] = df['ACTUAL_ELAPSED_TIME']
# our_df['carrier'] = df['UNIQUE_CARRIER']
# our_df = our_df.dropna() 



# our_df['crs-actual'] = (our_df['csr_time'] - our_df['actual_time'])
# our_df['color'] = np.where(our_df['crs-actual']< 0, '#db5a44', '#447adb')


# airlines = list(our_df['carrier'].unique())

# airlines_df = pd.read_csv('airlines.csv')


# for a in airlines:
#   subrows = our_df[our_df['carrier'] == a]
#   data = [go.Bar(
#               x = subrows['csr_time'],
#               y = subrows['crs-actual'],
#               marker = dict(
#                 color = subrows['color']

#                 ) 
#          )]


#   layout = go.Layout(title='A Simple Plot', width=2000, height=1240, margin=go.Margin(
#                                                                                     l=50,
#                                                                                     r=50,
#                                                                                     b=160,
#                                                                                     t=100,
#                                                                                     pad=4
#                                                                                 ), 
#                     xaxis= dict(autotick=False, tickangle=45, ticks='outside',ticklen=0, tickfont=dict(
#                                                                                           family='Courier New, monospace, serif',
#                                                                                           size=0,
#                                                                                           color='black')
#                                ), 
#                     bargap= 12.25)

#   fig = go.Figure(data=data, layout=layout)

#   py.image.save_as(fig, filename=(a + '.png'))

















#How many times the flight was actually early insteadf of being late




#which delay block is the most important













