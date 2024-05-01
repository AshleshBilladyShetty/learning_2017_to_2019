
#Dec6 EDA V1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df =    pd.read_pickle("rawDataMOA2.pickle")
vi =    pd.read_pickle("visualIDLvlData.pickle")

vi.head(2)
df.head(2)
vi.columns
df2.shape

df1 = df[['usageid', 'usetime', 'accesscode', 'visualid', 'rideid', 'qty',  'useno']]
ticket = pd.read_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\D_TicketsAttributes.csv') 
df2 = df1.merge(ticket[['AccessCode','Name','Type','Category','Subcategory','ParetoFl', 'UtilizationBase','Price','Tim_Pt_Fl']], \
how = 'left', left_on = ['accesscode'] , right_on = ['AccessCode'], )



df1.columns
df.dtypes
df1 = df.drop_duplicates(subset = ['visualid'])

grp = df.groupby(['visualid','Category']).agg({'visualid':"count",
                                                                    'usetime':[min,max],
                                                                    'Points': [sum]}).reset_index()
grp.columns = ["_".join(x) for x in grp.columns.ravel() ]
grp['timeSpent'] = (grp['usetime_max'] - grp['usetime_min']).dt.seconds


df1.groupby(['Category']).size().reset_index().to_csv('test.csv')

'usageid', 'usetime', 'accesscode', 'visualid', 'rideid', 'qty',
       'useno', 'filename', 'date', 'time', 'hour', 'Name_x', 'Type',
       'Category', 'ParetoFl', 'UtilizationBase', 'Price', 'Tim_Pt_Fl',
       'Name_y', 'Points', 'RideType', 'MinHeight', 'Direction',
       'RideIntensityRating', 'RollerCoasterFL', 'RideDurCliSec',
       'RideCapacity', 'VL_pointsSum', 'VL_rideCount', 'VL_timeSpent',
       'VL_satisfFl', 'VL_tickBehavClust'


###############################################################################
       
     
#EDA 29 Nov 2017
       
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:18:26 2017

@author: shett075
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series

clsdata.head(2)

clsdata.groupby(['clusters2']).mean()

clsdata = pd.read_csv('cluster_final_data.csv')
rawDataMOA1 = pd.read_pickle('rawDataMOA2.pickle')
visualIDLvlData = pd.read_pickle('visualIDLvlData.pickle')

tick1 = clsdata.merge(visualIDLvlData[['VL_visualid','VL_UtilizationBase']], how = 'left' , left_on = 'visualid', right_on = 'VL_visualid')

rawDataMOA1.head(2)
clsdata.head(2)
visualIDLvlData.head(2)

tick2 = tick1.groupby(['VL_UtilizationBase','clusters2']).size().reset_index(name  = 'customercnt').sort_values(['clusters2','VL_UtilizationBase'])
tick2.sort_values(['clusters2','VL_UtilizationBase'])
tick1.shape
tick1.head(2)

tick2.to_csv('test.csv')


overall = visualIDLvlData.groupby(['VL_UtilizationBase']).size().reset_index(name  = 'customercnt').sort_values(['VL_UtilizationBase'])
overall.to_csv('test.csv')

custride1 = rawDataMOA1[['visualid','rideid']].drop_duplicates()
custride2  = custride1.groupby(['rideid']).size().reset_index(name = 'customercnt')
custride2.to_csv('test.csv')


####################################################################################

#EDA V1 11Nov2017


import pandas as pd
import numpy as np


usage1 = pd.read_excel("Usage1.xlsx");
usage2 = pd.read_excel("Usage2.xlsx");
usage3 = pd.read_excel("Usage3.xlsx");
usage4 = pd.read_excel("Usage4.xlsx");
usage5 = pd.read_excel("Usage5.xlsx");
usage6 = pd.read_excel("Usage6.xlsx");
usage7 = pd.read_excel("Usage7.xlsx");
usage8 = pd.read_excel("Usage8.xlsx");
usage9 = pd.read_excel("Usage9.xlsx");
usage10 = pd.read_excel("Usage10.xlsx");
usage11 = pd.read_excel("Usage11.xlsx");
usage12 = pd.read_excel("Usage12.xlsx");
usage13 = pd.read_excel("Usage13.xlsx");
usage14 = pd.read_excel("Usage14.xlsx");
usage15 = pd.read_excel("Usage15.xlsx");
usage16 = pd.read_excel("Usage16.xlsx");
usage17 = pd.read_excel("Usage17.xlsx");

# new files that were shared as fix
usageFix = pd.read_excel("UsageFix.xlsx");
usageFix2 = pd.read_excel("UsageFix2.xlsx");

usage1.shape
usage1.head
usage1.columns
usage1.dtypes

usage1['filename'] = 'usage1'
usage2['filename'] = 'usage2'
usage3['filename'] = 'usage3'
usage4['filename'] = 'usage4'
usage5['filename'] = 'usage5'
usage6['filename'] = 'usage6'
usage7['filename'] = 'usage7'
usage8['filename'] = 'usage8'
usage9['filename'] = 'usage9'
usage10['filename'] = 'usage10'
usage11['filename'] = 'usage11'
usage12['filename'] = 'usage12'
usage13['filename'] = 'usage13'
usage14['filename'] = 'usage14'
usage15['filename'] = 'usage15'
usage16['filename'] = 'usage16'
usage17['filename'] = 'usage17'

# new files that were shared as fix
usageFix['filename'] = 'usageFix'
usageFix2['filename'] = 'usageFix2'

print({},format(usage1.shape))
print({},format(usage2.shape))
print({},format(usage3.shape))
print({},format(usage4.shape))
print({},format(usage5.shape))
print({},format(usage6.shape))
print({},format(usage7.shape))
print({},format(usage8.shape))
print({},format(usage9.shape))
print({},format(usage10.shape))
print({},format(usage11.shape))
print({},format(usage12.shape))
print({},format(usage13.shape))
print({},format(usage14.shape))
print({},format(usage15.shape))
print({},format(usage16.shape))
print({},format(usage17.shape))

usagedatacoll =  usage1.append([usage2,usage3,usage4,usage5,usage6,usage7,usage8,usage9,
          usage10,usage11,usage12,usage13,usage14,usage15,usage16,usage17],
          ignore_index = True)
          
usagedatacoll.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA.pickle')
#del usagedatacoll
data = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA.pickle');


data1 = data[~data['usageid'].isin(usageFix2['usageid'])]
data2 = data1.append([usageFix2],ignore_index = True)

#data2.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA1.pickle')

#new data 
data_new = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA1.pickle');
data_new.shape

############################################################################

#EDA v1 19Nov2017

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#########STEP1: Import and collate the data provided by client ########################

usage1 = pd.read_excel("Usage1.xlsx");
usage2 = pd.read_excel("Usage2.xlsx");
usage3 = pd.read_excel("Usage3.xlsx");
usage4 = pd.read_excel("Usage4.xlsx");
usage5 = pd.read_excel("Usage5.xlsx");
usage6 = pd.read_excel("Usage6.xlsx");
usage7 = pd.read_excel("Usage7.xlsx");
usage8 = pd.read_excel("Usage8.xlsx");
usage9 = pd.read_excel("Usage9.xlsx");
usage10 = pd.read_excel("Usage10.xlsx");
usage11 = pd.read_excel("Usage11.xlsx");
usage12 = pd.read_excel("Usage12.xlsx");
usage13 = pd.read_excel("Usage13.xlsx");
usage14 = pd.read_excel("Usage14.xlsx");
usage15 = pd.read_excel("Usage15.xlsx");
usage16 = pd.read_excel("Usage16.xlsx");
usage17 = pd.read_excel("Usage17.xlsx");

# new files that were shared as fix
usageFix = pd.read_excel("UsageFix.xlsx");
usageFix2 = pd.read_excel("UsageFix2.xlsx");

usage1.shape
usage1.head
usage1.columns
usage1.dtypes

usage1['filename'] = 'usage1'
usage2['filename'] = 'usage2'
usage3['filename'] = 'usage3'
usage4['filename'] = 'usage4'
usage5['filename'] = 'usage5'
usage6['filename'] = 'usage6'
usage7['filename'] = 'usage7'
usage8['filename'] = 'usage8'
usage9['filename'] = 'usage9'
usage10['filename'] = 'usage10'
usage11['filename'] = 'usage11'
usage12['filename'] = 'usage12'
usage13['filename'] = 'usage13'
usage14['filename'] = 'usage14'
usage15['filename'] = 'usage15'
usage16['filename'] = 'usage16'
usage17['filename'] = 'usage17'

# new files that were shared as fix
usageFix['filename'] = 'usageFix'
usageFix2['filename'] = 'usageFix2'

print({},format(usage1.shape))
print({},format(usage2.shape))
print({},format(usage3.shape))
print({},format(usage4.shape))
print({},format(usage5.shape))
print({},format(usage6.shape))
print({},format(usage7.shape))
print({},format(usage8.shape))
print({},format(usage9.shape))
print({},format(usage10.shape))
print({},format(usage11.shape))
print({},format(usage12.shape))
print({},format(usage13.shape))
print({},format(usage14.shape))
print({},format(usage15.shape))
print({},format(usage16.shape))
print({},format(usage17.shape))


usagedatacoll =  usage1.append([usage2,usage3,usage4,usage5,usage6,usage7,usage8,usage9,
          usage10,usage11,usage12,usage13,usage14,usage15,usage16,usage17],
          ignore_index = True)
          
usagedatacoll.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA.pickle')
#del usagedatacoll
data = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA.pickle');


data1 = data[~data['usageid'].isin(usageFix2['usageid'])]
data2 = data1.append([usageFix2],ignore_index = True)

data2.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA1.pickle')



#########STEP2: Import the cleaned data and conduct some basic EDA ########################

#new data 
data_new = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA1.pickle');
df = data_new.drop_duplicates(subset = ['usetime','accesscode','visualid','rideid','useno'] ,keep ='last')
df['date']  = df['usetime'].dt.date
df['time'] = df['usetime'].dt.time
df['hour'] = df['usetime'].dt.hour

#import tickets
ticket = pd.read_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\D_TicketsAttributes.csv') 
ride =  pd.read_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\D_RidesAttributes_v2.csv')

# join ride attributes and ticket attributes to the df file
df1 = df.merge(ticket[['AccessCode','Name','Type','Category','ParetoFl', 'UtilizationBase','Price','Tim_Pt_Fl']], \
how = 'left', left_on = ['accesscode'] , right_on = ['AccessCode'], )

df2 = df1.merge(ride[['RideID','Name','Points', 'RideType','MinHeight','Direction', 'RideIntensityRating', \
'RollerCoasterFL','RideDurCliSec','RideCapacity']], how = 'left',left_on = ['rideid'] , right_on = ['RideID'])

################# EDA Codes

#see accescodes freauency
df.groupby('accesscode').size().reset_index(name = 'rowcount').to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#Solved encoding error while exporting to csv
#a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv',sep='\t', encoding='utf-8')

#see rides freauency
df.groupby('rideid').size().reset_index(name = 'rowcount').to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#See hour level ride frequency
df.groupby(['hour']).size()
df.head(2)

# check for the frequency of the ride

a = df2.groupby(['visualid']).size().reset_index(name = 'rowcount')
a.groupby(['rowcount']).size().to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

b = df2.merge(a, how = 'left', left_on = ['visualid'], right_on = ['visualid'])
b.groupby(['rowcount','Name_x']).size().reset_index(name = "countrides"). \
to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

c = b.drop_duplicates(subset =['visualid','rowcount','UtilizationBase'] ,keep = 'last')
c.groupby(['rowcount','UtilizationBase']).size().reset_index(name = "countrides"). \
to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#########STEP3: Revolutionary customer satisfaction Indicator calculation ########################

grp = df2.groupby(['visualid','accesscode','UtilizationBase']).agg({'visualid':"count",
                                                                    'usetime':[min,max],
                                                                    'Points': [sum]}).reset_index()
grp.columns = ["_".join(x) for x in grp.columns.ravel() ]
grp['timeSpent'] = (grp['usetime_max'] - grp['usetime_min']).dt.seconds

distr = grp.groupby(['UtilizationBase_'])[['visualid_count','Points_sum','timeSpent']]. \
quantile([0.0,.002,0.023,0.159,0.25, 0.5, 0.75, 0.841,0.977,0.998,1.0],).reset_index()

distr.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

grp.groupby(['UtilizationBase_']).size().to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#Create satisfaction flag for all the customers

grp['satisfFl'] = np.where( (grp['UtilizationBase_'] == '1') & (grp['Points_sum'] >= 3), "VerySat", 
                  np.where( (grp['UtilizationBase_'] == '1') & (grp['Points_sum'] < 3), "NotSat", 
                  np.where( (grp['UtilizationBase_'] == '3') & (grp['Points_sum'] >= 3), "VerySat", 
                  np.where( (grp['UtilizationBase_'] == '3') & (grp['Points_sum'] < 3), "NotSat", 
                  np.where( (grp['UtilizationBase_'] == '6') & (grp['Points_sum'] >= 6), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '6') & (grp['Points_sum'] < 6), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '9') & (grp['Points_sum'] >= 9), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '9') & (grp['Points_sum'] < 9 ), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '18') & (grp['Points_sum'] >= 18), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '18') & (grp['Points_sum'] < 18), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '19') & (grp['Points_sum'] >= 19), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '19') & (grp['Points_sum'] < 19), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '20') & (grp['Points_sum'] >= 20), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '20') & (grp['Points_sum'] < 20), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '25') & (grp['Points_sum'] >= 25), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '25') & (grp['Points_sum'] < 25), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '30') & (grp['Points_sum'] >= 30), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '30') & (grp['Points_sum'] < 30), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '34') & (grp['Points_sum'] >= 34), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '34') & (grp['Points_sum'] < 34), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '35') & (grp['Points_sum'] >= 34), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '35') & (grp['Points_sum'] < 34), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '48') & (grp['Points_sum'] >= 48), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '48') & (grp['Points_sum'] < 48) & (grp['Points_sum'] >= 42), "Sat",
                  np.where( (grp['UtilizationBase_'] == '48') & (grp['Points_sum'] < 42) , "NotSat",
                  np.where( (grp['UtilizationBase_'] == '49') & (grp['Points_sum'] >= 48), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '49') & (grp['Points_sum'] < 48) & (grp['Points_sum'] >= 42), "Sat",
                  np.where( (grp['UtilizationBase_'] == '49') & (grp['Points_sum'] < 42), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '50') & (grp['Points_sum'] >= 48), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '50') & (grp['Points_sum'] < 48) & (grp['Points_sum'] >= 42), "Sat",
                  np.where( (grp['UtilizationBase_'] == '50') & (grp['Points_sum'] < 42), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '84') & (grp['Points_sum'] >= 84), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '84') & (grp['Points_sum'] < 84) & (grp['Points_sum'] >= 72), "Sat",
                  np.where( (grp['UtilizationBase_'] == '84') & (grp['Points_sum'] < 72), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '85') & (grp['Points_sum'] >= 84), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '85') & (grp['Points_sum'] < 84) & (grp['Points_sum'] >= 72), "Sat",
                  np.where( (grp['UtilizationBase_'] == '85') & (grp['Points_sum'] < 72), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '96') & (grp['Points_sum'] >= 96), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '96') & (grp['Points_sum'] < 96) & (grp['Points_sum'] >= 72), "Sat",
                  np.where( (grp['UtilizationBase_'] == '96') & (grp['Points_sum'] < 72), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '250') & (grp['Points_sum'] >= 34), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '250') & (grp['Points_sum'] < 34), "NotSat",
                  np.where( (grp['UtilizationBase_'] == 'AllDay') & (grp['Points_sum'] >= 72), "VerySat",
                  np.where( (grp['UtilizationBase_'] == 'AllDay') & (grp['Points_sum'] < 72) & (grp['Points_sum'] >= 36), "Sat",
                  np.where( (grp['UtilizationBase_'] == 'AllDay') & (grp['Points_sum'] <36), "NotSat",
                  np.where( (grp['UtilizationBase_'] == 'Hour3') & (grp['Points_sum'] >= 54), "VerySat",
                  np.where( (grp['UtilizationBase_'] == 'Hour3') & (grp['Points_sum'] < 54) & (grp['Points_sum'] >= 33), "Sat",
                  np.where( (grp['UtilizationBase_'] == 'Hour3') & (grp['Points_sum'] < 33), "NotSat",
                  np.where( (grp['UtilizationBase_'] == 'Hour5') & (grp['Points_sum'] >= 66), "VerySat",
                  np.where( (grp['UtilizationBase_'] == 'Hour5') & (grp['Points_sum'] < 66) & (grp['Points_sum'] >= 39), "Sat",
                  np.where( (grp['UtilizationBase_'] == 'Hour5') & (grp['Points_sum'] < 39), "NotSat",
                  np.where( (grp['UtilizationBase_'] == 'TwoDay') & (grp['Points_sum'] >= 153), "VerySat",
                  np.where( (grp['UtilizationBase_'] == 'TwoDay') & (grp['Points_sum'] < 153) & (grp['Points_sum'] >= 81), "Sat",
                  np.where( (grp['UtilizationBase_'] == 'TwoDay') & (grp['Points_sum'] < 81), "NotSat","miss"
                  ))))))))))))))))))))))))))))))))))))))))))))))))))))))  


a = grp.groupby(['UtilizationBase_','satisfFl']).size().reset_index(name = 'countOfViewerid')
a.pivot(index = 'UtilizationBase_', columns = 'satisfFl', values = 'countOfViewerid').reset_index().to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

a = grp.groupby(['satisfFl']).agg({'Points_sum' : ['mean'],
                                   'visualid_count':['mean'],
                                   'timeSpent':['mean']}).reset_index()
a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')


#add ticket final buckets 

grp['tickBehavClust'] = np.where( (grp['UtilizationBase_'] == '1'), "PointsLow", 
                        np.where( (grp['UtilizationBase_'] == '3'), "PointsLow", 
                        np.where( (grp['UtilizationBase_'] == '6'), "PointsLow", 
                        np.where( (grp['UtilizationBase_'] == '9'), "PointsLow", 
                        np.where( (grp['UtilizationBase_'] == '18'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '19'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '20'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '25'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '30'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '34'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '35'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '48'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '49'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '50'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '84'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '85'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '96'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '250'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == 'AllDay'), "WBMedTime",
                        np.where( (grp['UtilizationBase_'] == 'Hour3'), "WBLowTime",
                        np.where( (grp['UtilizationBase_'] == 'Hour5'), "WBLowTime",
                        np.where( (grp['UtilizationBase_'] == 'TwoDay'), "WBHighTime","Miss" ))))))))))))))))))))))

a = grp.groupby(['tickBehavClust','satisfFl']).agg({'Points_sum' : ['median'],
                                   'visualid_count':['median'],
                                   'timeSpent':['median'],
                                   'visualid_':['count']}).reset_index()

a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

a = grp.groupby(['tickBehavClust']).agg({'Points_sum' : ['median'],
                                   'visualid_count':['median'],
                                   'timeSpent':['median'],
                                   'visualid_':['count']}).reset_index()
                                   
a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')



a = grp.groupby(['satisfFl']).agg({'Points_sum' : ['median'],
                                   'visualid_count':['median'],
                                   'timeSpent':['median'],
                                   'visualid_':['count']}).reset_index()

a.columns = ["_".join(x) for x in a.columns.ravel()]
a['PercCust'] = (a['visualid__count'] / (a['visualid__count'].sum()))*100 
a.rename(columns = {'satisfFl_' : 'SatisfactionFlag'}, inplace = True)
a.index = a['SatisfactionFlag']
a[['PercCust']].T.plot.barh(stacked = True)
plt.grid(linestyle = ':', color ='grey')
plt.xlabel("Percentage Of The Population")
plt.title("Customer Satisfaction Flag Customer Proportion")
plt.text(1, 1, "35.93" ,wrap =True)
a.head(3)
a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#tidy = pd.melt(messy, id_vars='row', var_name='dimension', value_name='length')
#messy1 = tidy.pivot(index='row',columns='dimension',values='length')


#########STE4: Update the raw dataset with customer level such as satisfaction index, ticket bucket, timespent, ridescnt, pointsspent #########

a = grp
#VL stands for visualID level
a.columns =  (['VL_visualid', 'VL_accesscode', 'VL_UtilizationBase', 'VL_usetime_min','VL_usetime_max', 'VL_pointsSum', 
               'VL_rideCount', 'VL_timeSpent','VL_satisfFl', 'VL_tickBehavClust'])

b = df2.merge(a[['VL_visualid', 'VL_accesscode', 'VL_pointsSum','VL_rideCount','VL_timeSpent','VL_satisfFl','VL_tickBehavClust']],\
                how = 'left', left_on = ['visualid', 'accesscode'] , right_on = ['VL_visualid', 'VL_accesscode'] )


c = b.drop(b[['AccessCode','RideID','VL_visualid','VL_accesscode']],axis = 1)
a.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\visualIDLvlData.pickle')
c.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA2.pickle')

grp = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\visualIDLvlData.pickle')
df3 = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA2.pickle')

########### Now lets do some fancy EDA graphs
a = grp.groupby(['VL_tickBehavClust']).agg({'VL_pointsSum' : ['median'],
                                   'VL_rideCount':['median'],
                                   'VL_timeSpent':['median'],
                                   'VL_visualid':['count']}).reset_index()
                                   
a.columns = [ "_".join(x) for x in a.columns.ravel()]                                   

fig = plt.figure()
ax411 = plt.subplot(411)
width = 0.25
barcnt = np.arange(len(a['VL_tickBehavClust_']))
plt.bar( barcnt , a['VL_pointsSum_median'], width) 
plt.xticks(barcnt, a['VL_tickBehavClust_'], rotation = 0)
plt.ylabel('MedianPts')
plt.suptitle("Customer-Ticket-Group charachteristics")

ax412 = plt.subplot(412,  sharex = ax411)
width = 0.25
barcnt = np.arange(len(a['VL_tickBehavClust_']))
plt.bar( barcnt , a['VL_timeSpent_median'], width, color = 'b') 
plt.ylabel('MedianTime-Sec')

ax413 = plt.subplot(413, sharex = ax411)
width = 0.25
barcnt = np.arange(len(a['VL_tickBehavClust_']))
plt.bar( barcnt , a['VL_rideCount_median'], width, color = 'g') 
plt.ylabel('MedianRides')

ax414 = plt.subplot(414,  sharex = ax411)
width = 0.25
barcnt = np.arange(len(a['VL_tickBehavClust_']))
plt.bar( barcnt , a['VL_visualid_count'], width, color = 'c') 
plt.ylabel('CustCount')

plt.subplots_adjust(hspace =.0)
plt.show()

############## across customer-ticket-group understand the ride first, ride median and ride last pattern
df4 = df3[['usetime','accesscode','visualid','VL_tickBehavClust','VL_satisfFl','Name_y']].sort_values(['visualid','accesscode','usetime'])
lastride = df4.drop_duplicates( subset = ['visualid', 'accesscode'], keep = 'last')
firsride = df4.drop_duplicates( subset = ['visualid', 'accesscode'], keep = 'first')
allride = df4.drop_duplicates( subset =['visualid','accesscode', 'Name_y'], keep = 'first')

L = lastride.groupby(['VL_tickBehavClust','VL_satisfFl','Name_y']).size().reset_index
F = firsride.groupby(['VL_tickBehavClust','VL_satisfFl','Name_y']).size().reset_index
A = allride.groupby(['VL_tickBehavClust','VL_satisfFl','Name_y']).size().reset_index


L.shape
L.groupby()

a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')


#########################################################################

#22 Nov 2017

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use('ggplot')

#########STEP1: Import and collate the data provided by client ########################

usage1 = pd.read_excel("Usage1.xlsx");
usage2 = pd.read_excel("Usage2.xlsx");
usage3 = pd.read_excel("Usage3.xlsx");
usage4 = pd.read_excel("Usage4.xlsx");
usage5 = pd.read_excel("Usage5.xlsx");
usage6 = pd.read_excel("Usage6.xlsx");
usage7 = pd.read_excel("Usage7.xlsx");
usage8 = pd.read_excel("Usage8.xlsx");
usage9 = pd.read_excel("Usage9.xlsx");
usage10 = pd.read_excel("Usage10.xlsx");
usage11 = pd.read_excel("Usage11.xlsx");
usage12 = pd.read_excel("Usage12.xlsx");
usage13 = pd.read_excel("Usage13.xlsx");
usage14 = pd.read_excel("Usage14.xlsx");
usage15 = pd.read_excel("Usage15.xlsx");
usage16 = pd.read_excel("Usage16.xlsx");
usage17 = pd.read_excel("Usage17.xlsx");

# new files that were shared as fix
usageFix = pd.read_excel("UsageFix.xlsx");
usageFix2 = pd.read_excel("UsageFix2.xlsx");

usage1.shape
usage1.head
usage1.columns
usage1.dtypes

usage1['filename'] = 'usage1'
usage2['filename'] = 'usage2'
usage3['filename'] = 'usage3'
usage4['filename'] = 'usage4'
usage5['filename'] = 'usage5'
usage6['filename'] = 'usage6'
usage7['filename'] = 'usage7'
usage8['filename'] = 'usage8'
usage9['filename'] = 'usage9'
usage10['filename'] = 'usage10'
usage11['filename'] = 'usage11'
usage12['filename'] = 'usage12'
usage13['filename'] = 'usage13'
usage14['filename'] = 'usage14'
usage15['filename'] = 'usage15'
usage16['filename'] = 'usage16'
usage17['filename'] = 'usage17'

# new files that were shared as fix
usageFix['filename'] = 'usageFix'
usageFix2['filename'] = 'usageFix2'

print({},format(usage1.shape))
print({},format(usage2.shape))
print({},format(usage3.shape))
print({},format(usage4.shape))
print({},format(usage5.shape))
print({},format(usage6.shape))
print({},format(usage7.shape))
print({},format(usage8.shape))
print({},format(usage9.shape))
print({},format(usage10.shape))
print({},format(usage11.shape))
print({},format(usage12.shape))
print({},format(usage13.shape))
print({},format(usage14.shape))
print({},format(usage15.shape))
print({},format(usage16.shape))
print({},format(usage17.shape))


usagedatacoll =  usage1.append([usage2,usage3,usage4,usage5,usage6,usage7,usage8,usage9,
          usage10,usage11,usage12,usage13,usage14,usage15,usage16,usage17],
          ignore_index = True)
          
usagedatacoll.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA.pickle')
#del usagedatacoll
data = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA.pickle');


data1 = data[~data['usageid'].isin(usageFix2['usageid'])]
data2 = data1.append([usageFix2],ignore_index = True)

data2.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA1.pickle')



#########STEP2: Import the cleaned data and conduct some basic EDA ########################

#new data 
data_new = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA1.pickle');
df = data_new.drop_duplicates(subset = ['usetime','accesscode','visualid','rideid','useno'] ,keep ='last')
df['date']  = df['usetime'].dt.date
df['time'] = df['usetime'].dt.time
df['hour'] = df['usetime'].dt.hour

#import tickets
ticket = pd.read_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\D_TicketsAttributes.csv') 
ride =  pd.read_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\D_RidesAttributes_v2.csv')

# join ride attributes and ticket attributes to the df file
df1 = df.merge(ticket[['AccessCode','Name','Type','Category','ParetoFl', 'UtilizationBase','Price','Tim_Pt_Fl']], \
how = 'left', left_on = ['accesscode'] , right_on = ['AccessCode'], )

df2 = df1.merge(ride[['RideID','Name','Points', 'RideType','MinHeight','Direction', 'RideIntensityRating', \
'RollerCoasterFL','RideDurCliSec','RideCapacity']], how = 'left',left_on = ['rideid'] , right_on = ['RideID'])

################# EDA Codes

#see accescodes freauency
df.groupby('accesscode').size().reset_index(name = 'rowcount').to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#Solved encoding error while exporting to csv
#a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv',sep='\t', encoding='utf-8')

#see rides freauency
df.groupby('rideid').size().reset_index(name = 'rowcount').to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#See hour level ride frequency
df.groupby(['hour']).size()
df.head(2)

# check for the frequency of the ride

a = df2.groupby(['visualid']).size().reset_index(name = 'rowcount')
a.groupby(['rowcount']).size().to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

b = df2.merge(a, how = 'left', left_on = ['visualid'], right_on = ['visualid'])
b.groupby(['rowcount','Name_x']).size().reset_index(name = "countrides"). \
to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

c = b.drop_duplicates(subset =['visualid','rowcount','UtilizationBase'] ,keep = 'last')
c.groupby(['rowcount','UtilizationBase']).size().reset_index(name = "countrides"). \
to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#########STEP3: Revolutionary customer satisfaction Indicator calculation ########################

grp = df2.groupby(['visualid','accesscode','UtilizationBase']).agg({'visualid':"count",
                                                                    'usetime':[min,max],
                                                                    'Points': [sum]}).reset_index()
grp.columns = ["_".join(x) for x in grp.columns.ravel() ]
grp['timeSpent'] = (grp['usetime_max'] - grp['usetime_min']).dt.seconds

distr = grp.groupby(['UtilizationBase_'])[['visualid_count','Points_sum','timeSpent']]. \
quantile([0.0,.002,0.023,0.159,0.25, 0.5, 0.75, 0.841,0.977,0.998,1.0],).reset_index()

distr.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

grp.groupby(['UtilizationBase_']).size().to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#Create satisfaction flag for all the customers

grp['satisfFl'] = np.where( (grp['UtilizationBase_'] == '1') & (grp['Points_sum'] >= 3), "VerySat", 
                  np.where( (grp['UtilizationBase_'] == '1') & (grp['Points_sum'] < 3), "NotSat", 
                  np.where( (grp['UtilizationBase_'] == '3') & (grp['Points_sum'] >= 3), "VerySat", 
                  np.where( (grp['UtilizationBase_'] == '3') & (grp['Points_sum'] < 3), "NotSat", 
                  np.where( (grp['UtilizationBase_'] == '6') & (grp['Points_sum'] >= 6), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '6') & (grp['Points_sum'] < 6), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '9') & (grp['Points_sum'] >= 9), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '9') & (grp['Points_sum'] < 9 ), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '18') & (grp['Points_sum'] >= 18), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '18') & (grp['Points_sum'] < 18), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '19') & (grp['Points_sum'] >= 19), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '19') & (grp['Points_sum'] < 19), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '20') & (grp['Points_sum'] >= 20), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '20') & (grp['Points_sum'] < 20), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '25') & (grp['Points_sum'] >= 25), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '25') & (grp['Points_sum'] < 25), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '30') & (grp['Points_sum'] >= 30), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '30') & (grp['Points_sum'] < 30), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '34') & (grp['Points_sum'] >= 34), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '34') & (grp['Points_sum'] < 34), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '35') & (grp['Points_sum'] >= 34), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '35') & (grp['Points_sum'] < 34), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '48') & (grp['Points_sum'] >= 48), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '48') & (grp['Points_sum'] < 48) & (grp['Points_sum'] >= 42), "Sat",
                  np.where( (grp['UtilizationBase_'] == '48') & (grp['Points_sum'] < 42) , "NotSat",
                  np.where( (grp['UtilizationBase_'] == '49') & (grp['Points_sum'] >= 48), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '49') & (grp['Points_sum'] < 48) & (grp['Points_sum'] >= 42), "Sat",
                  np.where( (grp['UtilizationBase_'] == '49') & (grp['Points_sum'] < 42), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '50') & (grp['Points_sum'] >= 48), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '50') & (grp['Points_sum'] < 48) & (grp['Points_sum'] >= 42), "Sat",
                  np.where( (grp['UtilizationBase_'] == '50') & (grp['Points_sum'] < 42), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '84') & (grp['Points_sum'] >= 84), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '84') & (grp['Points_sum'] < 84) & (grp['Points_sum'] >= 72), "Sat",
                  np.where( (grp['UtilizationBase_'] == '84') & (grp['Points_sum'] < 72), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '85') & (grp['Points_sum'] >= 84), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '85') & (grp['Points_sum'] < 84) & (grp['Points_sum'] >= 72), "Sat",
                  np.where( (grp['UtilizationBase_'] == '85') & (grp['Points_sum'] < 72), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '96') & (grp['Points_sum'] >= 96), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '96') & (grp['Points_sum'] < 96) & (grp['Points_sum'] >= 72), "Sat",
                  np.where( (grp['UtilizationBase_'] == '96') & (grp['Points_sum'] < 72), "NotSat",
                  np.where( (grp['UtilizationBase_'] == '250') & (grp['Points_sum'] >= 34), "VerySat",
                  np.where( (grp['UtilizationBase_'] == '250') & (grp['Points_sum'] < 34), "NotSat",
                  np.where( (grp['UtilizationBase_'] == 'AllDay') & (grp['Points_sum'] >= 72), "VerySat",
                  np.where( (grp['UtilizationBase_'] == 'AllDay') & (grp['Points_sum'] < 72) & (grp['Points_sum'] >= 36), "Sat",
                  np.where( (grp['UtilizationBase_'] == 'AllDay') & (grp['Points_sum'] <36), "NotSat",
                  np.where( (grp['UtilizationBase_'] == 'Hour3') & (grp['Points_sum'] >= 54), "VerySat",
                  np.where( (grp['UtilizationBase_'] == 'Hour3') & (grp['Points_sum'] < 54) & (grp['Points_sum'] >= 33), "Sat",
                  np.where( (grp['UtilizationBase_'] == 'Hour3') & (grp['Points_sum'] < 33), "NotSat",
                  np.where( (grp['UtilizationBase_'] == 'Hour5') & (grp['Points_sum'] >= 66), "VerySat",
                  np.where( (grp['UtilizationBase_'] == 'Hour5') & (grp['Points_sum'] < 66) & (grp['Points_sum'] >= 39), "Sat",
                  np.where( (grp['UtilizationBase_'] == 'Hour5') & (grp['Points_sum'] < 39), "NotSat",
                  np.where( (grp['UtilizationBase_'] == 'TwoDay') & (grp['Points_sum'] >= 153), "VerySat",
                  np.where( (grp['UtilizationBase_'] == 'TwoDay') & (grp['Points_sum'] < 153) & (grp['Points_sum'] >= 81), "Sat",
                  np.where( (grp['UtilizationBase_'] == 'TwoDay') & (grp['Points_sum'] < 81), "NotSat","miss"
                  ))))))))))))))))))))))))))))))))))))))))))))))))))))))  


a = grp.groupby(['UtilizationBase_','satisfFl']).size().reset_index(name = 'countOfViewerid')
a.pivot(index = 'UtilizationBase_', columns = 'satisfFl', values = 'countOfViewerid').reset_index().to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

a = grp.groupby(['satisfFl']).agg({'Points_sum' : ['mean'],
                                   'visualid_count':['mean'],
                                   'timeSpent':['mean']}).reset_index()
a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')


#add ticket final buckets 

grp['tickBehavClust'] = np.where( (grp['UtilizationBase_'] == '1'), "PointsLow", 
                        np.where( (grp['UtilizationBase_'] == '3'), "PointsLow", 
                        np.where( (grp['UtilizationBase_'] == '6'), "PointsLow", 
                        np.where( (grp['UtilizationBase_'] == '9'), "PointsLow", 
                        np.where( (grp['UtilizationBase_'] == '18'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '19'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '20'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '25'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '30'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '34'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '35'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == '48'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '49'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '50'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '84'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '85'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '96'), "PointsVeryHigh",
                        np.where( (grp['UtilizationBase_'] == '250'), "PointsMed",
                        np.where( (grp['UtilizationBase_'] == 'AllDay'), "WBMedTime",
                        np.where( (grp['UtilizationBase_'] == 'Hour3'), "WBLowTime",
                        np.where( (grp['UtilizationBase_'] == 'Hour5'), "WBLowTime",
                        np.where( (grp['UtilizationBase_'] == 'TwoDay'), "WBHighTime","Miss" ))))))))))))))))))))))

a = grp.groupby(['tickBehavClust','satisfFl']).agg({'Points_sum' : ['median'],
                                   'visualid_count':['median'],
                                   'timeSpent':['median'],
                                   'visualid_':['count']}).reset_index()

a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

a = grp.groupby(['tickBehavClust']).agg({'Points_sum' : ['median'],
                                   'visualid_count':['median'],
                                   'timeSpent':['median'],
                                   'visualid_':['count']}).reset_index()
                                   
a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')



a = grp.groupby(['satisfFl']).agg({'Points_sum' : ['median'],
                                   'visualid_count':['median'],
                                   'timeSpent':['median'],
                                   'visualid_':['count']}).reset_index()

a.columns = ["_".join(x) for x in a.columns.ravel()]
a['PercCust'] = (a['visualid__count'] / (a['visualid__count'].sum()))*100 
a.rename(columns = {'satisfFl_' : 'SatisfactionFlag'}, inplace = True)
a.index = a['SatisfactionFlag']
a[['PercCust']].T.plot.barh(stacked = True)
plt.grid(linestyle = ':', color ='grey')
plt.xlabel("Percentage Of The Population")
plt.title("Customer Satisfaction Flag Customer Proportion")
plt.text(1, 1, "35.93" ,wrap =True)
a.head(3)
a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

#tidy = pd.melt(messy, id_vars='row', var_name='dimension', value_name='length')
#messy1 = tidy.pivot(index='row',columns='dimension',values='length')


#########STE4: Update the raw dataset with customer level such as satisfaction index, ticket bucket, timespent, ridescnt, pointsspent #########

a = grp
#VL stands for visualID level
a.columns =  (['VL_visualid', 'VL_accesscode', 'VL_UtilizationBase', 'VL_usetime_min','VL_usetime_max', 'VL_pointsSum', 
               'VL_rideCount', 'VL_timeSpent','VL_satisfFl', 'VL_tickBehavClust'])

b = df2.merge(a[['VL_visualid', 'VL_accesscode', 'VL_pointsSum','VL_rideCount','VL_timeSpent','VL_satisfFl','VL_tickBehavClust']],\
                how = 'left', left_on = ['visualid', 'accesscode'] , right_on = ['VL_visualid', 'VL_accesscode'] )


c = b.drop(b[['AccessCode','RideID','VL_visualid','VL_accesscode']],axis = 1)
a.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\visualIDLvlData.pickle')
c.to_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA2.pickle')

grp = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\visualIDLvlData.pickle')
df3 = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA2.pickle')

########### Now lets do some fancy EDA graphs
a = grp.groupby(['VL_tickBehavClust']).agg({'VL_pointsSum' : ['median'],
                                   'VL_rideCount':['median'],
                                   'VL_timeSpent':['median'],
                                   'VL_visualid':['count']}).reset_index()
                                   
a.columns = [ "_".join(x) for x in a.columns.ravel()]                                   

fig = plt.figure()
ax411 = plt.subplot(411)
width = 0.25
barcnt = np.arange(len(a['VL_tickBehavClust_']))
plt.bar( barcnt , a['VL_pointsSum_median'], width) 
plt.xticks(barcnt, a['VL_tickBehavClust_'], rotation = 0)
plt.ylabel('MedianPts')
plt.suptitle("Customer-Ticket-Group charachteristics")

ax412 = plt.subplot(412,  sharex = ax411)
width = 0.25
barcnt = np.arange(len(a['VL_tickBehavClust_']))
plt.bar( barcnt , a['VL_timeSpent_median'], width, color = 'b') 
plt.ylabel('MedianTime-Sec')

ax413 = plt.subplot(413, sharex = ax411)
width = 0.25
barcnt = np.arange(len(a['VL_tickBehavClust_']))
plt.bar( barcnt , a['VL_rideCount_median'], width, color = 'g') 
plt.ylabel('MedianRides')

ax414 = plt.subplot(414,  sharex = ax411)
width = 0.25
barcnt = np.arange(len(a['VL_tickBehavClust_']))
plt.bar( barcnt , a['VL_visualid_count'], width, color = 'c') 
plt.ylabel('CustCount')

plt.subplots_adjust(hspace =.0)
plt.show()

############## across customer-ticket-group understand the ride first, ride median and ride last pattern
df4 = df3[['usetime','accesscode','visualid','VL_tickBehavClust','VL_satisfFl','Name_y']].sort_values(['visualid','accesscode','usetime'])
lastride = df4.drop_duplicates( subset = ['visualid', 'accesscode'], keep = 'last')
firsride = df4.drop_duplicates( subset = ['visualid', 'accesscode'], keep = 'first')
allride = df4.drop_duplicates( subset =['visualid','accesscode', 'Name_y'], keep = 'first')

L = lastride.groupby(['VL_tickBehavClust','VL_satisfFl','Name_y']).size().reset_index(name  = 'custcnt')
F = firsride.groupby(['VL_tickBehavClust','VL_satisfFl','Name_y']).size().reset_index(name = 'custcnt')
A = allride.groupby(['VL_tickBehavClust','VL_satisfFl','Name_y']).size().reset_index(name = 'custcnt')

L['rank'] = L.groupby(['VL_tickBehavClust','VL_satisfFl'])['custcnt'].rank(method = 'first',ascending = False )
F['rank'] = F.groupby(['VL_tickBehavClust','VL_satisfFl'])['custcnt'].rank(method = 'dense',ascending = False )
A['rank'] = A.groupby(['VL_tickBehavClust','VL_satisfFl'])['custcnt'].rank(method = 'dense',ascending = False )


L.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')
F.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')
A.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

############ . Lets c which rides gets highest visitors, highest rides, /
###### repeats, repeats highest value to Nickelodeon universe

a = df3.groupby(['visualid','rideid','Name_x']).size().reset_index(name = 'repeat')
a['repeat_fl'] = np.where(a['repeat'] > 1, 1,0)
a = a.groupby(['rideid','Name_x']).agg({'repeat_fl' : ['sum']}).reset_index()
a.columns = [ "_".join(x) for x in a.columns.ravel() ]
a.columns = ['rideid','RideName','repeatCust']

b = df3.groupby(['rideid']).agg({ 'visualid' :['count', 'nunique'],
                               'Points' : ['sum']
        }).reset_index()

b.columns = [ '_'.join(x) for x in b.columns.ravel()]

c = b.merge(a , how = 'left', left_on = 'rideid_', right_on = 'rideid').drop('rideid',1)
c.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')


################## For Predictive Project find TimeRange for each day for each year

df3 = pd.read_pickle('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\rawDataMOA2.pickle')

a = df3.groupby(['date']).agg({'usetime':['min','max'],
                        'rideid':['nunique']}).reset_index()

a.columns = ['_'.join(x) for x in a.columns.ravel()]
a['timeSpent'] = ((a['usetime_max'] - a['usetime_min']).dt.seconds)/3600.0
a.to_csv('D:\\FT2 - Team 7\\Data\\Exploratory codes and data\\Ashlesh\\test.csv')

df3.columns
df3.head(2)
