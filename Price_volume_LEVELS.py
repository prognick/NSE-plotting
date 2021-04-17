def gbm(beg_days,end_days,param1,param2):
    #beg_days = how many days to substract from today to get the analysis beginning date
    #end_days = how many days to substract from today to get analysis end date
    #how many days to go back
    

    import pandas as pd
    from datetime import date,datetime , timedelta    
    from nsepy import get_history,get_index_pe_history
    
    from matplotlib.backends.backend_pdf import PdfPages
    
    
    
    import numpy as np
    import pandas as pd
    
    from sklearn.tree import DecisionTreeRegressor
    import matplotlib.pyplot as plt
    
    #get todays date    
    date1 = datetime.today().date()
    
    
    #create start date    
    st_dt1 = date1 - timedelta(beg_days)
    y1 = st_dt1.year
    m1 = st_dt1.month
    d1 = st_dt1.day    
    print (st_dt1)
    
    
    #create end date    
    end_dt1 = date1 - timedelta(end_days) 
    y2 = end_dt1.year
    m2 = end_dt1.month
    d2 = end_dt1.day
    
    if param1 == 'stock':
        ################FOR STOCKS
        data_stock = get_history(symbol=param2, end=date(y1,m1,d1), start=date(y2,m2,d2))       
        data_stock["DMA_20"] = pd.Series(data_stock["Close"]).rolling(20).mean()
        data_stock["DMA_100"] = pd.Series(data_stock["Close"]).rolling(100).mean()
        #df_short = data_stock
        data_stock.to_csv("stock.csv")
        print ("20 DMA is placed at"  , data_stock['DMA_20'].tail(1).values)
        print ("Last closing at"  , data_stock['Close'].tail(1).values)
        
        #return
    else:
        ##############FOR INDEX
        nifty_spot = get_history(symbol= param2, end=date(y1,m1,d1), start=date(y2,m2,d2),index = True)
        nifty_spot["DMA_20"] = pd.Series(nifty_spot["Close"]).rolling(20).mean()
        nifty_spot["DMA_100"] = pd.Series(nifty_spot["Close"]).rolling(100).mean()
        #nifty_mid = get_history(symbol= 'NIFTY MIDCAP 100', end=date(y1,m1,d1), start=date(y2,m2,d2),index = True)
        #nifty_sml = get_history(symbol= 'NIFTY SMLCAP 100', end=date(y1,m1,d1), start=date(y2,m2,d2),index = True)

        nifty_spot.to_csv("index.csv")
        print ("20 DMA is placed at"  , nifty_spot['DMA_20'].tail(1).values)
        print ("Last closing at"  , nifty_spot['Close'].tail(1).values)
        #nifty_mid.to_csv("nifty_mid.csv")
        #nifty_sml.to_csv("nifty_sml.csv")
        #return

    if param1 == 'stock':
        df = pd.read_csv('stock.csv')
    else:
        df = pd.read_csv('index.csv')
    #Take a smaller portion of the dataframe
    #df_short = df.head(200)   
    
    df_short = df    
    
    if param1 == 'stock':
        pp = supp_rest_stock(df_short)
    else:
        pp = supp_rest_index(df_short)
    

    #df_short.to_csv("base_data.csv")
    #df_short = df
    #print (len(pp))
    
     
    
    
    #CREATING COLUMNS IN THE DATASET FOR EACH CRITICAL LEVEL
    i = 0 
    for m in pp:
        i = i + 1
        df_short['level'+str(i)] = m
    
    #JUST PLOTTING THE LEVELS AS STRAIGHT LINES AND CLOSE PRICE
    i = 1
    
    while i <= len(pp):
        plt.plot(df_short['Date'], df_short['level'+str(i)])
        i = i + 1
    plt.plot(df_short['Date'], df_short['Close'])
    plt.plot(df_short['Date'], df_short['DMA_20'],color = 'k')
    plt.plot(df_short['Date'], df_short['DMA_100'],'--',color = 'r')

    plt.yticks(np.arange(df_short['Close'].min(), df_short['Close'].max(), (df_short['Close'].max() - df_short['Close'].min())/15))    
    plt.xticks([], [])
    
    #fig_size = plt.rcParams["figure.figsize"]
    #fig_size[0] = 10
    #fig_size[1] = 10
    #plt.rcParams["figure.figsize"] = fig_size
    
    
    #plt.show()
    
    
    #CREATING HEAVY VOLUME LEVELS 
    vol_prices = vol_lvls(df_short)
    
    #CREATE VOLUME LEVELS FOR PLOTTING AND RETAIN ONLY THE LAST DATE
    i = 0 
    for m in vol_prices:
        i = i + 1
        df_short['Vol_level'+str(i)] = m
    last_df = df_short.tail(1)
    
    #JUST PLOT THE VOLUMES AS A SINGLE LINE
    i = 1
    while i <= len(vol_prices):
        plt.plot(last_df['Date'], last_df['Vol_level'+str(i)],'_',color = 'k')
        i = i + 1
        
    
    plt.show()
    
    
    
    return

# =============================================================================
#         
#     
#         print ("critical_level is" , m)
#         df_short['dist_from_m'] = (np.abs(df_short['Close'] - m)/m)*100        
#         df_short['closeness_ind'] = np.where(df_short['dist_from_m'] <= 0.5,1,0)
#         
#         df_short['rct_pr_trnd'] = np.where(df_short['Close'].shift(5) > df_short['Close'],0,1 )
#         
#         df_short['tgt'] = (df_short['Close'].shift(-5) - df_short['Close'])/ df_short['Close']
# 
#         t = df_short.groupby(['closeness_ind','rct_pr_trnd'])['tgt'].mean()*100
#         s = df_short.groupby(['closeness_ind','rct_pr_trnd'])['tgt'].count()
#         print (t , s)
#         
# =============================================================================
        #df_short.to_csv("out.csv")
        #return
    
    
    return 

    
    ####MAKE RSI########################################
    
    RSI_lk_bk = 14

    df_short['abs_move'] = df_short['Close'] - df_short['Prev Close']
    df_short['RSI_gain'] = np.where(df_short['abs_move'] > 0 ,df_short['abs_move'],0)
    df_short['RSI_loss'] = np.where(df_short['abs_move'] <= 0 ,-1*df_short['abs_move'],0)
       
    
    df_short["RSI_avg_gain"] = pd.Series(df_short["RSI_gain"]).rolling(RSI_lk_bk).mean()
    df_short["RSI_avg_loss"] = pd.Series(df_short["RSI_loss"]).rolling(RSI_lk_bk).mean()
    
    df_short['RSI_RS'] = df_short["RSI_avg_gain"] / df_short["RSI_avg_loss"]
    df_short['RSI_act'] = np.where(df_short["RSI_avg_loss"] == 0 , 100,(100-(100/(1+df_short['RSI_RS']))))
    
    ###Smooth out RSI a little bit
    df_short["RSI_fin"] = pd.Series(df_short["RSI_act"]).rolling(5).mean()
    
    ###define if RSI is increasing or decreasing
    df_short['RSI_trend'] = np.where(df_short['RSI_fin'].shift(5) > df_short['RSI_fin'],0,1 )
    
    
    
    bins = [0, 30,40, 50,60, 70,80,90,100]
    df_short['binned'] = pd.cut(df_short['RSI_fin'], bins)
    ##define tgt i.e. 5 day returns
    df_short['tgt'] = (df_short['Close'].shift(-14) - df_short['Close'])/ df_short['Close']
    
    t = df_short.groupby(['binned','RSI_trend'])['tgt'].mean()*100
    s = df_short.groupby(['binned','RSI_trend'])['tgt'].count()
    print (t ,s)
    

    

    df_short.to_csv("out1.csv")
    return

    
    
    
   
   
    
    
    
    #################################################################################
    #######################INDIVIDUAL MODULES HERE###################################
    ###############################################################################
def supp_rest_stock(df_1):
    
    import pandas as pd
    from datetime import date,datetime , timedelta    
    from nsepy import get_history,get_index_pe_history
    
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import pandas as pd
    
    from sklearn.tree import DecisionTreeRegressor
    import matplotlib.pyplot as plt
    
    
    #HERE WE ARE TRYING TO FIND THE SUPPORT AND RESISTANCE LEVELS
    #Count number of up/low days from any given day and hence any given closing price
    df_1['mvmt'] = df_1['Close'] - df_1['Close'].shift(1)
    df_1['perc_move_abs'] = (abs(df_1['mvmt'] / df_1['Close'].shift(1)))*100
    
    avg_daily_move = df_1['perc_move_abs'].mean()
    
    print (" avg_daily_moves" , avg_daily_move)
    


    df_1['up_down'] = np.where(df_1['mvmt'] > 0 , 1,0)
    
    i = 0
    t = len(df_1)
    
    #over how many days you want to see the price bounce back or go down
    r = 5
    
    df_lvl = pd.DataFrame(columns = ['date','close','indicator'])
    
       
    while i < (t-r):
        df_int = df_1.iloc[i:i+r,:]
        a = np.sum(df_int['up_down'],axis = 0)
        b = float(a/r)
        
        df_alt = df_int.head(1)        
        
        data = [[df_alt.iloc[0,0],df_alt.iloc[0,8],b] ]
           
            
        df_alt1 = pd.DataFrame(data, columns = ['date','close', 'indicator'])
        
        df_lvl = df_lvl.append(df_alt1)
        i = i + 1
        
    
        
    df_critical_lvls = df_lvl[df_lvl['indicator'].isin([0,1,0.2,0.8])]    
    df_critical_lvls = df_critical_lvls.sort_values(by=['close'])    
    df_critical_lvls['gain'] = ((df_critical_lvls['close'] / df_critical_lvls['close'].shift(1))-1)*100
    
    i = 1 
    m = np.array([])
    
    pivot_price = np.array([])
    
    accu = 0    
    while i < len(df_critical_lvls):
                
        accu = accu + df_critical_lvls.iloc[i,3]
        m = np.append(m,df_critical_lvls.iloc[i,1])
        
        if accu > 5*avg_daily_move:
            m_1 = m[:-1]
            pivot_price = np.append(pivot_price,(np.mean(m_1)))  
                
            accu = 0
            m = m[-1]
        
        i = i + 1
        
    
    print ("Critical price action points   " , pivot_price)
    return pivot_price


def supp_rest_index(df_1):
    
    import pandas as pd
    from datetime import date,datetime , timedelta    
    from nsepy import get_history,get_index_pe_history
    
    
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import pandas as pd
    
    from sklearn.tree import DecisionTreeRegressor
    import matplotlib.pyplot as plt
    
    
    #HERE WE ARE TRYING TO FIND THE SUPPORT AND RESISTANCE LEVELS
    #Count number of up/low days from any given day and hence any given closing price
    df_1['mvmt'] = df_1['Close'] - df_1['Close'].shift(1)
    df_1['perc_move_abs'] = (abs(df_1['mvmt'] / df_1['Close'].shift(1)))*100
    
    avg_daily_move = df_1['perc_move_abs'].mean()
    
    print (" avg_daily_moves" , avg_daily_move)
    


    df_1['up_down'] = np.where(df_1['mvmt'] > 0 , 1,0)
    
    i = 0
    t = len(df_1)
    
    #over how many days you want to see the price bounce back or go down
    r = 5
    
    df_lvl = pd.DataFrame(columns = ['date','close','indicator'])
    
       
    while i < (t-r):
        df_int = df_1.iloc[i:i+r,:]
        a = np.sum(df_int['up_down'],axis = 0)
        b = float(a/r)
        
        df_alt = df_int.head(1) 
        
    
        
        data = [[df_alt.iloc[0,0],df_alt.iloc[0,4],b] ]
           
            
        df_alt1 = pd.DataFrame(data, columns = ['date','close', 'indicator'])
            
        df_lvl = df_lvl.append(df_alt1)
        i = i + 1
        
    
        
    df_critical_lvls = df_lvl[df_lvl['indicator'].isin([0,1,0.2,0.8])]    
    df_critical_lvls = df_critical_lvls.sort_values(by=['close'])    
    df_critical_lvls['gain'] = ((df_critical_lvls['close'] / df_critical_lvls['close'].shift(1))-1)*100
    
    i = 1 
    m = np.array([])
    
    pivot_price = np.array([])
    
    accu = 0    
    while i < len(df_critical_lvls):
                
        accu = accu + df_critical_lvls.iloc[i,3]
        m = np.append(m,df_critical_lvls.iloc[i,1])
        
        if accu > 3*avg_daily_move:
            m_1 = m[:-1]
            pivot_price = np.append(pivot_price,(np.mean(m_1)))  
                
            accu = 0
            m = m[-1]
        
        i = i + 1
        
    print ("Critical price action points   " , pivot_price)
    
    return pivot_price

def vol_lvls(df_short):
    ###IDEAS FOR VOLUME BASED CRITICAL LELVELS
    avg_vol = df_short['Volume'].mean()
    std_dev = df_short['Volume'].std()
    tgt_vol = 2*std_dev + avg_vol
    
    df_1_1 = df_short[df_short['Volume'] >= tgt_vol]
    
    #print(avg_vol , std_dev,tgt_vol)
    t = (df_1_1['Close']).values
    
    s = (df_1_1[['Date','Close']])
    s = s.sort_values(by =['Close'])
    s = s.values
    
    print ("Critical levels basis volume moves   ", s)
    return t


    
    
    