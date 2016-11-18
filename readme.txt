How to run the first.R

Code is essentially to predict the time series behavior, without any other features!
It is advantageous to take input from as many years of the past as possible

1.Code uses input train2010.csv to train2014.csv as input of 4 years and train2015.csv to test the predictions

2. stl decomposes the time series to 3 stationary time series, trend, seasonality and remainder

3. All three are predicted using nnetar function, the function uses nnet to predict n+1th output using past 10 inputs

4.The plots are self-generated.