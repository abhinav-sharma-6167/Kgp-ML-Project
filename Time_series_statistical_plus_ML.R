#setting the working dir
setwd("C:/Final year/Projects/ML")

#nnetar
#Feed-forward neural networks with a single hidden layer and lagged inputs for forecasting univariate time series.
#Libraries
library(data.table)
library(ggplot2)
library(plyr)
library(xgboost)
library(nnet)
library(xts)
library(forecast)
library(devtools)
library(ggfortify)

#Seed & Read
set.seed(100)
train <- fread("train2010.csv") #data of 2010
train2 <- fread("train2011.csv") #data of 2011
train3 <- fread("train2012.csv") #data of 2012
train4 <- fread("train2013.csv") #data of 2013
train <- rbind(train,train2,train3,train4)
train$Date <- as.Date(train$Date,format = "%d-%B-%Y")

#Making individul time series
pc <- ts(train$`Prev Close`, frequency = 252)
op <- ts(train$`Open Price`, frequency = 252)
hp <- ts(train$`High Price`, frequency = 252)
lp <- ts(train$`Low Price`, frequency = 252)
ap <- ts(train$`Average Price`, frequency = 252)
cp <- ts(train$`Close Price`, frequency = 252)


train[,`Last Price`:=NULL]


#Getting plots of the time series
train <-train[,colnames(train)[-c(5)]:= NULL,with=F]
train_ts <- ts(train, frequency = 252)
autoplot(train_ts,facets = FALSE)
#plot.ts(train$`Open Price`)


# 
# p <- ggplot() + 
#   geom_line(data = train, aes(x = Date, y = `Open Price`, color = "red")) +
#   geom_line(data = train, aes(x = Date, y = `Close Price`, color = "blue"))  +
#   xlab('Data_date') +
#   ylab('Stock_Price')

#Decomposing time series to individual components
train_ts_components = stl(cp,s.window = "periodic")

seasonal = train_ts_components$time.series[,1]
trend = train_ts_components$time.series[,2]
remainder = train_ts_components$time.series[,3]

#getting time series of trend, seasonality and remainder
train_ts_adjusted_r = train_ts - seasonal - trend
train_ts_adjusted_s = train_ts - trend - remainder
train_ts_adjusted_t = train_ts - seasonal - remainder


fit_r = nnetar(train_ts_adjusted_r)
fcast_r <- forecast(fit_r,h=244)
plot(fcast_r)
lines(fitted(fit_r),col="red")

fit_t = nnetar(train_ts_adjusted_t)
fcast_t <- forecast(fit_t,h=244)
plot(fcast_t)
lines(fitted(fit_t),col="red")

fit_s = nnetar(train_ts_adjusted_s)
fcast_s <- forecast(fit_s,h=244)
plot(fcast_s)
lines(fitted(fit_s),col="red")


fcast_final = fcast_r$mean + fcast_t$mean + fcast_s$mean
values = as.numeric(fcast_final)
values = as.data.frame(values)

values <- ts(values)
autoplot(values)

output <- fread("train2014.csv")
output$Date <- as.Date(output$Date,format = "%d-%B-%Y")

output <- output$`Close Price`
plot.ts(output)

values <- as.data.table(values)
#output <- ts(output)
values$output <- output
values <- ts(values)
autoplot(values)

################################################################################
################################################################################
######################RED IS PREDICTED,BLUE IS ACTUAL###########################
################################################################################
################################################################################
