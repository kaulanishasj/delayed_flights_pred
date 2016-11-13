#!/usr/bin/env Rscript

options(error=traceback)

if(!("plyr" %in% installed.packages())) install.packages("plyr")
library(plyr)
if(!("ggplot2" %in% installed.packages())) install.packages("ggplot2")
library(ggplot2)
if(!("ggthemes" %in% installed.packages())) install.packages("ggthemes")
library(ggthemes)
if(!("reshape2" %in% installed.packages())) install.packages("reshape2")
library(reshape2)
if(!("ROCR" %in% installed.packages())) install.packages("ROCR")
library(ROCR)
if(!("e1071" %in% installed.packages())) install.packages("e1071")
library(e1071)
if(!("reshape2" %in% installed.packages())) install.packages("reshape2")
library(reshape2)
if (!("MASS" %in% installed.packages())) install.packages("MASS")
library(MASS)
if (!("leaps" %in% installed.packages())) install.packages("leaps")
library(leaps)
if (!("lme4" %in% installed.packages())) install.packages("lme4")
library(lme4)
if (!("pbkrtest" %in% installed.packages())) install.packages("pbkrtest_0.4-6.zip")
library(pbkrtest)
if (!("caret" %in% installed.packages())) install.packages("caret")
library(caret)
if (!("randomForest" %in% installed.packages())) install.packages("randomForest")
library(randomForest)
if (!("pROC" %in% installed.packages())) install.packages("pROC")
library(pROC)
if (!("doParallel" %in% installed.packages())) install.packages("doParallel")
library(doParallel)

registerDoParallel(4)

set.seed(42)

# utility function for import from csv file
import.csv = function(filename) {
  return(read.csv(filename, sep = ',', header = T))
}

##############################################################################

# utility function for export to csv file
write.csv = function(ob, filename) {
  write.table(ob, filename, quote = T, sep = ',', row.names = F)
}

##############################################################################
## Prediction models
##############################################################################

##############################################################################
#Get predictions using Random Forest
###############################################################################
get.pred.rf = function(train,test,params=c(500)) {

  nf = dim(train)[2]
  output = colnames(train)[nf]
  formula = as.formula(paste(output, '~ .'))
  actual = test[nf]

  best.roc = 0
  best.param = 0
  best.fit = 0
  for(param in params) {

      print('')
      print(paste('Fitting RandomForest with ',param,'trees'))
      train.ctrl = trainControl(method="cv", number = 5, summaryFunction=twoClassSummary, classProb=T)
      fit = train(formula, data=data, method="rf", trControl=train.ctrl, metric="ROC", ntree=param)

      print(fit$times$everything['elapsed'])

      print(paste('Number features selected',fit$bestTune$mtry))
      roc = subset(fit$results, mtry == fit$bestTune$mtry)$ROC
      print(paste('AUC for model fit', roc))

      if(roc > best.roc) {
        best.roc = roc
        best.param = param
        best.fit = fit
      }

      pred = as.data.frame(as.numeric(predict(fit, test, type = 'prob')[,2]))
      prediction = prediction(pred, actual)
      auc.val = round(unlist(slot(performance(prediction,'auc'), 'y.values')),2)
      print(paste('AUC for predicted fit',auc.val))
    }


  #predicted probabilities are for the second factor level
  pred = as.data.frame(as.numeric(predict(best.fit, test, type = 'prob')[,2]))

  result = cbind(pred, actual)
  return(result)
}


##############################################################################
#Get predictions using Logistic Regression classifier (implementation: glm with binomial(logit) model)
###############################################################################

get.pred.logreg = function(train,test,params) {

  nf = dim(train)[2]
  output = colnames(train)[nf]
  formula = as.formula(paste(output, '~ .'))

  fit = glm(formula = formula, data = train, family = 'binomial')

  #predicted probabilities are for the second factor level
  pred = predict.glm(fit, newdata = test, type = 'response')
  actual = test[nf]

  result = cbind(pred, actual)
  return(result)
}

##############################################################################
#Get predictions using SVM classifier (implementation: e1071:svm)
##############################################################################

get.pred.svm = function(train,test,params) {

  nf = dim(train)[2]
  output_level = levels(train[,nf])[2]
  output = colnames(train)[nf]
  formula = as.formula(paste(output, '~ .'))

  fit = e1071::svm(formula = formula, data = train, prob = T)

  #predicted values are given for both levels, extract probabilities for second level
  pred = predict(fit, newdata = test, prob = T)
  pred = as.data.frame(attr(pred, 'probabilities')[,output_level])
  actual = test[nf]

  result = cbind(pred, actual)
  return(result)
}

##############################################################################
#Get predictions using Naive Bayes classifier (implementation e1071:naiveBayes)
##############################################################################

get.pred.nb = function (train,test,params) {

  nf = dim(train)[2]
  output_level = levels(train[,nf])[2]
  output = colnames(train)[nf]
  formula = as.formula(paste(output, '~ .'))

  fit = e1071::naiveBayes(formula = formula, data = train, prob = T)

  #predicted values are given for both levels, extract probabilities for first level
  pred = predict(fit, newdata = test, type = 'raw')
  pred = as.data.frame(pred[,output_level])
  actual = test[nf]

  result = cbind(pred, actual)
  return(result)
}

##############################################################################
#Get predictions using k Nearest Neighbors classifier (implementation e1071:knn)
##############################################################################

get.pred.knn = function (train,test,kmax=3) {

  best.acc = 0
  best.k = 1
  best.result = c()

  nf = dim(train)[2]
  ks = seq(from=3, to=kmax, by=2)
  for(k in ks) {

    model = class::knn(
      train = train[-nf], test = test[-nf], cl = train[,nf], k = k, l = k / 2, prob = T)

    #predicted probabilities arr given by model for both levels, with prob attribute for each factor
    #change all predictions for just one level
    output_level = levels(train[,nf])[2]
    pred = c()
    for (i in 1:length(model)) {
      #in majority voting it is possible that the model is un-decisive on classification, which will give NA
      if (is.na(model[i])) {
        model[i] = output_level
      }

      if (model[i] == output_level) {
        pred = c(pred, attr(model, 'prob')[i])
      } else {
        pred = c(pred, 1 - attr(model, 'prob')[i])
      }
    }

    actual = test[nf]
    result = cbind(pred, actual)

    metrics = get.metrics(result)
    acc = metrics$acc
    print(paste('Metrics for k=',k,sep=''),q=F)
    print(metrics, q=F)

    if(acc > best.acc) {
      best.acc = acc
      best.k = k
      best.result = result
    }
  }

  print(paste('Best k is',best.k))

  return(best.result)
}

##############################################################################
#Get predictions using default model
#A simplistic default model will just see which class is more prevelant in training data,
# and will assume that all test data will belong to that class
##############################################################################

get.pred.default = function (train,test,params) {

  nf = dim(train)[2]
  nr = dim(train)[1]

  output_level = levels(train[,nf])[2]
  count = sum(train[,nf] == output_level)
  pred = c()
  if (count / nr > 0.5) {
    pred = rep(1, nrow(test))
  } else {
    pred = rep(0, nrow(test))
  }

  actual = test[nf]

  result = cbind(pred, actual)
  return(result)
}


##############################################################################
#k-fold cross validation
#Returns a data frame with predicted and actual values
#k-fold cross validation of the given data
##############################################################################

do.cv.pred.symbolic = function(df, output, k, model_name, params, subset=5, select.features=F) {
  select.features = F
  kn = 1
  model = c()
  if (model_name == 'Default') {
    model = get.pred.default
  } else if (model_name == 'LogReg') {
    model = get.pred.logreg
  } else if (model_name == 'SVM') {
    model = get.pred.svm
  } else if (model_name == 'NaiveBayes') {
    model = get.pred.nb
  } else if (model_name == 'RandomForest') {
    model = get.pred.rf
  } else if (grepl('NN$', model_name)) {
    model = get.pred.knn
    kn = as.numeric(substr(model_name, 1, nchar(model_name) - 2))
  } else {
    stop(
      paste('Invalid model name ', model_name, ' Accepted values are Default, LogReg, SVM, Naive Bayes and kNN, where k can be an integer',sep=''))
  }

  nr = nrow(df)
  nf = ncol(df)
  out_idx = which(colnames(df) == output)
  if (out_idx != nf) {
    #make last column as the output dim
    df = df[,c(1:nf, out_idx)][-out_idx]
    colnames(df)[nf] = output
  }

  #randomize data
  df = df[sample(nr),]
  # get k folds, case handled for the extra ones also
  folds = get.folds(nr, k)
  #perform k iterations for cross-validations
  result = c()
  for (i in 1:k) {

    #Use ith fold as test data, rest as training data
    print(paste('Fold',i))
    startfold = Sys.time()
    test = df[folds[[i]],]
    train = df[-folds[[i]],]

    #feature selection
    if (select.features & model_name != 'Default' & !grepl('NN$', model_name)) {
      print('Entering feature selection with LDA')
    	start=Sys.time()
    	print('Running feature selection')
    	rfFuncs$summary = twoClassSummary
    
    		rfe.ctrl = rfeControl(functions = rfFuncs,
    						   method = 'boot',
    						   #number = 5,
    						   verbose = FALSE,
    						   returnResamp = 'final')

    	train.ctrl = trainControl(classProbs= TRUE, summaryFunction = twoClassSummary)
    	subsets = c(1:subset)

    	#using ROC as the metric for feature selection
    	profile = rfe(train[,-nf], train[,nf],
    					sizes = subsets,
    					method= 'svmRadial',
    					metric = 'ROC',
    					trControl  = train.ctrl,
    					rfeControl = rfe.ctrl)
    	

    	
    	
    	print('--------------------------------------------')
    	print(model_name)
    	print(paste('Fold', i, sep = ""))
    	print(profile)
    	
    	png(filename=model_name)
    	plot(ldaProfile, type = c("o", "g"))
    	dev.off()
    	

    	trellis.par.set(caretTheme())
    	print('--------------------------------------------')
    	print(model_name)
    	print(paste('Fold', i, sep = ""))
    	print(profile)
    	
    	png(filename=model_name)
      plot(profile, type = c("g", "o"), main = model_name)
      dev.off()
      
    	formula = paste(output, '~', paste(predictors(profile), collapse = '+'))
    	print(paste('Using formula',formula))
    	print(paste('Selection'))
    	# print(predictors(profile))
    	print(Sys.time()-start)

    	# subset train and test as per feature selection
    	train = train[,c(predictors(profile), output)]
    	test = test[,c(predictors(profile), output)]
    
    }

    #get predictions from the model
    if (grepl('NN$', model_name)) {
      pred = model(train, test, kn)
    } else {
	    pred = model(train, test, params)
	  }

    result = rbind(result, pred)
    print(paste('Completed Outer Fold',i))
    print(Sys.time()-startfold)
  }
  # return("dumb")
  return(result)
}

##############################################################################
# Get k folds for given number of rows, this is so that we can tune, which k gives us the best results
##############################################################################
get.folds = function(n, k) {

  remainder = n %% k
  folds = split(1:(n - remainder), 1:k)
  if (remainder > 0) {
    for (i in 1:remainder) {
      folds[[i]] = c(folds[[i]], n - i + 1)
      }
  }
  return(folds)
}



##############################################################################
# Feature selection
##############################################################################

filter.features.by.cor = function(df, fit) {

  nf = ncol(df)
  output.name = colnames(df)[nf]
  corr.metrics = data.frame()

  for (i in 1:(nf - 1)) {
    input.name = colnames(df)[i]
    r.squared = summary.lm(fit)$r.squared

    corr.metrics[i,1] = input.name
    corr.metrics[i,2] = sqrt(r.squared)
  }
  colnames(corr.metrics) = c('feature', 'correlation')
  corr.metrics = corr.metrics[order(corr.metrics$correlation, decreasing = T),]
  return(corr.metrics)
}

##############################################################################
# Get CV results for k fold CV
##############################################################################
get.cv.results = function(data, models, k, params, select.features=F) {
  
  print(paste('Starting',k,'Fold CV for',nrow(data),'rows'))
  
  results = list()
 
  for(i in 1:length(models)) {
    print(paste('Running for', models[i], 'model'))
    start = Sys.time()
    result = do.cv.pred.symbolic(data, 'DELAYED', k, models[i], params, select.features)
    print("0000000000000000000000000000000000000000000000000000000000000000")
    print('time taken for the do.cv.pred.symbolic')
    print(Sys.time()-start)
    save(result, file=paste('cv.data.',models[i],sep=''))
    results[[models[i]]] = result
  }

  return(results)
}

##############################################################################
# Get Mean MSE
##############################################################################
get.mean.mse.for.cv.data = function(model.results) {

  mean.mse.results = c()
  for (result in model.results) {
    result[,2] = round(as.numeric(as.character(result[,2])), 4)
    mean.mse = mean((result[,1] - result[,2])^2)
    mean.mse.results = rbind(mean.mse.results, c(names(result), mean.mse))
  }
  mean.mse.results[,1] = as.character(mean.mse.results[,1])
  mean.mse.results[,2] = round(as.numeric(as.character(mean.mse.results[,2])),3)
  colnames(mean.mse.results) = c('Model', 'Mean MSE')
  return(mean.mse.results)
}

##############################################################################
# Get Confusion Matrix metrics
##############################################################################
get.confusion.metrics.for.cv.data = function(model.results, cut.off) {

  metrics.results = list()
  i=0
  for (result in model.results) {
    i=i+1
    metrics = get.metrics(result, cut.off)
    print(paste('Metrics for',names(model.results)[i]),q=F)
    print(metrics)
    metrics.results[i] = metrics
  }

  return(metrics.results)
}

##############################################################################
# Get confusion matrix metrics
##############################################################################
get.metrics = function(model.result, cut.off = 0.5) {

  output_level = levels(model.result[,2])[2]

  total = dim(model.result)[1]
  pos = sum(model.result[,2] == output_level)
  neg = total - pos
  tp = 0; fp = 0; tn = 0; fn = 0

  for (i in 1:total) {
    #the probabilities correspond to the output level label
    prob = model.result[i,1]
    label = model.result[i,2]

    if(is.na(label)) {
      print('NA data')
      print(model.result)
    }

    if (prob > cut.off && label == output_level) {
      tp = tp + 1
    } else if (prob > cut.off && label != output_level) {
      fp = fp + 1
    } else if (prob < cut.off && label != output_level) {
      tn = tn + 1
    } else if (prob < cut.off && label == output_level) {
      fn = fn + 1
    }
  }

  tpr = tp / pos
  fpr = fp / neg
  accuracy = (tp + tn) / total
  precision = tp / (tp + fp)
  recall = tpr

  result = data.frame(
    tpr = tpr, fpr = fpr, acc = accuracy, precision = precision, recall = recall)

  return(result)
}


##############################################################################
#Get month as string from number
##############################################################################

getMonthAsStringFromNumber = function(list) {
  months = c()
  for(month in list) months = c(months, switch(as.integer(month), 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
  return(months)
}

##############################################################################
#Get day of week as string from number
##############################################################################

getDayOfWeekAsStringFromNumber = function(list) {
  days = c()
  for(day in list) days = c(days, switch(as.integer(day),'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'))
  return(days)
}

##############################################################################
#Create plot
##############################################################################

createPlot = function(obj, type='png', plot.num, width=960, height=540) {
  tryCatch(
    if (type == 'png') {
      png(
        filename = paste('project_plot_',plot.num,'.png',sep = ''), width = width, height = height
      )
      print(obj)
      dev.off()
    } else {
      print(obj)
      dev.off()
    }, error = function(e) {
      for (d in dev.list())
        dev.off(which = d)
    }
  )
}

##############################################################################
# Data Loading
##############################################################################

load.data = function() {

  print('Loading data from file')
  if(file.exists('ALL_PIT_DEC_2014_NOV_2015_raw.csv')) {
    print('Loading file ALL_PIT_DEC_2014_NOV_2015.csv')
    raw_df = import.csv('ALL_PIT_DEC_2014_NOV_2015_raw.csv')
  } else {
    raw_df = mergeFiles()
    write.csv(raw_df, 'ALL_PIT_DEC_2014_NOV_2015.csv')
  }
  return(raw_df)
}

##############################################################################
# Data Cleaning
##############################################################################

fix.data = function(raw_df) {

  #Convert to factors for symbolic features
  print('Converting symbolic features to factors')
  raw_df$DAY_OF_WEEK = as.factor(raw_df$DAY_OF_WEEK)
  raw_df$DAY_OF_MONTH = as.factor(raw_df$DAY_OF_MONTH)
  raw_df$MONTH = as.factor(raw_df$MONTH)
  raw_df$DEP_DEL15 = as.factor(raw_df$DEP_DEL15)
  raw_df$ARR_DEL15 = as.factor(raw_df$ARR_DEL15)
  raw_df$CANCELLED = as.factor(raw_df$CANCELLED)
  raw_df$DIVERTED = as.factor(raw_df$DIVERTED)
  raw_df$DISTANCE_GROUP = as.factor(raw_df$DISTANCE_GROUP)
  raw_df$DEP_DELAY_GROUP = as.factor(raw_df$DEP_DELAY_GROUP)
  raw_df$ARR_DELAY_GROUP = as.factor(raw_df$ARR_DELAY_GROUP)
  raw_df$FL_NUM = as.factor(raw_df$FL_NUM)

  #converting to date objects for flight dates
  print('Converting FL_DATE to Date')
  raw_df$FL_DATE = as.Date(as.character(raw_df$FL_DATE), format='%Y-%m-%d')

  #non cancelled/diverted entries, remove missing data features
  print('Pruning cancelled/diverted flights data')
  df = subset(raw_df, CANCELLED == 0 & DIVERTED == 0)
  df = df[,c(1:42)]

  #Rename prediction output column
  df$DELAYED = df$ARR_DEL15
  df$ARR_DEL15 = NULL

  return(df)
}

##############################################################################
# Remove data that is not relevant for prediction
# remove features which won't be available as input to prediction model
###############################################################################

clean.for.delay.prediction = function(df) {

  print('Removing redundant features for prediction')
  #remove features which won't be available
  #at time of prediction
  df$CANCELLATION_CODE = NULL
  df$DIVERTED = NULL
  df$CANCELLED = NULL

  #Features after take-off
  df$DEP_TIME = NULL
  df$ARR_TIME = NULL
  df$ARR_DELAY = NULL
  df$ARR_DELAY_GROUP = NULL
  df$DEP_DELAY = NULL
  df$DEP_DELAY_NEW = NULL
  df$DEP_DELAY_GROUP = NULL
  df$DEP_DEL15 = NULL
  df$ACTUAL_ELAPSED_TIME = NULL
  df$CRS_ELAPSED_TIME = NULL
  df$TAXI_OUT = NULL
  df$TAXI_IN = NULL
  df$WHEELS_ON = NULL
  # df$WHEELS_OFF = NULL

  #Redundant info
  # df$DISTANCE = NULL
  df$FL_DATE = NULL
  df$ORIGIN_AIRPORT_ID = NULL
  df$ORIGIN_STATE_NM = NULL
  df$ORIGIN_CITY_NAME = NULL
  df$DEST_AIRPORT_ID = NULL
  df$DEST_STATE_NM = NULL
  df$DEST_CITY_NAME = NULL
  df$ARR_DELAY_NEW = NULL
  df$AIR_TIME = NULL
  df$FL_NUM = NULL
  df$TAIL_NUM = NULL

  print('')
  print('Columns retained after cleaning data for prediction')
  print(colnames(df))

  return(df)
}


# c("YEAR", "MONTH" ,  "DAY_OF_MONTH" , "DAY_OF_WEEK", "UNIQUE_CARRIER" , "TAIL_NUM","FL_NUM"   ,"ORIGIN", 
#   "DEST","CRS_DEP_TIME","DEP_TIME_BLK", "WHEELS_OFF","CRS_ARR_TIME",
#   "ARR_TIME_BLK", "DISTANCE","DISTANCE_GROUP", "DELAYED") 


#
##
###
##############################################################################
# Data Loading and Cleaning
##############################################################################

#Load data from disk, if available
raw_df = load.data()

#Fix data
df = fix.data(raw_df)

##############################################################################
#Data exploration
##############################################################################

#Clear and remove all graphic devices
for(d in dev.list())
  dev.off(which=d)

#initiate plot count
plot.num = 0
# Turn on/iff exploratory graph plotting
plot.exploratory = F


print('Creating exploratory graphs')
# ##############################################################################

if(plot.exploratory) {
  print('Plotting Overall delay')

  data = c()
  data = rbind(data, c('On Time', round(100*sum(df$DELAYED==0)/nrow(df),2)))
  data = rbind(data, c('Delayed', round(100*sum(df$DELAYED==1)/nrow(df),2)))
  data = as.data.frame(data)
  colnames(data) = c('status', 'percent')
  data$percent = as.numeric(as.character(data$percent))

  plot.num = plot.num + 1
  createPlot(qplot(status, weight=percent, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Overall Delays', x='Flight Status', y='Percentage %')
             + theme_gdocs()
             ,'png',plot.num, 500, 500)

  # ##############################################################################
  print('Plotting Timeseries for entire year for delay minutes')

  data = df[which(df$ARR_DELAY_NEW > 0),]
  data = aggregate(ARR_DELAY_NEW ~ FL_DATE, data=data, FUN=mean)
  data$MONTH = months(data$FL_DATE)

  plot.num = plot.num + 1
  createPlot(qplot(FL_DATE, ARR_DELAY_NEW, data = data, color=MONTH, geom='line')
             + labs(title='Time-series of average daily Delay for an year', x='Flight Date', y='Arrival Delay (minutes)')
             + theme(axis.text.x=element_text(size=12), axis.text.y=element_text(size=12), axis.title.x=element_text(size=14), axis.title.y=element_text(size=14))
             + theme_gdocs()
             ,'png',plot.num)

  ##############################################################################
  print('Plotting Timeseries for entire year for delay percentage')

  data=df
  data$DELAYED = as.numeric(as.character(data$DELAYED))
  data = aggregate(DELAYED ~ FL_DATE, data=data, FUN=function(x) 100*mean(x))
  data$MONTH = months(data$FL_DATE)

  plot.num = plot.num + 1
  createPlot(qplot(FL_DATE, DELAYED, data = data, color=MONTH, geom='line')
             + labs(title='Time-series of daily Delay Percentage for an year', x='Flight Date', y='% of Filghts Delayed')
             + theme(axis.text.x=element_text(size=12), axis.text.y=element_text(size=12), axis.title.x=element_text(size=14), axis.title.y=element_text(size=14))
             + theme_gdocs()
             ,'png',plot.num)

  ##############################################################################
  print('Plotting Monthly mean of delay minutes')

  data = df[which(df$ARR_DELAY_NEW > 0),]
  data$MONTH = getMonthAsStringFromNumber(data$MONTH)
  data = aggregate(ARR_DELAY_NEW ~ MONTH, data=data, FUN=mean)

  plot.num = plot.num + 1
  createPlot(qplot(reorder(MONTH, ARR_DELAY_NEW), weight=ARR_DELAY_NEW, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Delay grouped by Month', x='Month', y='Average Arrival Delay (minutes)')
             + theme_gdocs()
             ,'png',plot.num)

  ##############################################################################
  print('Plotting Monthly delay percentage')

  data=df
  data$DELAYED = as.numeric(as.character(data$DELAYED))
  data$MONTH = getMonthAsStringFromNumber(data$MONTH)
  data = aggregate(DELAYED ~ MONTH, data=data, FUN=function(x) 100*mean(x))

  plot.num = plot.num + 1
  createPlot(qplot(reorder(MONTH,DELAYED), weight=DELAYED, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Delay Percentage grouped by Month', x='Month', y='% of Flights Delayed')
             + theme_gdocs()
             ,'png',plot.num)

  ##############################################################################
  print('Plotting Weekly mean of delay minutes')

  data = df[which(df$ARR_DELAY_NEW > 0),]
  data$DAY_OF_WEEK = getDayOfWeekAsStringFromNumber(data$DAY_OF_WEEK)
  data = aggregate(ARR_DELAY_NEW ~ DAY_OF_WEEK, data=data, FUN=mean)

  plot.num = plot.num + 1
  createPlot(qplot(reorder(DAY_OF_WEEK, ARR_DELAY_NEW), weight=ARR_DELAY_NEW, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Delay grouped by day of week', x='Day of Week', y='Average Arrival Delay (minutes)')
             + theme_gdocs()
             ,'png',plot.num)

  ##############################################################################
  print('Plotting Weekly delay percentage')

  data=df
  data$DELAYED = as.numeric(as.character(data$DELAYED))
  data$DAY_OF_WEEK = getDayOfWeekAsStringFromNumber(data$DAY_OF_WEEK)
  data = aggregate(DELAYED ~ DAY_OF_WEEK, data=data, FUN=function(x) 100*mean(x))

  plot.num = plot.num + 1
  createPlot(qplot(reorder(DAY_OF_WEEK, DELAYED), weight=DELAYED, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Delay Percentage grouped by day of week', x='Day of Week', y='% of Flights Delayed')
             + theme_gdocs()
             ,'png',plot.num)

  ##############################################################################
  print('Plotting Hourly arrival time blocks mean of delay minutes')

  data = df[which(df$ARR_DELAY_NEW > 0),]
  data = aggregate(ARR_DELAY_NEW ~ ARR_TIME_BLK, data=data, FUN=mean)

  plot.num = plot.num + 1
  createPlot(qplot(reorder(ARR_TIME_BLK, ARR_DELAY_NEW), weight=ARR_DELAY_NEW, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Delay grouped by Arrival time blocks', x='Arrival Time block', y='Average Arrival Delay (minutes)')
             + theme_gdocs() + theme(axis.text.x=element_text(angle=45, hjust=1))
             ,'png',plot.num)

  ##############################################################################
  print('Plotting Hourly arrival time blocks delay percentage')

  data=df
  data$DELAYED = as.numeric(as.character(data$DELAYED))
  data = aggregate(DELAYED ~ ARR_TIME_BLK, data=data, FUN=function(x) 100*mean(x))

  plot.num = plot.num + 1
  createPlot(qplot(reorder(ARR_TIME_BLK,DELAYED), weight=DELAYED, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Delay Percentage grouped by Arrival time blocks', x='Arrival Time block', y='% of Flights Delayed')
             + theme_gdocs() + theme(axis.text.x=element_text(angle=45, hjust=1))
             ,'png',plot.num)

  ##############################################################################
  print('Plotting Hourly departure time blocks mean of delay minutes')

  data = df[which(df$ARR_DELAY_NEW > 0),]
  data = aggregate(ARR_DELAY_NEW ~ DEP_TIME_BLK, data=data, FUN=mean)

  plot.num = plot.num + 1
  createPlot(qplot(reorder(DEP_TIME_BLK, ARR_DELAY_NEW), weight=ARR_DELAY_NEW, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Delay grouped by Departure time blocks', x='Departure Time block', y='Average Arrival Delay (minutes)')
             + theme_gdocs() + theme(axis.text.x=element_text(angle=45, hjust=1))
             ,'png',plot.num)

  #############################################################################
  print('Plotting Hourly departure time blocks delay percentage')

  data=df
  data$DELAYED = as.numeric(as.character(data$DELAYED))
  data = aggregate(DELAYED ~ DEP_TIME_BLK, data=data, FUN=function(x) 100*mean(x))

  plot.num = plot.num + 1
  createPlot(qplot(reorder(DEP_TIME_BLK,DELAYED), weight=DELAYED, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Delay Percentage grouped by Departure time blocks', x='Departure Time block', y='% of Filghts Delayed')
             + theme_gdocs() + theme(axis.text.x=element_text(angle=45, hjust=1))
             ,'png',plot.num)

  #############################################################################

  data = data.frame('Actual'=strptime(as.character(df$ARR_TIME), '%H%M'), 'CRS'=strptime(as.character(df$CRS_ARR_TIME), '%H%M'))
  plot.num = plot.num + 1
  createPlot(ggplot(melt(data), aes(value, color = variable)) + geom_density()
             + labs(title='Density distribution of Actual and CRS Arrival Times', x='Time (HH:MM)', y='Density')
             + scale_color_hue(name='Feature')
             + theme_gdocs()
             ,'png',plot.num)

  ##############################################################################

  data = df
  plot.num = plot.num + 1
  createPlot(qplot(y=ARR_TIME-CRS_ARR_TIME, x=1, data=data, geom='boxplot')
             + coord_flip()
             + labs(title='Difference between Actual and CRS Arrival times', y='Time difference (minutes)', x='')
             + theme_gdocs() + theme(axis.text.y = element_blank())
             ,'png',plot.num)

  ##############################################################################

  data = df
  print(summary(data$ARR_TIME-data$CRS_ARR_TIME))
  diff.mean = mean(data$ARR_TIME-data$CRS_ARR_TIME)
  diff.sd = sd(data$ARR_TIME-data$CRS_ARR_TIME)

  plot.num = plot.num + 1
  createPlot(qplot(ARR_TIME-CRS_ARR_TIME, data=data, geom='density')
             + geom_vline(color='blue', xintercept = diff.mean)
             + geom_vline(color='red', xintercept = diff.mean + 2.58*diff.sd/sqrt(nrow(data)))
             + geom_vline(color='red', xintercept = diff.mean - 2.58*diff.sd/sqrt(nrow(data)))
             + labs(title='Difference between Actual and CRS Arrival times', y='Density', x='Time difference (minutes)')
             + theme_gdocs()
             ,'png',plot.num)

  ##############################################################################

  plot.num = plot.num + 1
  createPlot(qplot(ARR_TIME-CRS_ARR_TIME, data=data, geom='density')
             + geom_vline(color='blue', xintercept = diff.mean)
             + geom_vline(color='red', xintercept = diff.mean + 2.58*diff.sd/sqrt(nrow(data)))
             + geom_vline(color='red', xintercept = diff.mean - 2.58*diff.sd/sqrt(nrow(data)))
             + labs(title='Difference between Actual and CRS Arrival times', y='Density', x='Time difference (minutes)')
             + xlim(-60,60)
             + theme_gdocs()
             ,'png',plot.num)


  # ##############################################################################
  print('Plotting destination wise plot')

  df1 <- df
  delayed_flights <- df1[which(df1$DELAYED == 1), ]

  delayed_flights_from_pitts <- delayed_flights[which(delayed_flights$ORIGIN == 'PIT'), ]
  delayed_flights_to_pitts <- delayed_flights[which(delayed_flights$DEST == 'PIT'), ]


  data <- delayed_flights_from_pitts[c('DEST', 'DELAYED')]

  count_per_dest <- with(data, table(DEST))
  a<- attr(count_per_dest, 'dimnames')
  plot(count_per_dest, main = 'No. of Flights delays distributed by Destination Airports',
       xlab='Destination', ylab = 'Number of flights delayed')


  #To pitts
  #  # ##############################################################################
  delayed_flights_to_pitts <- delayed_flights[which(delayed_flights$DEST == 'PIT'), ]

  data <- delayed_flights_to_pitts[c('ORIGIN', 'DELAYED')]
  count_per_dest <- with(data, table(ORIGIN))
  a<- attr(count_per_dest, 'dimnames')
  a <- unlist(a, recursive = TRUE, use.names = FALSE)
  plot(count_per_dest, main = 'No. of Flights delays distributed by Destination Airports', xlab='Destination', ylab = 'Number of flights delayed')


  print('Plotting destination wise plot')

  data = rbind(data, c('On Time', round(100*sum(df$DELAYED==0)/nrow(df),2)))
  data = rbind(data, c('Delayed', round(100*sum(df$DELAYED==1)/nrow(df),2)))
  data = as.data.frame(data)
  colnames(data) = c('status', 'percent')
  data$percent = as.numeric(as.character(data$percent))

  plot.num = plot.num + 1
  createPlot(qplot(status, weight=percent, data=data, geom='bar', stat='identity', fill=I('deepskyblue4'))
             + labs(title='Overall Delays', x='Flight Status', y='Percentage %')
             + theme_gdocs()
             ,'png',plot.num, 500, 500)

}

##############################################################################
# ROC
##############################################################################

plot.roc = function(results, plot.num) {

  png(filename = paste('project_plot_',plot.num,'.png',sep = ''), width = 800, height = 800)

  cols = c('black', 'darkgreen', 'red', 'blue', 'orange', 'violet', 'magenta')
  cols = cols[1:length(results)]

  # plotting the ROC curve
  par(mar=c(5,5,2,2),xaxs = 'i',yaxs = 'i',cex.axis=1.3,cex.lab=1.4)

  auc.data = c()
  obj = c()
  i=0
  for(result in results) {
    i = i + 1

    pred = prediction(as.matrix(result[,1]), as.matrix(result[,2]))
    perf = performance(pred,'tpr','fpr')
    # plotting the ROC curve
    if(i==1)
      plot(perf, col=cols[i], lwd=2)
    else
      plot(perf, add=T, col=cols[i], lwd=2)

    # Calculate AUC
    auc = performance(pred,'auc')
    # now converting S4 class to vector
    auc.val = round(unlist(slot(auc, 'y.values')),2)
    auc.data = c(auc.data, paste(names(results)[i],'=',auc.val))
  }

  grid()
  legend('bottomright', title = 'Model and AUC', legend = auc.data, col = cols, lty=1)
  dev.off()
}


##############################################################################
#Feature engineering
##############################################################################

data = clean.for.delay.prediction(df)
# cols = c('YEAR','DAY_OF_WEEK', 'DAY_OF_MONTH', 'MONTH',
#          'DEP_TIME_BLK', 'ARR_TIME_BLK',
#          'DISTANCE_GROUP',
#          'DEST', 'ORIGIN',
#          'UNIQUE_CARRIER',
#          'DELAYED', 'TAIL_NUM', '')

cols <- c("MONTH",  "DAY_OF_MONTH" , "DAY_OF_WEEK", "UNIQUE_CARRIER","ORIGIN", 
          "DEST","CRS_DEP_TIME","DEP_TIME_BLK","CRS_ARR_TIME"   ,
           "ARR_TIME_BLK", "DISTANCE","DISTANCE_GROUP", "DELAYED") 

data = data[, cols]
data = model.matrix(~+ MONTH + DAY_OF_MONTH + DAY_OF_WEEK + UNIQUE_CARRIER  +ORIGIN + 
                    DEST  + CRS_DEP_TIME + DEP_TIME_BLK  +  CRS_ARR_TIME+
                    ARR_TIME_BLK + DISTANCE_GROUP + DELAYED,
                    data=data )


#add numeric data column
num.cols = c('AIR_TIME', "DISTANCE")
data = cbind(data, df[,num.cols])

#skip first intercept column
data = as.data.frame(data[,-1])

#factorize delayed into character factor, not 0,1
data$DELAYED = as.factor(data$DELAYED1)
data$DELAYED1 = NULL
levels(data$DELAYED) = c('N', 'Y')

# fix column names for fitting models
colnames(data) = make.names(colnames(data))

print('')
print('Columns used for prediction algos')
print(colnames(data))

##############################################################################
##############################################################################

data = data[sample(10000), ]

# models = c('Default', 'LogReg', '7NN', 'NaiveBayes', 'SVM', 'RandomForest')
models = c('LogReg', 'RandomForest')

plot.num = 0
plot.num = plot.num + 1

# start = Sys.time()
cv.data = get.cv.results(data, models, 10, c(400))



print(paste('get.cv.results'))
print(Sys.time()-start)
save(cv.data, file='cv.data')

print("***********************************************")
print(cv.data)


cv.data = list()

for(model in models) {
  cv.data[[model]] = get(load(file=paste('cv.data.',model,sep='')))
}

plot.num=plot.num+1
plot.roc(cv.data, plot.num)

cut.offs = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
metrics.list = list()
for(cut.off in cut.offs) {
  metrics.list[[as.character(cut.off)]] = get.confusion.metrics.for.cv.data(cv.data, cut.off)
}
save(metrics.list, file='metrics')


random forest param selection
start = Sys.time()
res = get.pred.rf(data[1:900,], data[901:1000,], c(300,400,500,600,700,800))
print(Sys.time()-start)

fit random forest on entire data
train.ctrl = trainControl(method="cv", number = 5, summaryFunction=twoClassSummary, classProb=T)
fit = train(as.formula('DELAYED ~ .'), data=data, method="rf", trControl=train.ctrl, metric="ROC", ntree=400)
save(fit, file='rf.400.fit')

fit = get(load('rf.400.fit'))

print('Most important features sorted in decreasing order of MeanDecreaseGini')
print(' A low Gini (i.e. higher descrease in Gini) means that a particular predictor variable plays a greater role in partitioning the data into the defined classes.')
imp.predictors = names(fit$finalModel$importance[order(-fit$finalModel$importance),])
print(head(imp.predictors, 10))

most.imp = c()

for(col in cols) {
  found.first = F
  for(name in imp.predictors) {
    if(!found.first & grepl(paste(col,'*',sep=''), name)) {
      if(name != 'DESTPIT' & name != 'ORIGINPIT') {
        most.imp = c(most.imp, name)
        found.first = T
      }
    }
  }
}

print('Most important features per category in')
print(most.imp)

##############################################################
##############################################################
get.pred.delayed = function(test, cols, vals) {

  test$AIR_TIME = as.numeric(vals[length(vals)])
  test$DELAYED = 'N'
  for(i in 1:(length(cols)-1)) {
    new.col = paste(cols[i],vals[i],'$',sep='')
    for(name in colnames(test)) {
      if(grepl(new.col, name)) {
        test[1, name] = 1
      }
    }
  }

  pred = as.data.frame(as.numeric(predict(fit, test, type = 'prob')[,2]))
  return(pred[1,1])
}




delays.by.flight = aggregate(DELAYED ~ FL_NUM, data=df, FUN=function(x) 100*mean(as.numeric(as.character(x))))

head(subset(delays.by.flight, DELAYED>50 & DELAYED < 100), 10)

nrow(subset(df, FL_NUM==1081))
nrow(subset(df, FL_NUM==322))

head(df[,c('FL_DATE', 'DAY_OF_WEEK', 'ORIGIN', 'DEST', 'UNIQUE_CARRIER', 'DEP_TIME_BLK', 'ARR_TIME_BLK', 'DISTANCE_GROUP', 'AIR_TIME')], 1)

day.of.week = '1'
day.of.month = '1'
month = '12'
dep.time.blk = '0900.0959'
arr.time.blk = '1100.1159'
dest = 'DFW'
origin = 'PIT'
carrier = 'AA'
dist.grp = '5'
air.time = 174

#prepare query
vals = c(day.of.week, day.of.month, month, dep.time.blk, arr.time.blk, dist.grp, dest, origin, carrier, air.time)
test = data[1, 1:(ncol(data)-1)]
test[1,] = 0

#predict delay
get.pred.delayed(test, cols, vals)




#Schlumberger Project New
#===========================================================================
#Get only rows that have to or from pittsburgh

raw_df = import.csv('output.csv')

get the Na quantiaty
See if there is a diff between crs dep time and dep time
How many times was the dep actuallu delayed, in before time?
everything is in minutes



perfs <-data.frame(Model_Names = c('LogReg', 'Knn', 'Random Forest', 'Naïve Bayes', 'SVM'), TPR = c(0.01829587 , 0.2843701, 0.7334, 0.5405123, 0.004704652), 
                   FPR = c(0.006924694, 0.1491282, 0.02745147, 0.3550142, 0.002102139),
                   Accoracy = c(0.8066, 0.7334, 0.85, 0.625, 0.8079),
                   Precision = c(0.3846154 , 0.3108571, 0.7418605, 0.2647887, 0.3461538),
                   Recall = c(0.01829587 , 0.2843701, 0.3335076, 0.5405123, 0.004704652))



library(plotly)


p <- plot_ly(
  x = as.character(perfs[["Model_Names"]]),
  y = perfs$TPR,
  name = "SF Zoo",
  type = "bar"
)
p

p <- plot_ly(
  x = as.character(perfs[["Model_Names"]]),
  y = perfs$FPR,
  name = "SF Zoo",
  type = "bar")
p





p <- plot_ly(
  x = as.character(perfs[["Model_Names"]]),
  y = perfs$Accoracy,
  name = "SF Zoo",
  type = "bar"
)
p

p <- plot_ly(
  x = as.character(perfs[["Model_Names"]]),
  y = perfs$Precision,
  name = "SF Zoo",
  type = "bar"
)
p

p <- plot_ly(
  x = as.character(perfs[["Model_Names"]]),
  y = perfs$Recall,
  name = "SF Zoo",
  type = "bar"
)


















