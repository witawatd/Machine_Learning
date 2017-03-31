setwd("~/Desktop/CSC529/Project")

require(caret)
require(ggplot2)
require(randomForest)
require(e1071)
require(pROC)
require("foreach")
require("doSNOW")
require(party)

myd = read.csv("LIDC_Data_SEIDEL_01-05-2015.csv",header = T, sep = ',')
mycl = data.frame(matrix(NA, ncol = 1, nrow = nrow(myd)) )
names(mycl)[1]<-paste("class")

for(i in 1:nrow(myd))
{
  row <- myd[i,]
  avg <- (row[5]+row[6]+row[7]+row[8])/4
  mycl[i,1] <-  ifelse(avg >=4 , "Likely", "Inconclusive")
}

qplot(mycl$class, geom="bar",ylab = "count", main = "Class Distribution ", xlab = "Class", fill=as.factor(mycl$class)) 

###############################################################

outcome = as.factor(mycl$class)
levels(outcome) = make.names(levels(outcome))

##################################

set.seed(4567)
#trainIndex <- createDataPartition(outcome , p=0.66, list=FALSE)
#train_X <- myd[,9:72][ trainIndex,]
#train_Y <- as.data.frame(outcome)[trainIndex,]

#test_X <- myd[,9:72][-trainIndex,]
#test_Y <- as.data.frame(outcome)[-trainIndex,]

##################################
train_X = myd[,9:72]
train_Y = outcome


#######################   SMOTE     #######################################
######## Balance the data set, comment this part if you don't want ########
require(DMwR)
sdata = cbind(train_Y,train_X)
names(sdata)[1]<-paste("class")

Bal_data_train <- SMOTE(class ~ ., sdata, perc.over = 500,perc.under=110)
table(Bal_data_train$class)
qplot(Bal_data_train$class, geom="bar",ylab = "count", main = "Class Distribution ", xlab = "Class", fill=Bal_data_train$class ) 

train_X = Bal_data_train[,2:65]
train_Y = as.factor(Bal_data_train[,1])

###################   Feature Extraction     ##############################
#######    Extract new feature, reduce feture dimention, ############
#######     comment this part if you don't want #####################
require(clusterSim)
fdata = cbind(train_Y,train_X)
Norm_data_train = data.Normalization(fdata[2:65],type="n4",normalization="column");

#Apply PCA
data_train.pca <- princomp(Norm_data_train, cor = TRUE, center = TRUE, scale. = TRUE) 
plot(data_train.pca)
summary(data_train.pca)

train_X = data_train.pca$scores[,1:22]
train_Y = as.factor(Bal_data_train[,1])

############################################################
fitControl <- trainControl(method = "repeatedcv",
                           repeats = 10,
                           number = 10,
                           #summaryFunction = multiClassSummary,
                           summaryFunction = twoClassSummary,
                           savePredictions =T,
                           ## Estimate class probabilities
                           classProbs = TRUE)

#######################################################

fit.rf = train(y = train_Y, x = train_X, method= "rf"  , 
                  tuneLength = 5,
                  metric="ROC",
                  trControl = fitControl)
plot(fit.rf)
fit.rf

pred <- predict(fit.rf, train_X)
xtab <- table(pred, train_Y)
confusionMatrix(fit.rf)

#########################################################

fit.ada = train(y = train_Y, x = train_X, method= "ada"  , 
               metric="ROC",
               trControl = fitControl)
plot(fit.ada)
fit.ada

trellis.par.set(caretTheme())
plot(fit.ada, metric = "ROC", plotType = "level",
     scales = list(x = list(rot = 90)))

############################################################

fit.treeBag = train(y = train_Y, x = train_X, method= "treebag", 
                    metric="ROC",
                    trControl = fitControl)
plot(fit.treeBag)
fit.treeBag

###########################################################
fit.LogitBoost = train(y = train_Y, x = train_X, method= "LogitBoost", 
                    metric="ROC",
                    tuneLength = 8,
                    trControl = fitControl)
plot(fit.LogitBoost)
fit.LogitBoost

###########################################################

# fit.AdaM1 = train(y = train_Y, x = train_X, method= "AdaBoost.M1", 
#                        metric="ROC",
#                        trControl = fitControl)
# plot(fit.AdaM1)
# fit.AdaM1

###########################################################

fit.gbm = train(y = train_Y, x = train_X, method= "gbm", 
                  metric="ROC",
                  trControl = fitControl)
plot(fit.gbm)
fit.gbm

##########################################################

fit.bagLDA <- train(train_X, train_Y, 
                 "bag", 
                 B = 10, 
                 bagControl = bagControl(fit = ldaBag$fit,
                                         predict = ldaBag$pred,
                                         aggregate = ldaBag$aggregate),
                 trControl = fitControl,
                 tuneGrid = data.frame(vars = c((1:10)*10 , ncol(train_X))))

plot(fit.bagLDA)
fit.bagLDA
################Perf #######################################

resamps <- resamples(list(RF = fit.rf,
                          BoostedTree = fit.ada,
                          BaggedCart = fit.treeBag,
                          LogitBoost= fit.LogitBoost,
                          BaggedLDA = fit.bagLDA,
                          GBM = fit.gbm
))
resamps
summary(resamps)
bwplot(resamps, layout = c(5, 1))







##################### Testing ########################

pred <- predict(fit.rf, test_X)
xtab <- table(pred, test_Y)
confusionMatrix(xtab)

pred <- predict(fit.ada, test_X)
xtab <- table(pred, test_Y)
confusionMatrix(xtab)

pred <- predict(fit.treeBag, test_X)
xtab <- table(pred, test_Y)
confusionMatrix(xtab)

pred <- predict(fit.LogitBoost, test_X)
xtab <- table(pred, test_Y)
confusionMatrix(xtab)

