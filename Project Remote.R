#################################
# clean the training data

#read training data and replace "#DIV/0!" with "NA"
traingdata <- read.csv("pml-training.csv", na.strings=c("#DIV/0!","NA"))

#remove columns with no data (all NA)
trainnacols <- traingdata[,colSums(is.na(traingdata))>nrow(traingdata)/10]

#find index of columns with high counts of NA's
naIndexs <- which(names(traingdata) %in% names(trainnacols))

#I'm also going to remove unused columns (X, user_name, time stamps and window: column indexs: 1:6)
naIndexs <- append(1:6, naIndexs)

#now remove these columns from the dataset
cleandata <- traingdata[,-naIndexs]

#also try to reduce to minimum features with max variablity (plus num_window)
savenames <- c("num_window","roll_belt","pitch_belt","yaw_belt","roll_arm","pitch_arm","yaw_arm","roll_forearm","pitch_forearm","yaw_forearm","classe")
useIndexs <- which(names(cleandata) %in% savenames)
cleandata2 <- cleandata[,useIndexs]

if (require("caret") == FALSE)
{
  install.packages("caret", dependencies = c("Depends", "Suggests"))
  library(caret)
}

set.seed(12345)
inTrain <- createDataPartition(y=cleandata$classe, p=0.6, list=FALSE)

cleantraining <- cleandata[inTrain,]
cleantesting <- cleandata[-inTrain,]

#also split the reduced data as well for futher testing

library(caret)

set.seed(12345)
inTrain2 <- createDataPartition(y=cleandata2$classe, p=0.6, list=FALSE)

cleantraining2 <- cleandata2[inTrain2,]
cleantesting2 <- cleandata2[-inTrain2,]

#now do model fitting

set.seed(5436)


system.time(
  modelFit <- train(classe ~ .,data=cleantraining,method="rpart")
)
#13 seconds

modelFit

modelFit$finalModel

################################################
#experiment with various rf configuration to view and analyze
if (require("doParallel") == FALSE)
{
  install.packages("doParallel")
  library(doParallel)
}
#cl <- makeCluster(10) #setup run to use 10 cores
registerDoParallel(cores = 16)

#standard default train setup
set.seed(4532)
system.time(
  modFit <- train(classe ~ .,data=cleantraining,method="rf",prox=TRUE)
)
modFit
saveRDS(modFit,"rfFullModel.rds")
rm(modFit)

#standard default with tweak to tuneLength from 3 to 6
set.seed(4532)
system.time(
  modFitb <- train(classe ~ .,data=cleantraining,method="rf",prox=TRUE, tuneLength=6)
)
modFitb
saveRDS(modFitb,"rfFullBModel.rds")
rm(modFitb)


#change to repeated cv with repeats of 5
set.seed(4532)
ctrl <- trainControl(method = "repeatedcv", repeats = 5)


system.time(
  modFit2 <- train(classe ~ ., data = cleantraining, method = "rf",prox=TRUE, trControl = ctrl)
)

modFit2
saveRDS(modFit2,"rfFullModel2.rds")
rm(modFit2)

#change to repeated cv with repeats of 5 and tuneLength=6
set.seed(4532)
ctrl <- trainControl(method = "repeatedcv", repeats = 5)


system.time(
  modFit2b <- train(classe ~ ., data = cleantraining, method = "rf",prox=TRUE, trControl = ctrl, tuneLength=6)
)

modFit2b
saveRDS(modFit2b,"rfFullModel2b.rds")
rm(modFit2b)

####
#change to repeated cv with repeats of 5 and tuneLength=6 and use reduced data
set.seed(4532)
ctrl <- trainControl(method = "repeatedcv", repeats = 5)


system.time(
  modFit2b2 <- train(classe ~ ., data = cleantraining2, method = "rf",prox=TRUE, trControl = ctrl, tuneLength=6) # 5124.727 secs to train to get 99.8% accuracy 
)

modFit2b2
saveRDS(modFit2b2,"rfFullModel2b2.rds")
rm(modFit2b2)

####

#change performance metrics (note, twoclass summary is not appropirate for 5 outputs)
set.seed(4532)
ctrl <- trainControl(method="repeatedcv", repeats=10, classProbs=TRUE)

system.time(
  modFit3 <- train(classe ~ ., data = cleantraining, method = "rf",prox=TRUE, trControl = ctrl)
)

modFit3
saveRDS(modFit3,"rfFullModel3.rds")
rm(modFit3)
