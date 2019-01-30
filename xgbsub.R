#xgboost crossvalidation model for kaggle competion:Springleaf
library(xgboost);
library(verification);
  set.seed(666)
  data2 <-as.data.frame(data2)
  datatest <- as.data.frame(datatest)
  k = 4
  
  
  target <- length(names(data2))
  
  #subset <- sample(1:dim(data2)[1],0.3*dim(data2)[1])
  
  #cvtest <- data2[subset,]
  #y<- data2[subset,target]
  
  cvtest <- as.matrix(datatest)
  mode(cvtest) <- "numeric"
  cvtest <- xgb.DMatrix(cvtest)
  

  cvtrain <- as.matrix(data2)
  mode(cvtrain) <- "numeric"
  cvtrain <- xgb.DMatrix(cvtrain[,-target],label=cvtrain[,target])
  
  
    d <- 4
    e <- 0.1
    Nit <- 50
    param2 <- list(objective='binary:logistic',max.depth=d,eta=e,eval_metric="auc")
    timestart <- proc.time()["elapsed"]
    best <- xgb.train(param=param2,data=cvtrain,nrounds=Nit,verbose=0)
    time <- proc.time()["elapsed"]-timestart
    pred <- predict(best,cvtest)
    submission <- data.frame(ID=datatest$ID)
    submission$target <- pred
    write.csv(submission,"xgbsub.csv")
    #result <-paste("model"," auc ",roc.area(y,pred)$A,"dp:",d,"eta:",e,"it:",Nit,"time:",time,sep="")
    #print(result)
    #write.table(result,file="results.txt",col.names = F,row.names = F,append = T)
    xgb.save(best,"subxgb.model")
