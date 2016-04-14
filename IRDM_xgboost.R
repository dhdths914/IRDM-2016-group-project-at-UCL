#################################
#package
#################################
library(data.table)
library(xgboost)
library(lattice)
library(ggplot2)
library(caret)
#################################
#xgboost
#################################
dim(data)
train <- data[1:484,]
test <- data[485:536,]
y<-train$ISE
ytest<-test$ISE
test$ISE <- NULL
trainmatrix<-data.matrix(train)
testmatrix<-data.matrix(test)
trainmatrix<-scale(trainmatrix)
testmatrix<-scale(testmatrix)
param <- list("nthread" = 8,   # number of threads 
              "max_depth" = 20,    # maximum depth of tree 
              "eta" = 1,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns 
              "min_child_weight" = 12  # minimum sum of instance weight 
)
cv.nround<-5
cv.nfold<-3
bst.cv = xgb.cv(param=param, data = trainmatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround,missing=NaN)
nround = 100
bst = xgboost(param=param, data = trainmatrix, label = y, nrounds=nround,missing=NaN,verbose = 1)

#predict
ypred<-predict(bst,testmatrix,missing=NaN)
predmatrix<-data.frame(matrix(ypred,byrow=T))
write.csv(predmatrix,'predict.csv',quote=F,row.names = F)

#line chart
time<-1:52
plot(time,ypred, type="b", xlab="time",ylab="stock market return", main="ISE predictive values" ) 
lines(time,ytest , type="b", lwd=1.5,col=2)
legend("topright",c("output","target"),lwd=1.5, col=1:2)


#evaluation
MSE<-1/52*sum((predmatrix$matrix.ypred..byrow...T.-test$ISE.1)^2)
correlation<-cor(ytest,ypred)


#reference:https://www.kaggle.com/yib2irvine/airbnb-recruiting-new-user-bookings/ndcg-xgboost-example/run/123656