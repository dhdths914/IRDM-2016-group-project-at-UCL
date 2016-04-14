library(data.table)
library(xgboost)
library(lattice)
library(ggplot2)
library(caret)
setwd("/Users/dhdths914/Documents/UCL-CSML/DM/ICA2")
ISE <- fread("/Users/dhdths914/Documents/UCL-CSML/DM/ICA2/data.csv")
data <- fread("/Users/dhdths914/Documents/UCL-CSML/DM/ICA2/train.csv")
ob<-ISE[1:30,]
id<-1:30
ob<-c(ISE,id)
#################################
#variable relationship
#################################
plot(id,ob$ISE, type="b", xlab="time",ylab="stock market return", main="line chart for variables" ) 
lines(id,ob$SP , type="b", lwd=1.5,col=2) 
lines(id,ob$DAX , type="b", lwd=1.5,col=3)
lines(id,ob$FTSE , type="b", lwd=1.5,col=4)
lines(id,ob$NIKKEI , type="b", lwd=1.5,col=5)
lines(id,ob$BOVESPA , type="b", lwd=1.5,col=6)
lines(id,ob$EU , type="b", lwd=1.5,col=7)
lines(id,ob$EM , type="b", lwd=1.5,col=8)
legend("topright",c("ISE","SP","DAX","FTSE","NIKKEI","BOVESPA","EU","EM"),lwd=1.5, col=1:8, title = "stock")

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
param <- list("nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)
cv.nround<-5
cv.nfold<-3
bst.cv = xgb.cv(param=param, data = trainmatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround,missing=NaN)
nround = 100
bst = xgboost(param=param, data = trainmatrix, label = y, nrounds=nround,missing=NaN,verbose = 1)
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





