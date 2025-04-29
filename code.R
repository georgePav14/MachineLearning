rm(list=ls())

# Euclidean distance
eDis<-function(x,y){
  sqrt(sum((x - y)^2))
}

##### packages installed #####
# we need now to call them
install.packages(c('corrgram','nnet','class','tree','MASS','pgmm','klaR',
                   'dplyr','ggplot2','scoring','e1071','randomForest','readxl','nortest'))

library(corrgram)
library(nnet)
library(class)
library(tree)
library(MASS)
library(pgmm)
library(klaR)
library(dplyr)
library(ggplot2)
library(scoring)
library(e1071)
library(randomForest)
library(readxl)
library(psych)
library(nortest)


## read the data
x<-read_excel(file.choose())
x<-as.data.frame(x)
View(x)
str(x)
colnames(x)
head(x)

# define the categorical variables
for(i in c(2:10,15,21) ){
  x[,i]<-factor(x[,i])
} ; str(x)

# Observations for validation
nValid<-round(0.1*nrow(x),0)
indValid<-sample(1:nrow(x),nValid,replace = FALSE)
valid<-x[indValid,] ; x<-x[-indValid,]

########### LDA ############
# LDA needs continuous variables to work
lapply(x,class) # class of each variable
cont<-x[,c(21,1,11,19)]# The continuous variables & Subscr.
sum(x$pdays==999) # How many havent been previously contacted
cont<-cont[,-4] # remove pdays too little meaningful observations
m1<-lda(SUBSCRIBED~., data=cont) 
m2<-predict(m1)
t<- table(cont$SUBSCRIBED,m2$class) ; t #confusion matrix. The rows are the predictions the column the true

sum(diag(t))/sum(t) # In sample Accuracy

describe(cont)

# for LDA  we assume multivariate normal clusters
# so we have to test which variables we you use so
# the normality can be supported

for (i in names(cont)[-1]){ # I remove some because they are discrete numerical
  print(i)
print(lillie.test(x[,i]))
print(shapiro.test(x[1:5000,i]))
}
cont<-x[,c(21,1,11,19)]
# the test rejects normality for the variables
# but if we try anyway

#### arbitrary train-test split
nTrain<-round(0.7*nrow(cont),0) ; nTrain # Number of observations for training

ind<- sample(1:nrow(x),nTrain,replace=FALSE)
train<- cont[ind,]
test<- cont[-ind,]

m1<-lda(SUBSCRIBED~., data=train)
m2<-predict(m1, newdata=test)
m2$posterior # probabilities
t<- table(test$SUBSCRIBED,m2$class) ; t
t[2,2]/sum(t[2,]) # sensetivity

cbind(m2$posterior,m2$class,test$SUBSCRIBED)[test$SUBSCRIBED=='yes',]

####### ROC curve 

  model<- lda(SUBSCRIBED~., data=cont)
  pred<-predict(model,cont)
  res<-NULL
  variab<-2:4
  
for (threshold in seq(0.01,0.9,by=0.01)){
  bootLDA<-matrix(nrow = 10,ncol = 3)
  for ( j in 1:10) { # bootstrap
    ind<- sample(1:nrow(cont),nrow(cont), replace=TRUE)
    check<- 1:nrow(cont)
    t<- check %in% unique(ind)
    notin<- check[!t]
    
    train <- cont[ind,variab]
    test <-   cont[notin,variab]
    z <-  lda(train, cont[ind,1])
    pr<-  predict(z, test)
    class<- (pr$posterior[,2]>threshold)
    class<-factor(class,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
    t<-table(cont[notin,1],class)
    sensBoot<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
    accBoot<-sum(diag(t))/nrow(test)
    specBoot<- t[1,1]/apply(t,1,sum)[1]
    bootLDA[j,]<- c(sensBoot,accBoot,specBoot)
  }
  
  sens<-mean(bootLDA[,1])
  acc<- mean(bootLDA[,2])
  spec<- mean(bootLDA[,3])
  res<-rbind(res, c(sens,1-spec,acc))
  if( (10*threshold) %% 1 == 0){
    cat(paste0("Progress: ", round(threshold/0.009,2),"%."),"\n")
  }
}
colnames(res)<- c("Sensitivity", "1-Specificity",'Accuracy') ; rownames(res)=as.character(seq(0.01,0.9,by=0.01))
res

# plot the roc curve
plot(res[,2],res[,1], xlab="FN", ylab="TP", type="l",col=3, xlim=c(0,1), ylim=c(0,1),lwd=2,main = 'ROC curve')
abline(0,1)

seq(0.01,0.9,by=0.01)[which.max(res[,1])] #the threshold with the highest sensitivity

res[ which.max(apply(res[,1:2],1,function(x) eDis(x,c(0,1)))) ,] # calculate the euclidian dis from (0,1)
#  I choose 0.06

## repeat 1000 times
B<-100
resultLDA<-matrix(nrow = 100,ncol = 2)
for ( i in 1:B) {
  ind<- sample(1:nrow(x),nTrain,replace=FALSE)
  train<- cont[ind,]
  test<- cont[-ind,]
  
  m1<-lda(SUBSCRIBED~., data=train)
  m2<-predict(m1, newdata=test)
  class<- (m2$posterior[,2]>0.06)
  class<-factor(class,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
  t<- table(test$SUBSCRIBED,class)
  resultLDA[i,]<-c(t[2,2]/sum(t[2,]),sum(diag(t))/nrow(test))
  if(i %% 10 == 0){
    cat(paste0("Progress: ", round(i/1,2),"%."),"\n")
  }
}
colnames(resultLDA)=c('Sensitivity','Accuracy')

hist(resultLDA[,1],main='Histogram of the Test Sensitivity of 1000 runs (LDA)')
hist(resultLDA[,2],main='Histogram of the Test Accuracy of 1000 runs (LDA)')

#### the percentage of each classification in the confusion matrix
round(aggregate(m2$posterior[,] ,list(as.numeric(test$SUBSCRIBED) ), mean),2)
# the TRUE classification 1 = 'no' & 2 = 'yes'


#####  k-fold cross validation 

deiktes<-sample(1:nrow(cont))
variab<-2:4
crossLDA<-matrix(ncol = 2)
folds<-c(2,3,4,5,7,10,12,15,20)

for (omades in folds) {  # omades is the number of folds
  k<-round((nrow(cont)/omades) - 1) # k is the number of observations per fold
  t2=t3<-NULL ; t<-matrix(nrow = 2,ncol = 2)
  
  for (i in 1:omades) {
    te<- deiktes[ ((i-1)*k+1):(i*k-1)]
    train <- cont[-te,variab]
    test <-   cont[te,variab]
    z <-  lda(train, cont[-te,1])
    pr<-  predict(z, test)
    class<- (pr$posterior[,2]>0.06)
    class<-factor(class,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
    t<-table(cont[te,1],class)
    t2<- c(t2,t[2,2]/sum(t[2,])) # sensitivity TP / totalTruePositives
    t3=c(t3,sum(diag(t))/nrow(test))
  }
  crossLDA<-rbind(crossLDA,c(mean(t2),mean(t3)))
}
crossLDA=crossLDA[-1,] 
rownames(crossLDA)<-folds ; colnames(crossLDA)=c('Sensitivity','Accuracy') ; crossLDA

########### bootstrap 
B<-100
variab<-2:4
bootLDA<-matrix(nrow = 100,ncol = 2)

for ( j in 1:B) {
  ind<- sample(1:nrow(cont),nrow(cont), replace=TRUE)
  check<- 1:nrow(cont)
  t<- check %in% unique(ind)
  notin<- check[!t]
  
  train <- cont[ind,variab]
  test <-   cont[notin,variab]
  z <-  lda(train, cont[ind,1])
  pr<-  predict(z, test)
  class<- (pr$posterior[,2]>0.06)
  class<-factor(class,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
  t<-table(cont[notin,1],class)
  sens<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
  acc<-sum(diag(t))/nrow(test)
  bootLDA[j,]<- c(sens,   acc)
  if(j %% 10 == 0){
    cat(paste0("Progress: ", round(j/1,2),"%."),"\n")
  }
}
rownames(bootLDA)<-1:B ; colnames(bootLDA)=c('Sensitivity','Accuracy') ; head(bootLDA)

par(mfrow=c(1,2))
hist(bootLDA[,1],xlab='Sensitivity',col='darkorange',main='')
hist(bootLDA[,2],xlab='Accuracy',col='darkblue',main='')
mtext('Histogram of the Test Sensitivity and Accuracy of 100 runs by Bootstrap (LDA)',side = 3,
      line = - 2,outer = TRUE,cex=1.5)

par(mfrow=c(1,1))
boxplot(bootLDA,main='Bootstrap',col=c('darkorange','darkblue'))

summary(bootLDA)


########### QDA ############
model<- qda(SUBSCRIBED~., data=cont)
pred<-predict(model,newdata = cont)
t<-table(cont[,1],pred$class)
t[2,2]/sum(t[2,]) # sensitivity in sample

####### ROC curve 
model<- qda(SUBSCRIBED~., data=cont)
pred<-predict(model,newdata = cont)
B<-10
variab<-2:3
res<-NULL

for (threshold in seq(0.01,0.9,by=0.01)){
bootQDA<-matrix(nrow = 10,ncol = 3)
  for ( j in 1:B) {
    ind<- sample(1:nrow(cont),nrow(cont), replace=TRUE)
    check<- 1:nrow(cont)
    t<- check %in% unique(ind)
    notin<- check[!t]
    
    train <- cont[ind,c(1,variab)]
    test <-   cont[notin,c(1,variab)]
    z <-  qda(SUBSCRIBED~., data=train)
    pr<-  predict(z,newdata = test)
    class<-factor(pr$posterior[,2]>threshold,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
    t<-table(cont[notin,1],class)
    sensBoot<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
    accBoot=sum(diag(t))/nrow(test)
    specBoot<- t[1,1]/apply(t,1,sum)[1]
    bootQDA[j,]<- c(sensBoot,accBoot,specBoot )
  }
  sens<-mean(bootQDA[,1])
  acc<- mean(bootQDA[,2])
  spec<- mean(bootQDA[,3])
  res<-rbind(res, c(sens,1-spec,acc))
  
  if( (10*threshold) %% 1 == 0){
    cat(paste0("Progress: ", round(threshold/0.009,2),"%."),"\n")
  }
}

colnames(res)<- c("Sensitivity", "1-Specificity",'Accuracy') ; rownames(res)=as.character(seq(0.01,0.9,by=0.01))
res
# plot the roc curve
plot(res[,2],res[,1], xlab="FN", ylab="TP", type="l",col=3, xlim=c(0,1), ylim=c(0,1),lwd=2,main = 'ROC curve')
abline(0,1)

seq(0.01,0.9,by=0.01)[which.max(res[,1])] # the threshold with the highest sensitivity

apply(res[,1:2],1,function(x) eDis(x,c(0,1))) # calculate the euclidian dis from (0,1)
res[ which.max(apply(res[,1:2],1,function(x) eDis(x,c(0,1)))) ,] #  I choose 0.07


# run it many times
B<-1000
resultQDA<-matrix(nrow=1000,ncol = 2)
for ( i in 1:B) {
  ind<- sample(1:nrow(x),nTrain,replace=FALSE)
  train<- cont[ind,]
  test<- cont[-ind,]
  
  mq1<-qda(SUBSCRIBED~., data=train)
  mq2<-predict(mq1,newdata = test)
  class<-factor(mq2$posterior[,2]>0.07,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
  t<- table(test$SUBSCRIBED,class)
  resultQDA[i,]<-c(t[2,2]/sum(t[2,]),sum(diag(t))/nrow(test))
  if(i %% 100 == 0){
    cat(paste0("Progress: ", round(i/10,2),"%."),"\n")
  }
}
colnames(resultQDA)=c('Sensitivity','Accuracy') ; head(resultQDA)

hist(resultQDA[1,],main='Histogram of the Test Sensitivity of 1000 runs (QDA)')
hist(resultQDA[2,],main='Histogram of the Test Accuracy of 1000 runs (QDA)')

#####  k-fold cross validation 

deiktes<-sample(1:nrow(cont))
variab<-2:3
crossQDA<-matrix(ncol = 2)
folds<-c(2,3,4,5,7,10,12,15,20)

for (omades in folds) {  # omades is the number of folds
  k<-round((nrow(cont)/omades) - 1) # k is the number of observations per fold
  t<-matrix(nrow=1,ncol=2) ; t2=t3=NULL
  
  for (i in 1:omades) {
    te<- deiktes[ ((i-1)*k+1):(i*k-1)]
    train <- cont[-te,c(1,variab)]
    test <-   cont[te,c(1,variab)]
    z <-  qda(SUBSCRIBED~., data=train)
    pr<-  predict(z,newdata = test)
    class<-factor(pr$posterior[,2]>0.07,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
    t<-table(cont[te,1],class)
    t2<- c(t2,t[2,2]/sum(t[2,])) # sensitivity TP / totalTruePositives
    t3=c(t3,sum(diag(t))/nrow(test))
  }
  crossQDA<-rbind(crossQDA,c(mean(t2),mean(t3)))
}
crossQDA=crossQDA[-1,]
rownames(crossQDA)<-folds; colnames=c('Sensitivity','Accuracy') ; crossQDA

########### bootstrap 
B<-100
variab<-2:3
bootQDA<-matrix(nrow = 100,ncol = 2)

for ( j in 1:B) {
  ind<- sample(1:nrow(cont),nrow(cont), replace=TRUE)
  check<- 1:nrow(cont)
  t<- check %in% unique(ind)
  notin<- check[!t]
  
  train <- cont[ind,c(1,variab)]
  test <-   cont[notin,c(1,variab)]
  z <-  qda(SUBSCRIBED~., data=train)
  pr<-  predict(z,newdata = test)
  class<-factor(pr$posterior[,2]>0.07,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
  t<-table(cont[notin,1],class)
  sens<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
  acc=sum(diag(t))/nrow(test)
  bootQDA[j,]<- c(sens,acc   )
  if(j %% 10 == 0){
    cat(paste0("Progress: ", round(j/1,2),"%."),"\n")
  }
}
rownames(bootQDA)<-1:B ; colnames(bootQDA)=c('Sensitivity','Accuracy') ; head(bootQDA)

par(mfrow=c(1,2))
hist(bootQDA[,1],xlab='Sensitivity',col='darkorange',main='')
hist(bootQDA[,2],xlab='Accuracy',col='darkblue',main='')
mtext('Histogram of the Test Sensitivity and Accuracy of 100 runs by Bootstrap (QDA)',side = 3,
      line = - 2,outer = TRUE,cex=1.5)

par(mfrow=c(1,1))
boxplot(bootQDA,main='Bootstrap',col=c('darkorange','darkblue'))

summary(bootQDA)

########## Linear regression (LPM) ##########
Y1<- 1*(x$SUBSCRIBED=='yes') # transform the rensponce variab. to 0-1 scale

mod1<- lm(Y1 ~. ,data = x[,-21])

ypred<- (mod1$fitted>=0.5)*1
#all(ypred==(predict(mod1,newdata = x[,-21])>=0.5)*1)

t<-table(x$SUBSCRIBED, ypred ) ; t # confusion matrix
t[2,2]/sum(t[2,]) # in sample sens

summary(mod1)
anova(mod1)

# Variable selection stepwise
Y1<- 1*(x$SUBSCRIBED=='yes')
first<-c()

for( i in 1 : (dim(x)[2]-1) ){
  model<-lm(as.formula( paste('Y1 ~',names(x[,-21])[i],sep = ' ') ),data = x)
  clas<- (model$fitted >= 0.5)*1
  t<-table(Y1,clas)
  acc<- sum(diag(t))/nrow(x)
  first<-c(first,acc)  
}

names(first)<-names(x[,-21]) ; first

### Step algorithm

prevModel<-paste('Y1 ~ ',names(x[,-21])[which.max(first)],sep='') # poutcome
selVarLPM<-c(which.max(first)) # the variables I have selected so far
prevAcc<-c(0,min(first)) # vector that saves the best Acc for models with same number of var
counter = 1

while( prevAcc[counter+1] > prevAcc[counter] ){
  stepAcc<-c()
  var<-c()
  counter<-counter+1
  for(i in ( 1 : (dim(x)[2]-1) ) [-selVarLPM]){
    
    model<-lm(as.formula(paste(prevModel,paste(names(x[,-21])[i],sep=''),sep=' + ') ), data = x)
    var<-c(var,names(x[,-21])[i]) # the name of the variable I add
    clas<- (model$fitted >= 0.5)*1
    
    t<-table(Y1,clas)
    acc<- sum(diag(t))/nrow(x)
    stepAcc<-c(stepAcc,acc) 
  }
  names(stepAcc) <- var ; stepAcc 
  prevAcc<-c(prevAcc,max(stepAcc)) ; names(prevAcc)<-0:counter ; prevAcc
  selVarLPM<-c(selVarLPM,which(names(x) == names(stepAcc)[which.max(stepAcc)] )  )
  prevModel<-paste(prevModel,paste(names(x[,-21])[tail(selVarLPM,1)],sep=''),sep=' + ')
}

selVarLPM<-selVarLPM[-length(selVarLPM)] ; names(selVarLPM) <- names(x[,-21])[selVarLPM]  
selVarLPM # the variable Ι selected | selVarLPM <- c(13,11,20,9,4,5,6)
prevAcc<-prevAcc[-length(prevAcc)] 
prevAcc # the acc on the number of var I have

# The important predictor variables based on the step algorithm are 
selVarLPM # with acc
prevAcc[length(prevAcc)]


#######  brier score  
### we need package scoring and soft clustering, i.e. probabilities
sum(brierscore(Y1~mod1$fitted)) # measures the accuracy of the probabilistic
                                # predictions

####### ROC curve 

Y1<- 1*(x$SUBSCRIBED=='yes')
B<-10
res<-NULL

for (threshold in seq(0.01,0.9,by=0.01)){
bootLPM<-matrix(nrow = 10,ncol = 3)
  for ( j in 1:B) {
    ind<- sample(1:nrow(cont),nrow(cont), replace=TRUE)
    check<- 1:nrow(cont)
    t<- check %in% unique(ind)
    notin<- check[!t]
    
    train <- x[c(ind,sample(which(x$default=='yes'),1)),]
    test <- x[notin,]
    
    Y1<- 1*(train$SUBSCRIBED=='yes') 
    
    z <-lm(Y1 ~. ,data = train[,selVarLPM]) 
    
    pr<-( predict(z,newdata = test[,selVarLPM] )>= threshold )*1
    t<-table(test[,21],pr)
    sensBoot<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
    accBoot<- sum(diag(t))/nrow(test)
    specBoot<- t[1,1]/apply(t,1,sum)[1]
    bootLPM[j,]<- c(sensBoot,accBoot,specBoot)
    }
  
  sens<- mean(bootLPM[,1])
  acc<- mean(bootLPM[,2])
  spec<- mean(bootLPM[,3])
  res<-rbind(res, c(sens,1-spec,acc))
  if( (10*threshold) %% 1 == 0){
    cat(paste0("Progress: ", round( threshold/0.009,2),"%."),"\n")
  }
}
colnames(res)<- c("Sensitivity", "1-Specificity",'Accuracy') ; rownames(res)=as.character(seq(0.01,0.9,by=0.01))
res
# plot the roc curve
plot(res[,2],res[,1], xlab="FN", ylab="TP", type="l",col=3, xlim=c(0,1), ylim=c(0,1),lwd=2,main = 'ROC curve')
abline(0,1)

seq(0.01,0.9,by=0.01)[which.max(res[,1])] #the threshold with the highest sensitivity

apply(res[,1:2],1,function(x) eDis(x,c(0,1))) # calculate the euclidian dis from (0,1)
res[ which.max(apply(res[,1:2],1,function(x) eDis(x,c(0,1)))) ,] #  I choose 0.14


#### run it many times 
resultLPM<-matrix(nrow = 100,ncol = 2)
for(i in 1:100){
  ind<-sample(1:nrow(x),nTrain-1,replace = FALSE)
  train<-x[c(ind,sample(which(x$default=='yes'),1) ),] # the default has only 3 yes
  test<-x[-c(ind,sample(which(x$default=='yes'),1) ),] # sometimes the the train data doesnt have such observation
                          # and it cant predict without the coefficient so i put one on purpose
  Y1<- 1*(train$SUBSCRIBED=='yes') 
  
   mod1<- lm(Y1 ~.,data = train[,selVarLPM])
  
  ypred<- (predict(mod1,newdata = test[,-21])>=0.18)*1
  t<-table(test$SUBSCRIBED,ypred) # confusion matrix
  resultLPM[i,]<-c(t[2,2]/sum(t[2,]),sum(diag(t))/nrow(test))
  if(i %% 10 == 0){
    cat(paste0("Progress: ", round(i/1,2),"%."),"\n")
  }
}
colnames(resultLPM)<-c('Sensitivity','Accuracy') ; head(resultLPM)

hist(resultLPM[,1],main='Histogram of the Test Sensitivity of 1000 runs (LPM)')
hist(resultLPM[,2],main='Histogram of the Test Accuracy of 1000 runs (LPM)')

#####  k-fold cross validation 

deiktes<-sample(1:nrow(cont))
crossLPM<-matrix(ncol=2)
folds<-c(2,3,4,5,7,10,12,15,20)

for (omades in folds) {  # omades is the number of folds
  k<-round((nrow(cont)/omades) - 1) # k is the number of observations per fold
  t<-matrix(nrow=1,ncol=2) ; t2=NULL
  
  for (i in 1:omades) {
    te<- deiktes[ -(((i-1)*k+1):(i*k-1))]
    train <- x[c(te,sample(which(x$default=='yes'),1) ),]
    test <- x[-c(te,sample(which(x$default=='yes'),1) ),]
    
    Y1<- 1*(train$SUBSCRIBED=='yes') 
    z <-  lm(Y1 ~.,data = train[,selVarLPM])
    
    pr<-  (predict(z,newdata = test[,-21])>=0.18)*1
    t<-table(test$SUBSCRIBED,pr)
    t2<- c(t2,t[2,2]/sum(t[2,])) # sensitivity TP / totalTruePositives
    t3<-c(t3,sum(diag(t))/nrow(test))
  }
  crossLPM<-rbind(crossLPM,c(mean(t2),mean(t3)))
}
crossLPM<-crossLPM[-1,]
rownames(crossLPM)<-folds ; colnames(crossLPM)=c('Sensitivity','Accuracy') ; crossLPM

########### bootstrap 
B<-100
bootLPM<-matrix(nrow = 100,ncol = 2)

for ( j in 1:B) {
  ind<- sample(1:nrow(cont),nrow(cont), replace=TRUE)
  check<- 1:nrow(cont)
  t<- check %in% unique(ind)
  notin<- check[!t]
  
  train <- x[c(ind,sample(which(x$default=='yes'),1) ),]
  test <- x[notin,]
  
  Y1<- 1*(train$SUBSCRIBED=='yes') 
  
  z <-  lm(Y1 ~.,data = train[,selVarLPM])
  
  pr<-  (predict(z,newdata = test[,-21])>=0.18)*1
  t<-table(test[,21],pr)
  sens<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
  acc<-sum(diag(t))/nrow(test)
  bootLPM[j,]<- c(sens,   acc)
  if(j %% 10 == 0){
    cat(paste0("Progress: ", round(j/1,2),"%."),"\n")
  }
}
rownames(bootLPM)<-1:B ; colnames(bootLPM)=c('Sensitivity','Accuracy') ; bootLPM

par(mfrow=c(1,2))
hist(bootLPM[,1],xlab='Sensitivity',col='darkorange',main='')
hist(bootLPM[,2],xlab='Accuracy',col='darkblue',main='')
mtext('Histogram of the Test Sensitivity and Accuracy of 100 runs by Bootstrap (LPM)',side = 3,
      line = - 2,outer = TRUE,cex=1.5)

par(mfrow=c(1,1))
boxplot(bootLPM,main='Bootstrap',col=c('darkorange','darkblue'))

summary(bootLPM)


#####  logistic regression ######################
Y1<- 1*(x$SUBSCRIBED=='yes')

mbin=glm(Y1~., family='binomial',data = x[,-21])
summary(mbin)
t<-table(Y1,round(mbin$fitted)) ; t
t[2,2]/sum(t[2,])

# Variable selection stepwise
Y1<- 1*(x$SUBSCRIBED=='yes')
first<-c()

for( i in 1 : (dim(x)[2]-1) ){
  model<-glm(as.formula( paste('Y1 ~',names(x[,-21])[i],sep = ' ') ),family = 'binomial',data = x)
  clas<- (model$fitted >= 0.5)*1
  t<-table(Y1,clas)
  acc<- sum(diag(t))/nrow(x)
  first<-c(first,acc)  
}

names(first)<-names(x[,-21]) ; first

### Step algorithm

prevModel<-paste('Y1 ~ ',names(x[,-21])[which.max(first)],sep='') # poutcome
selVarGLM<-c(which.max(first)) # the variables I have selected so far
prevAcc<-c(0,min(first)) # vector that saves the best Acc for models with same number of var
counter = 1

while( prevAcc[counter+1] > prevAcc[counter] ){
  stepAcc<-c()
  var<-c()
  counter<-counter+1
  for(i in ( 1 : (dim(x)[2]-1) ) [-selVarGLM]){
    
    model<-glm(as.formula(paste(prevModel,paste(names(x[,-21])[i],sep=''),sep=' + ') ), 
               family='binomial',data = x)
    var<-c(var,names(x[,-21])[i]) # the name of the variable I add
    clas<- (model$fitted >= 0.5)*1
    
    t<-table(Y1,clas)
    acc<- sum(diag(t))/nrow(x)
    stepAcc<-c(stepAcc,acc) 
  }
  names(stepAcc) <- var ; stepAcc 
  prevAcc<-c(prevAcc,max(stepAcc)) ; names(prevAcc)<-0:counter ; prevAcc
  selVarGLM<-c(selVarGLM,which(names(x) == names(stepAcc)[which.max(stepAcc)] )  )
  prevModel<-paste(prevModel,paste(names(x[,-21])[tail(selVarGLM,1)],sep=''),sep=' + ')
}

selVarGLM<-selVarGLM[-length(selVarGLM)] ; names(selVarGLM) <- names(x[,-21])[selVarGLM]  
selVarGLM # the variable i selected |  selVarGLM<- c(11,15,20,9,10,8,13 )
prevAcc<-prevAcc[-length(prevAcc)] 
prevAcc # the acc on the number of var I have

# The important predictor variables based on the step algorithm are 
selVarGLM # with acc
prevAcc[length(prevAcc)]


#######  brier score  
### we need package scoring and soft clustering, i.e. probabilities
sum(brierscore(Y1~mbin$fitted))

####### ROC curve 
res<-NULL
B<-5
for (threshold in seq(0.01,0.9,by=0.01)){
  bootLR<-matrix(nrow=B,ncol=3)
  for ( j in 1:B) {
  
    ind<- c(sample(1:nrow(cont),nrow(cont), replace=TRUE),sample(which(x$default=='yes'),1) )
    check<- 1:nrow(cont)
    t<- check %in% unique(ind)
    notin<- check[!t]
    
    train <- x[ind,]
    test <- x[notin,]
    
    Y1<- 1*(train$SUBSCRIBED=='yes') 
    
    z=glm(Y1~., family=binomial ,data = train[,selVarGLM])
    pr<- (predict(z,newdata = test[,selVarGLM],type='response')>=threshold)*1
    
    t<-table(test[,21],pr)
    sensBoot<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
    accBoot<-sum(diag(t))/nrow(test)
    specBoot<-t[1,1]/apply(t,1,sum)[1]
    bootLR[j,]<- c(sensBoot, accBoot,specBoot)
  }
  
  sens<-mean(bootLR[,1])
  acc<- mean(bootLR[,2])
  spec<- mean(bootLR[,3])
  res<-rbind(res, c(sens,1-spec,acc))
  
  if( (10*threshold) %% 1 == 0){
    cat(paste0("Progress: ", round(threshold/0.009,2),"%."),"\n")
  }
}
colnames(res)<- c("Sensitivity", "1-Specificity",'Accuracy') ; rownames(res)=as.character(seq(0.01,0.9,by=0.01))
res
# plot the roc curve
plot(res[,2],res[,1], xlab="FN", ylab="TP", type="l",col=3, xlim=c(0,1), ylim=c(0,1),lwd=2,main='ROC curve')
abline(0,1)

seq(0.01,0.9,by=0.01)[which.max(res[,1])] # the threshold with the highest sensitivity

apply(res[,1:2],1,function(x) eDis(x,c(0,1))) # calculate the euclidian dis from (0,1)
res[ which.max(apply(res[,1:2],1,function(x) eDis(x,c(0,1)))) ,] #  I choose 0.07

#### run it many times 
resultLR<-matrix(nrow=100,ncol=2)
for(i in 1:100){
  ind<-sample(1:nrow(x),nTrain-1,replace = FALSE)
  train<-x[c(ind,sample(which(x$default=='yes'),1) ),] # the default has only 3 yes
  test<-x[-c(ind,sample(which(x$default=='yes'),1) ),] # sometimes the the train data doesnt have such observation
                          # and it cant predict without the coefficient so i put one on purpose
  Y1<- 1*(train$SUBSCRIBED=='yes')
  
   mbin=glm(Y1~., family=binomial ,data = train[,selVarGLM])
  
  ypred<- (predict(mbin,newdata = test[,selVarGLM],type='response')>=0.07)*1
  t<-table(test$SUBSCRIBED,ypred) # confusion matrix
  resultLR[i,]<-c(t[2,2]/sum(t[2,]),sum(diag(t))/nrow(test))
  if(i %% 10 == 0){
    cat(paste0("Progress: ", round(i,2),"%."),"\n")
  }
}
colnames(resultLR)<-c('Sensitivity','Accuracy') ; head(resultLR)

hist(resultLR[,1],main='Histogram of the Test Sensitivity of 100 runs (LR)')
hist(resultLR[,2],main='Histogram of the Test Accuracy of 100 runs (LR)')

#####  k-fold cross validation 

deiktes<-sample(1:nrow(cont))
crossLR<-matrix(ncol=2)
folds<-c(2,3,4,5,7,10,12,15,20)

for (omades in folds) {  # omades is the number of folds
  k<-round((nrow(cont)/omades) - 1) # k is the number of observations per fold
  t<-matrix(nrow=1,ncol=2) ; t2=t3=NULL
  
  for (i in 1:omades) {
    te<- deiktes[ -(((i-1)*k+1):(i*k-1))]
    train <- x[c(te,sample(which(x$default=='yes'),1) ),]
    test <- x[-c(te,sample(which(x$default=='yes'),1) ),]
    
    Y1<- 1*(train$SUBSCRIBED=='yes') 
    
     z=glm(Y1~., family=binomial,data = train[,selVarGLM])
    
    
    pr<- (predict(z,newdata = test[,selVarGLM],type='response')>=0.07)*1
    t<-table(test$SUBSCRIBED,pr)
    t2<- c(t2,t[2,2]/sum(t[2,])) # sensitivity TP / totalTruePositives
    t3<-c(t3,sum(diag(t))/nrow(test))
  }
  crossLR<-rbind(crossLR,c(mean(t2),mean(t3)))
}
crossLR<-crossLR[-1,]
rownames(crossLR)<-folds ; colnames(crossLR)<-c('Sensitivity','Accuracy') ; crossLR

########### bootstrap 
B<-100
bootLR<-matrix(nrow=B,ncol=2)

for ( j in 1:B) {
  ind<- c(sample(1:nrow(cont),nrow(cont), replace=TRUE),sample(which(x$default=='yes'),1) )
  check<- 1:nrow(cont)
  t<- check %in% unique(ind)
  notin<- check[!t]
  
  train <- x[ind,]
  test <- x[notin,]
  
  Y1<- 1*(train$SUBSCRIBED=='yes') 
  
   z=glm(Y1~., family=binomial,data = train[,selVarGLM])
  
  
  pr<- (predict(z,newdata = test[,selVarGLM],type='response')>=0.07)*1
  
  t<-table(test[,21],pr)
  sens<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
  acc<-sum(diag(t))/nrow(test)
  bootLR[j,]<- c(sens, acc)
  if(j %% 10 == 0){
    cat(paste0("Progress: ", round(j/1,2),"%."),"\n")
  }
}
rownames(bootLR)<-1:B; colnames(bootLR)=c('Sensitivity','Accuracy') ; bootLR

par(mfrow=c(1,2))
hist(bootLR[,1],xlab='Sensitivity',col='darkorange',main='')
hist(bootLR[,2],xlab='Accuracy',col='darkblue',main='')
mtext('Histogram of the Test Sensitivity and Accuracy of 100 runs by Bootstrap (LR)',side = 3,
      line = - 2,outer = TRUE,cex=1.5)

par(mfrow=c(1,1))
boxplot(bootLR,main='Bootstrap',col=c('darkorange','darkblue'))

summary(bootLR)

########  knn  ############
### wanring only euclidean distance
### we create a dataset with scaled data

sCont<- cbind(cont[,1],scale(cont[,-1]))
### knn with k=3, standardize data

km1s<-knn(sCont[,2:3],sCont[,2:3], cl=sCont[,1],k=3)

t<- table(sCont[,1],km1s) ; t
t[2,2]/sum(t[2,])

# find the best k
scoreK<-NULL
for ( i in 1:10){
  km1s<-knn(sCont[,2:3],sCont[,2:3], cl=sCont[,1],k=i)
  t<- table(sCont[,1],km1s) ; t
  scoreK<-c(scoreK,t[2,2]/sum(t[2,]))
    cat(paste0("Progress: ", round(i*10,2),"%."),"\n")
}
plot(scoreK,pch=18,type='b') # k = 4

#### different test and training
ind<- sample(1:nrow(x),nTrain,replace=FALSE)
train<- sCont[ind,]
test<- sCont[-ind,]

km1<-knn(train[,2:3],test[,2:3], cl=train[,1],k=4)
t<-table(test[,1],km1) ; t
sum(diag(t))/nrow(test)

#### soft knn classification ####@###
km2<-knn(sCont[,2:3],sCont[,2:3], cl=sCont[,1],k=3 ,prob=TRUE)
table(sCont[,1],km2)
knn.cv(cont[,2:3], cl=cont[,1],k=3) # leave one out cross validation
attributes(km2)$prob # the probabilities

#####  k-fold cross validation 
deiktes<-sample(1:nrow(cont))
crossKnn<-NULL
folds<-c(2,3,4,5,7,10,12,15,20)

for (omades in folds) {  # omades is the number of folds
  k<-round((nrow(cont)/omades) - 1) # k is the number of observations per fold
  t<-matrix(nrow=1,ncol=2) ; t2=NULL
  
  for (i in 1:omades) {
    te<- deiktes[ ((i-1)*k+1):(i*k-1)]
    train <- sCont[-te,]
    test <- sCont[te,]
    
    z=knn(train[,2:3],test[,2:3], cl=train[,1],k=4)
    t<-table(test[,1],z)
    t2<- c(t2,t[2,2]/sum(t[2,])) # sensitivity TP / totalTruePositives
  }
  crossKnn<-c(crossKnn,mean(t2))
}
names(crossKnn)<-folds ; crossKnn

########### bootstrap 
B<-100
bootKnn<-NULL

for ( j in 1:B) {
  ind<- sample(1:nrow(cont),nrow(cont), replace=TRUE)
  check<- 1:nrow(cont)
  t<- check %in% unique(ind)
  notin<- check[!t]
  
  train <- sCont[ind,]
  test <- sCont[notin,]
  
  z=knn(train[,2:3],test[,2:3], cl=train[,1],k=4)
  
  t<-table(test[,1],z)
  accur<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
  bootKnn<- c(bootKnn,   accur)
  if(j %% 10 == 0){
    cat(paste0("Progress: ", round(j/1,2),"%."),"\n")
  }
}
names(bootKnn)<-1:B ; bootKnn
hist(bootKnn,main='Histogram of the Test Sensitivity of 100 runs by Bootstrap (Knn)')

#########  tree ###############

fit1<-tree(SUBSCRIBED~.,data=x)
fit1$frame 
fit1$where # where each observation was classified in the tree
t<-table(x$SUBSCRIBED, predict(fit1,type='class')) ; t
t[2,2]/sum(t[2,])

# plot the tree
plot(fit1)
summary(fit1)
text(fit1)
# the true classification | the predicted | 0 = 'correct' 1 = 'wrong'
cbind(x$SUBSCRIBED,predict(fit1,type="class"), resid(fit1))
# sum(sapply(wines$Type,function(x)(x-mean(wines$Type))^2)) IDK what that is
# sum(sapply(resid(fit1),function(x)(x-mean(resid(fit1)))^2))

fit2<-tree(SUBSCRIBED~.,data=x, split="gini")
plot(fit2)
text(fit2)


predict(fit1,type='vector') # probabilities
predict(fit1,type='class') # classification
predict(fit1,type='tree') # the frame of the tree

####### ROC curve 
res<-NULL
B<-5
for (threshold in seq(0.01,0.9,by=0.01)){
  bootTree<-matrix(nrow = B,ncol=3)
  for ( j in 1:B) {
    ind<- c(sample(1:nrow(cont),nrow(cont), replace=TRUE),21581)
    check<- 1:nrow(cont)
    t<- check %in% unique(ind)
    notin<- check[!t]
    
    train <- x[ind,]
    test <- x[notin,]
    
    z=tree(SUBSCRIBED~.,data=train)
    pr<- (predict(z,type='vector',newdata = test[,-21])[,2] > threshold)
    pr<-factor(pr,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
    
    t<-table(test[,21],pr)
    sensBoot<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
    accBoot<-sum(diag(t))/nrow(test)
    specBoot<-t[1,1]/apply(t,1,sum)[1]
    bootTree[j,]<- c(sensBoot,accBoot,specBoot)
  }
  
  sens<-mean(bootTree[,1])
  acc<- mean(bootTree[,2])
  spec<- mean(bootTree[,3])
  res<-rbind(res, c(sens,1-spec,acc))
  if( (10*threshold) %% 1 == 0){
    cat(paste0("Progress: ", round(threshold/0.009,2),"%."),"\n")
  }
}
colnames(res)<- c("Sensitivity", "1-Specificity",'Accuracy') ; rownames(res)=as.character(seq(0.01,0.9,by=0.01))
res
# plot the roc curve
plot(res[,2],res[,1], xlab="FN", ylab="TP", type="l",col=3, xlim=c(0,1), ylim=c(0,1),lwd=2)
abline(0,1)

seq(0.01,0.9,by=0.01)[which.max(res[,1])] # the threshold with the highest sensitivity

apply(res[,1:2],1,function(x) eDis(x,c(0,1))) # calculate the euclidian dis from (0,1)
res[ which.max(apply(res[,1:2],1,function(x) eDis(x,c(0,1)))) ,] #  I choose 0.15

# run it many times with train & test
resultTreeSens=resultTreeAcc<-NULL
for(i in 1:100){
  ind<-sample(1:nrow(x),nTrain-1,replace = FALSE)
  train<-x[c(ind,21581),] # the default has only 3 yes
  test<-x[-c(ind,21581),] # sometimes the the train data doesnt have such observation
                            # and it cant predict without the coefficient so i put one on purpose
  fit1<-tree(SUBSCRIBED~.,data=train)
  ypred<- (predict(fit1,type='vector',newdata = test[,-21])[,2] > 0.15)
  ypred<-factor(ypred,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
  t<-table( test$SUBSCRIBED,ypred) # confusion matrix
  resultTreeSens<-c(resultTreeSens,t[2,2]/sum(t[2,]))
  resultTreeAcc<-c(resultTreeAcc,sum(diag(t))/nrow(test))
  
  if(i %% 10 == 0){
    cat(paste0("Progress: ", round(i,2),"%."),"\n")
  }
}
hist(resultTreeSens,main='Histogram of the Test Sensitivity of 1000 runs (Tree)')
hist(resultTreeAcc,main='Histogram of the Test Accuracy of 1000 runs (Tree)')
#####  k-fold cross validation 

deiktes<-sample(1:nrow(cont))
crossTree<-matrix(nrow=1,ncol=2)
folds<-c(2,3,4,5,7,10,12,15,20)

for (omades in folds) {  # omades is the number of folds
  k<-round((nrow(cont)/omades) - 1) # k is the number of observations per fold
  t<-matrix(nrow=1,ncol=2) ; t2=t3=NULL
  
  for (i in 1:omades) {
    te<- deiktes[ -(((i-1)*k+1):(i*k-1))]
    train <- x[c(te,21581),]
    test <- x[-c(te,21581),]
    
    z=tree(SUBSCRIBED~.,data=train)
    pr<- (predict(z,type='vector',newdata = test[,-21])[,2] > 0.15)
    pr<-factor(pr,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
    t<-table(test$SUBSCRIBED,pr)
    t2<- c(t2,t[2,2]/sum(t[2,])) # sensitivity TP / totalTruePositives
    t3<-c(t3,sum(diag(t))/nrow(test))
  }
  crossTree<-rbind(crossTree,c(mean(t2),mean(t3)))
}
crossTree<-crossTree[-1,] 
rownames(crossTree)<-folds ; colnames(crossTree)<- c('Sensitivity','Accuracy')
crossTree

########### bootstrap 
B<-100
bootTree<-matrix(nrow = B,ncol=2)

for ( j in 1:B) {
  ind<- c(sample(1:nrow(cont),nrow(cont), replace=TRUE),21581)
  check<- 1:nrow(cont)
  t<- check %in% unique(ind)
  notin<- check[!t]
  
  train <- x[ind,]
  test <- x[notin,]
  
  z=tree(SUBSCRIBED~.,data=train)
  pr<- (predict(z,type='vector',newdata = test[,-21])[,2] > 0.15)
  pr<-factor(pr,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
  
  t<-table(test[,21],pr)
  sens<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
  acc<-sum(diag(t))/nrow(test)
  bootTree[j,]<- c(sens,acc)
  if(j %% 10 == 0){
    cat(paste0("Progress: ", round(j/1,2),"%."),"\n")
  }
}
rownames(bootTree)<-1:B ; colnames(bootTree)<-c('Sensitivity','Accuracy')  ; bootTree
hist(bootTree[,1],main='Histogram of the Test Sensitivity of 100 runs by Bootstrap (Tree)')
hist(bootTree[,2],main='Histogram of the Test Accuracy of 100 runs by Bootstrap  bootTree(Tree)')

############# Random Forest ################
myRF<- randomForest(SUBSCRIBED ~ ., data=x, ntree=100,importance=TRUE,cutoff=c(0.99,0.01))

myRF

t<-table(x$SUBSCRIBED,myRF$predicted)
t[2,2]/sum(t[2,]) # sens
sum(diag(t))/nrow(x) # acc
1 - t[1,1]/apply(t,1,sum)[1] # 1-spec

## the reported oob error is the last one
## each row shows the error up to this tree
##  not the error of the specific tree

plot(myRF)
myRF$predicted

t<-table(x$SUBSCRIBED,myRF$predicted) # confusion matrix
t[2,2]/sum(t[2,]) # sensitivity based on oob out of bag sampling (bootstrap)

myRF$importance # importance of each variable

myRF$importance[order(myRF$importance[,3],decreasing = TRUE),] # ordered based on acc (better var from top to bottom)
myRF$importance[order(myRF$importance[,4],decreasing = TRUE),] # ordered based on acc (better var from top to bottom)

myRF$err.rate # error rates for all trees up to the i-th. use it to figure out how many trees i need
myRF$votes # precentage of the votes of the trees for each class

cbind(myRF$votes,true=x$SUBSCRIBED,pred=myRF$predicted)[x$SUBSCRIBED == 'yes',] # figure out the 
                                                                                # prob of classification
myRF$mtry # number of variable sampled for each tree
predict(myRF,x[1:10,])

# How to find the hyperparameters, m = mtry & ntree
sens<-NULL
for (ntree in c(50,100,200,300)) {
  for (mtry in c(3,4,5)) {
    
    myRF<- randomForest(SUBSCRIBED ~ ., data=x, ntree=ntree,mtry=mtry)
    t<-table(x$SUBSCRIBED,myRF$predicted) # confusion matrix
    t[2,2]/sum(t[2,]) # sensitivity based on oob out of bag sampling (bootstrap)
    
    sens<-c(sens,t[2,2]/sum(t[2,]))
  }}

resultsRfM<- matrix(sens,byrow=TRUE,4,3)
colnames(resultsRfM)<- c(3,4,5)
rownames(resultsRfM) <- c(50,100,200,300)
resultsRfM # ntree = 300 & m = 5

# save the results
save(resultsRfM,file="C:/Users/USER/Desktop/classification/resultsRfM")
load("C:/Users/USER/Desktop/classification/resultsRfM")


# Run it many times
for ( i in 1:10){
  myRF<- randomForest(SUBSCRIBED ~ ., data=x, ntree=100,importance=TRUE)
  t<-table(x$SUBSCRIBED,myRF$predicted) # confusion matrix
  sens<-c(sens,t[2,2]/sum(t[2,])) # sensitivity based on oob out of bag sampling (bootstrap)
    cat(paste0("Progress: ", round(10*i,2),"%."),"\n")
  }
hist(sens,main='Histogram of the Test Sensitivity of 10 runs (RandomForrest)')


########
myRF<- randomForest(SUBSCRIBED ~ ., data=x, ntree=100,importance=TRUE,cutoff=c(0.93,0.07))

predict(myRF,newdata = valid,type='prob',cutoff = c(0.93,0.07))
predRF<-predict(myRF,newdata = valid,type='response',cutoff = c(0.93,0.07))

t<-table(valid$SUBSCRIBED,predRF)
t[2,2]/sum(t[2,]) # sens
sum(diag(t))/nrow(valid) # acc
1 - t[1,1]/apply(t,1,sum)[1] # 1-spec

##########  SVM ###############
svm_model <- svm(SUBSCRIBED ~ ., data=cont,gamma=1.5)
summary(svm_model)
pred <- predict(svm_model,cont[,-1])

t<-table(cont$SUBSCRIBED,pred)
t[2,2]/sum(t[2,])

### different choices ######3
ind<- sample(1:nrow(cont),nTrain,replace=FALSE)
train<- cont[ind,]
test<- cont[-ind,]
score<-NULL
for (cost in c(0.5,1,1.5) ) { # takes a lot of time
  for (gamma in c(0.7,1,1.2,1.5,2)) {
    
    svm_model <- svm(SUBSCRIBED ~ ., data=train,gamma=gamma, cost=cost)
    pred<- predict(svm_model,test)
    score<- c(score, sum(diag(table(pred,test$SUBSCRIBED)))/(dim(test)[1]))
  }}

resultsSvmGamma<- matrix(score,byrow=TRUE,3,5)

colnames(resultsSvmGamma)<- c(0.7,1,1.2,1.5,2)
rownames(resultsSvmGamma) <- c(0.5,1,1.5)
resultsSvmGamma # gamma = 2 & cost = 1

#save the results
save(resultsSvmGamma,file="C:/Users/USER/Desktop/classification/resultsSvmGamma")
load("C:/Users/USER/Desktop/classification/resultsSvmGamma")

############ Naive Bayes ########################
nb<- naiveBayes(SUBSCRIBED~. , data=x)
nb$apriori
nb$tables
nb$levels
nb$call 


classBayes<-predict(nb,x)
t<-table(x$SUBSCRIBED,classBayes) ; t 
t[2,2]/sum(t[2,]) # in sample sensitivity

####  brier score  
### we need package scoring and soft clustering, i.e. probabilities
pred1<-predict(nb,x,type = 'raw')
sum(brierscore(x$SUBSCRIBED~pred1[,2])) # idk i tried

####### ROC curve 
res<-NULL
omades=10
deiktes<-sample(1:nrow(cont))
k<-round((nrow(cont)/omades) - 1) # k is the number of observations per fold

for (threshold in seq(0.01,0.9,by=0.01)){ # takes too long RUN with caution
  t2=t3=t4=NULL
  for (i in 1:omades) {
    te<- deiktes[ -(((i-1)*k+1):(i*k-1))]
    train <- x[c(te,sample(which(x$default=='yes'),1) ),]
    test <- x[-c(te,sample(which(x$default=='yes'),1) ),]
    
    z=naiveBayes(SUBSCRIBED~. , data=train)
    pred<-predict(z,newdata = test[,-21],type = 'raw')
    pr<- (pred[,2]>threshold)
    pr<-factor(pr,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
    t<-table(test$SUBSCRIBED,pr)
    t2<- c(t2,t[2,2]/sum(t[2,])) # sensitivity TP / totalTruePositives
    t3<-c(t3,sum(diag(t))/nrow(test))
    t4<-c(t4,t[1,1]/apply(t,1,sum)[1])
  }
  sens<- mean(t2)
  acc<- mean(t3)
  spec<- mean(t4)
  res<-rbind(res, c(sens,1-spec,acc))
  
  if( (10*threshold) %% 1 == 0){
    cat(paste0("Progress: ", round(threshold/0.009,2),"%."),"\n")
  }
}
colnames(res)<- c("Sensitivity", "1-Specificity",'Accuracy') ; rownames(res)=as.character(seq(0.01,0.9,by=0.01))
res

# plot the roc curve
plot(res[,2],res[,1], xlab="FN", ylab="TP", type="l",col=3, xlim=c(0,1), ylim=c(0,1),lwd=2)
abline(0,1)

seq(0.01,0.9,by=0.01)[which.max(res[,1])] # the threshold with the highest sensitivity

apply(res[,1:2],1,function(x) eDis(x,c(0,1))) # calculate the euclidian dis from (0,1)
res[ which.max(apply(res[,1:2],1,function(x) eDis(x,c(0,1)))) ,] #  I choose 0.16

# Run it many times
resultBayesSens=resultBayesAcc<-NULL
for(i in 1:100){
  ind<-sample(1:nrow(x),nTrain-1,replace = FALSE)
  train<-x[c(ind,sample(which(x$default=='yes'),1) ),] # the default has only 3 yes
  test<-x[-c(ind,sample(which(x$default=='yes'),1) ),] # sometimes the the train data doesnt have such observation
                          # and it cant predict without the coefficient so i put one on purpose
  nb<- naiveBayes(SUBSCRIBED~. , data=train)
  pred<-predict(nb,newdata = test[,-21],type = 'raw')
  ypred<- (pred[,2]>0.16)*1
  t<-table(test$SUBSCRIBED,ypred) # confusion matrix
  resultBayesSens<-c(resultBayes,t[2,2]/sum(t[2,]))
  resultBayesAcc<-c(resultBayesAcc,sum(diag(t))/nrow(test))
  if(i %% 10 == 0){
    cat(paste0("Progress: ", round(i,2),"%."),"\n")
  }
}
hist(resultBayesSens,main='Histogram of the Test Sensitivity of 100 runs (Naive bayes)')
hist(resultBayesAcc,main='Histogram of the Test Accuracy of 100 runs (Naive bayes)')
summary(resultBayes)

#####  k-fold cross validation 

deiktes<-sample(1:nrow(cont))
crossBayes<-matrix(ncol=2)
folds<-c(2,3,4,5,7,10,12,15,20)

for (omades in folds) {  # omades is the number of folds
  k<-round((nrow(x)/omades) - 1) # k is the number of observations per fold
  t2=t3=NULL
  
  for (i in 1:omades) {
    te<- deiktes[ -(((i-1)*k+1):(i*k-1))]
    train <- x[c(te,sample(which(x$default=='yes'),1) ),]
    test <- x[-c(te,sample(which(x$default=='yes'),1) ),]
    
    z=naiveBayes(SUBSCRIBED~. , data=train)
    pred<-predict(z,newdata = test[,-21],type = 'raw')
    pr<- (pred[,2]>0.16)
    pr<-factor(pr,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
    t<-table(test$SUBSCRIBED,pr)
    t2<- c(t2,t[2,2]/sum(t[2,])) # sensitivity TP / totalTruePositives
    t3<-c(t3,sum(diag(t))/nrow(test))
  }
  crossBayes<-rbind(crossBayes,c(mean(t2),mean(t3)))
}
crossBayes<-crossBayes[-1,]
rownames(crossBayes)<-folds; colnames(crossBayes)<-c('Sensitivity','Accuracy') ; crossBayes

########### bootstrap 
B<-10
bootBayes<-matrix(nrow=B,ncol=2)

for ( j in 1:B) {
  ind<- c(sample(1:nrow(cont),nrow(cont), replace=TRUE),21581)
  check<- 1:nrow(cont)
  t<- check %in% unique(ind)
  notin<- check[!t]
  
  train <- x[ind,]
  test <- x[notin,]
  
  z=naiveBayes(SUBSCRIBED~. , data=train)
  pred<-predict(z,newdata = test[,-21],type = 'raw')
  pr<- (pred[,2]>0.16)
  pr<-factor(pr,levels = c(FALSE,TRUE),labels = c('noPred','yesPred'))
  
  t<-table(test[,21],pr)
  sens<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
  acc<-sum(diag(t))/nrow(test)
  bootBayes[j,]<- c(sens,   acc)
    cat(paste0("Progress: ", round(j/0.1,2),"%."),"\n")
}
rownames(bootBayes)<-1:B ; colnames(bootBayes)<-c('Sensitivity','Accuracy') ; bootBayes

hist(bootBayes[,1],main='Histogram of the Test Sensitivity of 100 runs by Bootstrap (Bayes)')
hist(bootBayes[,2],main='Histogram of the Test Accuracy of 100 runs by Bootstrap (Bayes)')

save(bootBayes,file='C:/Users/USER/Desktop/classification/bootBayes')
load('C:/Users/USER/Desktop/classification/bootBayes')

#### Validation #### 

########### bootstrap 
B<-100
boot2LR<-matrix(nrow=B,ncol=2)

for ( j in 1:B) {
  ind<- sample(1:nrow(valid),nrow(valid), replace=TRUE) 
  check<- 1:nrow(valid)
  t<- check %in% unique(ind)
  notin<- check[!t]
  
  train <- valid[ind,]
  test <- valid[notin,]
  
  Y1<- 1*(train$SUBSCRIBED=='yes') 
  
  z=glm(Y1~., family=binomial,data = train[,selVarGLM])
  
  
  pr<- (predict(z,newdata = test[,selVarGLM],type='response')>=0.07)*1
  
  t<-table(test[,21],pr)
  sens<- t[2,2]/sum(t[2,]) # sensitivity TP / totalTruePositives
  acc<-sum(diag(t))/nrow(test)
  boot2LR[j,]<- c(sens, acc)
  if(j %% 10 == 0){
    cat(paste0("Progress: ", round(j/1,2),"%."),"\n")
  }
}
rownames(boot2LR)<-1:B; colnames(boot2LR)=c('Sensitivity','Accuracy') ; boot2LR

par(mfrow=c(1,2))
hist(boot2LR[,1],xlab='Sensitivity',col='darkorange',main='')
hist(boot2LR[,2],xlab='Accuracy',col='darkblue',main='')
mtext('Histogram of the Test Sensitivity and Accuracy of 100 runs by Bootstrap (LR)',side = 3,
      line = - 2,outer = TRUE,cex=1.5)

par(mfrow=c(1,1))
boxplot(boot2LR,main='Bootstrap',col=c('darkorange','darkblue'))

summary(boot2LR)
