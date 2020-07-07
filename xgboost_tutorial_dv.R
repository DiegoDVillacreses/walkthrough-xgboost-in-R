######## 0) Load libraries #########################

rm(list = ls())

#To manage data (data.table > dplyr)
library(data.table)
  setDTthreads(percent = 100)
#To use pipes and manage strings
library(magrittr)
library(stringr)

#To load data from Stata, SPSS or SAS
require(haven)
require(labelled)

#To compute ROC 
library(pROC)

#Easy parallelization
library(doSNOW)
#Parallelization compatible with "mlr" library  
library(parallelMap)

#To compute xgboost
library(xgboost)
#To perform hyperparameter tuning
library(mlr)
  #I know this library is deprecated. In the future I hope I could
  #migrate this code to mlr3

######## 1) Load an example database #########################  
# As child malnutrition is a pretty difficult variable to predict
  # I'd like to present this example using data from Ecuador of 2018 on this topic:
dir.dt1 = "C:/Users/dvillacreses/Desktop/tmp_xgb_res"
dt1 = haven::read_dta(file.path(dir.dt1,"1_BDD_ENS2018_f1_personas.dta")) %>% setDT
# We have in a separate database information about physical house characteristics
dt1.aux = haven::read_dta(file.path(dir.dt1,"2_BDD_ENS2018_f1_hogar.dta"),
                          col_select = c("id_hogar","f1_s1_1","f1_s1_3","f1_s1_4",
                                         "f1_s1_5","f1_s1_11","f1_s1_13","f1_s1_21")) %>% setDT


dt1 %>% dim()
dt1 = merge(x = dt1, y = dt1.aux, by = "id_hogar")
dt1 %>% dim()
x.vars.h.c = c("f1_s1_1","f1_s1_3","f1_s1_4","f1_s1_5",
               "f1_s1_11","f1_s1_13","f1_s1_21")
dt1[,mget(x.vars.h.c)] %>% is.na %>% sum
  #A perfect merge.

# With labelled library we can create data.table to easily explore all the name's variables
# and it's original descriptions:
dt1.varL = labelled::var_label(dt1) %>% unlist
dt1.varL = data.table("var_name" = names(dt1.varL),
                      "var_label" = unname(dt1.varL))

######## 2) A basic data handling to have meaningful variables ######################### 

### This function takes a variable and repeats it for all members of a household
# Takes as input the information of one member of the household. 
# It's recommended to use information of household-head

var_to_jh = function(.dt,.y,.by,.dic.jh){
  get(.dt)[, .tmp01 := 0]
  get(.dt)[get(.dic.jh)==1, .tmp01 := get(.y)]
  get(.dt)[, .tmp01 := max(.tmp01), by = .by]
  
  .y.nn = paste0(.y,"_hh")
  get(.dt) %>% setnames(".tmp01",.y.nn)
}
### Compute a dummy variable which identifies the household-head 
dt1[, id.hh:=0]
dt1[f1_s2_7==1, id.hh:=1]

### Execute var_to_jh to a couple of variables
x.vars.to_hh = c("nivins","gedad_anios","edadanios","etnia","estado_civil", "sexo")
.dt1 = "dt1"
.by1 = "id_hogar"
.dic.jh1 = "id.hh"
for (.y1 in x.vars.to_hh){
  print(.y1)
  var_to_jh(.dt1,.y1,.by1,.dic.jh1)
}
#This new variables are going to have these names:
x.vars.hh = paste0(x.vars.to_hh,"_hh")
#Check if those variables are in dt1
(x.vars.hh %in% names(dt1)) %>% table
  #Perfect :)

### This variable should be numeric
dt1[,prov:=as.numeric(prov)]


### Now that I have household-head variables, I'll keep a couple
# of variables that affect all members of household as:
x.vars.all.hh = c("prov","area")

### And a couple of variables from the child
x.vars.child = c("sexo","edadmeses")

### Remember that we have information about the physical house characteristics
x.vars.h.c

### Merge all potential Xs variables in one vector
all.pot.Xs = c(x.vars.hh,x.vars.all.hh,x.vars.child,x.vars.h.c)

### Subset our database
dt1 %<>% .[!(is.na(dcronica)), mget(c("dcronica",all.pot.Xs))]
  # For sake of simplicity we are not going to take into consideration
  # that this database is a complex survey.

### We must drop these provinces as they are extremely different from the rest of the country
dt1 %<>% .[!(prov %in% c(24,90)),]

# For talla as a continuos variable
# dt1[,f1_s7_5_1] %>% summary
# dt1[,f1_s7_6_1] %>% summary


######## 8) XGBOOST #########################
######## 8.0) Initial parameters for hyperparameter tuning #########################

### Total iterations of different hyperparameters
max.iter.par.tun = 100 %>% as.integer()
  # This is a hard decision, ideally one should at least make 1000 steps. 
  # But that scenario could easily transform your "couple of hours" tuning into days

### Cross-Validation during tuning
# Total folds
folds.par.tun = 10L
# Total repetitions of folding process 
reps.par.tun  = 2L

### Within each xgboost:
# How many rounds should it try? 
nRound.w = 1*10^3 %>% as.integer()
  # This is other hard-to-know parameter. I'll leave it as one thousand for the sake of this
  # example. To have an approximate idea you could use xgb.cv with standard xgboost configuration
  # and under a try-and-error approach search for an approximately ideal value.
  # I´ve found that trying 100, 1000, 2000 and 10,000 in that order gave good results.
    # Obviously you stop trying (for example) in 100 if xgb.cv doesn't keep improving.

  # Unless you have access to a big server. Where you can leave running xgb.cv with a standard xgboost
  # under (for example) 100,000 rounds.

# After how many without rounds without improvement should stop?
EarlyStop.nR = 10 %>% as.integer()
  # Here we have the same problem as above

### Vectors containing names of y and Xs variables
.y.n = "dcronica"
.Xs.n = all.pot.Xs

### From here we are going to assume that our database is called: 
  # dt1

######## 8.1) Hyperparameter Tuning with mlr #########################

######### 8.1.1) Hyperparameter Space ~ Hyperparameter "candidates" ##################
xgb_params <- makeParamSet(
  # Learning rate ~ "shrinkage" parameter, prevents overfitting
  makeNumericParam("eta", lower = .01, upper = 0.7),
  
  # Number of splits in each tree
  makeIntegerParam("max_depth", lower = 3, upper = 9),

  # L2 regularization - prevents overfitting
  makeNumericParam("lambda", lower = 0, upper = 15),
  makeNumericParam("gamma", lower = 0, upper = 5),
  
  # Subsample ratio of columns for each level
  makeNumericParam("colsample_bylevel", lower = 0.01, upper = 1),
  # Subsample ratio of columns when constructing each tree
  #makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  
  #Minimum sum of instance weight (hessian) needed in a child
  #makeIntegerParam("min_child_weight", lower = 0, upper = 10),
  # Maximum delta step we allow each leaf output to be
  makeIntegerParam("max_delta_step", lower = 0, upper = 9)
  
  #Setting it to 0.5 means that XGBoost would randomly sample half of the training 
  #data prior to growing trees. and this will prevent overfitting
  #makeNumericParam("subsample", lower = 0.3, upper = 0.9)
  
)
#Reference: https://xgboost.readthedocs.io/en/latest/parameter.html

### From my personal research I've found that ideally one should
# take the Tuning on the following space. But that could require more time (easily 2 days) 
# on a "big" computer (year 2020) (30 cores of 2.4ghz and 35 gb RAM)
# Also take into account that as max_depth and max_delta_step gets bigger, also the
# boost problem, which would need more RAM and CPU power.

  # makeNumericParam("eta", lower = .01, upper = 0.7),
  # makeIntegerParam("max_depth", lower = 3, upper = 15),
  # makeNumericParam("lambda", lower = 0, upper = 15),
  # makeNumericParam("gamma", lower = 0, upper = 10),
  # makeNumericParam("colsample_bylevel", lower = 0.01, upper = 1),
  # makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  # makeIntegerParam("min_child_weight", lower = 0, upper = 10),
  # makeIntegerParam("max_delta_step", lower = 0, upper = 15),
  # makeNumericParam("subsample", lower = 0.3, upper = 0.9)

######### 8.1.2) Hyperparameter Tuning configuration ##################
##### 8.1.2.1) Total iterations of different hyperparameters & kind of search ######
control <- mlr::makeTuneControlRandom(maxit = max.iter.par.tun)

##### 8.1.2.2) Cross Validation configuration ######
resample_desc <- mlr::makeResampleDesc("RepCV", reps  = reps.par.tun,
                                       folds = folds.par.tun)

##### 8.1.2.3) User defined evaluation metric for xgboost. In this case: F1-Score ######
### F1-Score for a given cut point 
f1.score.dv = function(y_obs, y_pred_response, cut_point){
  tmp1 = data.table(
    y_pred = as.numeric(plogis(y_pred_response) > cut_point),
    y_real = y_obs
  ) %>% .[, .N, by = c("y_pred","y_real")]
  
  total_obs_positive = tmp1[y_real==1][,N] %>% sum()
  total_pred_positive = tmp1[y_pred==1][,N] %>% sum()
  total_true_pred_positive = tmp1[y_pred ==1 & y_real ==1][,N] %>% sum()
  .precision = total_true_pred_positive/total_pred_positive
  .recall = total_true_pred_positive/total_obs_positive
  f1.score = 2 * (.precision*.recall)/(.precision+.recall)
  if (is.na(f1.score)) f1.score = -1
  return(f1.score)
}
### F1-Score for cut points = seq(0,1,by = 0.01).

# Function compatible with xgboost package
evalerror.dv1 <- function(preds, dtrain) {
  y_obs <- xgboost::getinfo(dtrain, "label")
  y_pred_response <-preds
  
  f1.score.dv = function(y_obs, y_pred_response, cut_point){
    tmp1 = data.table(
      y_pred = as.numeric(plogis(y_pred_response) > cut_point),
      y_real = y_obs
    ) %>% .[, .N, by = c("y_pred","y_real")]
    
    total_obs_positive = tmp1[y_real ==1][,N] %>% sum()
    total_pred_positive = tmp1[y_pred ==1][,N] %>% sum()
    total_true_pred_positive = tmp1[y_pred ==1 & y_real ==1][,N] %>% sum()
    .precision = total_true_pred_positive/total_pred_positive
    .recall = total_true_pred_positive/total_obs_positive
    f1.score = 2 * (.precision*.recall)/(.precision+.recall)
    if (is.na(f1.score)) f1.score = -1
    return(f1.score)
  }

  tmp = sapply(seq(0,1,by = 0.01),function(x) 
    f1.score.dv(y_obs,y_pred_response,x))
  
  pos.max = which(tmp==max(tmp))
  if (length(pos.max)>1) pos.max = min(pos.max)
  .f1s = tmp[pos.max]
  return(list(metric = "F1_score", value = .f1s))
}

# Function compatible with mlr package
evalerror.dv2 <- function(task, model, pred, ...) {
  
  y_obs <- pred$data$truth
  y_pred_response <- pred$data$response
  
  f1.score.dv = function(y_obs, y_pred_response, cut_point){
    tmp1 = data.table(
      y_pred = as.numeric(plogis(y_pred_response) > cut_point),
      y_real = y_obs
    ) %>% .[, .N, by = c("y_pred","y_real")]
    tmp1
    total_obs_positive = tmp1[y_real ==1][,N] %>% sum()
    total_pred_positive = tmp1[y_pred ==1][,N] %>% sum()
    total_true_pred_positive = tmp1[y_pred ==1 & y_real ==1][,N] %>% sum()
    .precision = total_true_pred_positive/total_pred_positive
    .recall = total_true_pred_positive/total_obs_positive
    f1.score = 2 * (.precision*.recall)/(.precision+.recall)
    if (is.na(f1.score)) f1.score = -1
    return(f1.score)
  }
  
  tmp = sapply(seq(0,1,by = 0.1),function(x) 
    f1.score.dv(y_obs,y_pred_response,x))
  
  pos.max = which(tmp==max(tmp))
  if (length(pos.max)>1) pos.max = min(pos.max)
  .f1s = tmp[pos.max]
  return(.f1s)
}
m.dv = makeMeasure(id = "F1_score", minimize = F,
                   properties = c("regr", "response"), fun = evalerror.dv2)


##### 8.1.4) Here mlr configures which "learner" (and it's basic configuration) is going to use (xgboost, SVM, keras and so on...) ######
xgb_learner <- makeLearner(
  cl = "regr.xgboost",
  objective = "reg:squarederror",
  nrounds = nRound.w,
  early_stopping_rounds=EarlyStop.nR,
  predict.type = "response",
  eval_metric = evalerror.dv1,
  maximize  = T
)
### Just to see what's going on within this step
getLearnerProperties(xgb_learner)
getLearnerType(xgb_learner)
getHyperPars(xgb_learner)


##### 8.1.6) mlr and xgboost need an special data format, so: ######
  ### Just to have an idea of how the original data set looked like 
  dt1[, id.xgb := .I]

### We need to identify categorical and dummy variables as factors
# Following the recomendations from:
  #https://cran.r-project.org/web/packages/xgboost/vignettes/discoverYourData.html#numeric-v.s.-categorical-variables
l.cat = dt1[,mget(all.pot.Xs)] %>% sapply(., function(x) table(x) %>% length)
cat.names = l.cat[l.cat<24] %>% names
l.cat[l.cat<24]
l.cat[l.cat>24]
for (.x in cat.names) dt1[, (.x) := factor(get(.x))]

### Create one data.matrix for y-variable and other matrix for our data of x variables
yF = dt1[,get(.y.n)]  %>% data.matrix()

tmp = Matrix::sparse.model.matrix(as.formula(paste0(.y.n,"~",".")), data = dt1[,mget(c(.Xs.n,.y.n))],
                                  row.names = F)[,-1]
colnames(tmp)
XF = tmp
# With the following information we could have an idea of the size in memory of our X matrix
# During tuning with parallelization we could see how it's memory usage explodes,
# given us a vague idea of the amount RAM needed for tuning given the number of cores used.
XF %>% dim
XF %>% object.size() %>% format(units = "Mb")

### For further xgbost package usage
dfull = xgb.DMatrix(data = XF, 
                    label = yF)
### For further mlr package usage
dt.F = data.frame(XF %>% data.matrix(), "y" = yF)
fullTask <- makeRegrTask(data = dt.F, target = "y")

##### 8.1.7) Finally! Let's tune our hyperparameters ########
gc()
set.seed(1)
total_cores = parallel::detectCores()
total_cores
parallelMap::parallelStart(mode="socket", cpu=total_cores, 
                           level="mlr.tuneParams")
parallelMap::parallelLibrary(packages = c("magrittr",
                                          "data.table"))

parallelGetOptions()
t0 = Sys.time()
t0
tuned_params <- tuneParams(
  learner = xgb_learner,
  task = fullTask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control,
  measures = m.dv
)
parallelStop()
gc()
t1 =  Sys.time()
t1-t0

  #!!! Be careful, this function is going to use the full potential
  # of your CPU as no other program has ever done, it could lead to
  # overheating !!!

# Using the presented F1-Score personalized function, 100 iterations under random search for tuning,
# repeated cross-validation of 10 folds under 2 repetitions, 
# 1000 iterations within each xgboost with 10 non-increasing iterations as a rule to stop.  
# An i7-9750H* Windows-10 laptop took 7.167037 hours to compute this step.
  # * 6 Cores, 12 Threads, 2.60 GHz Base Frequency, 4.50 GHz Max Turbo Frequency
  # I did´t bios-overlocked the cpu, but during computation Windows
  # task manager stated that cpu was at 100% usage. Curiosly, speed fluctuated
  # around 2.6 to 4.0 GHz. 
  # Laptop temperature was pretty high, I didn't measured it, but pretty sure
  # was nearly maximum recomended.

### I highly recomend saving the results of this step. An RDS file will keep all the information
# of the original object:
  #saveRDS(tuned_params, "TunParms_2r10f1k10rs_100it_v1.rds")



#### 8.2) Optimal iterations for xgboost with our hyperparameters ################
  #tuned_params = readRDS("TunParms_2r10f1k10rs_100it_v1.rds")

# Now we need to know how many iterations our xgboost should do in order
# to have optimal results. We use xgb.cv function from xgboost package.

# Here xgb.cv performs a k-fold cross-validation. As you can see it's different 
# from RepCV from mlr. In simulations I´ve performed on other exercises (I wish I 
# could upload those in the future) I´ve found that in many cases, we need to at least 
# repeat the k-fold process 25 times.

# xgb.cv perform "just" one time a k-fold-CV. If your data is sufficiently heterogeneous 
# to suspect that k partitions of it can't address to all of it's heterogeneity, 
# as I believe is this case, we should migrate this chunk of code to mlr. 
# But, for now, it´s what it´s

best.tuned_params = tuned_params$x

t0 = Sys.time()
set.seed(1)
xgb_cv_res.x1 <- xgb.cv(data=XF,
                        label=yF,
                        objective = "reg:squarederror",
                        eval_metric = evalerror.dv1, 
                        params = best.tuned_params, 
                        nthread=total_cores, 
                        nfold=folds.par.tun, 
                        nrounds=nRound.w,
                        verbose = T,
                        early_stopping_rounds = EarlyStop.nR, # The number of rounds with no improvement in the evaluation metric in order to stop the training.
                        maximize=T)
t1 = Sys.time()
t1-t0
# 31.27099 mins, 75 iter
  
  # Even though xgb.cv states that it's natevely parallelized, as we can see, it doesn't
  # use 100% of our cpu. It's matter of further research if this step could be better
  # parallelized.


# Get the full log (all iterations and its results) into a data.table
xgb_cv_res.x1.dt = data.table(xgb_cv_res.x1$evaluation_log)
# As I believe the standard error of any measure is an important part (and often ignored) 
# of any optmization process using data, I´d like to propose to you it´s inclusion in our computations. 
# So, if we want to take the maximun value we "should" rest test_F1_score_std from test_F1_score_mean.

# Under this idea, the best iteration would be:
xgb_cv_res.x1.dt[, eval_metric_and_sd := test_F1_score_mean  - test_F1_score_std]
xgb_cv_res.x1.dt %<>% .[order(-eval_metric_and_sd)]
best.iter = xgb_cv_res.x1.dt[1,1][[1]]
best.iter

#### 8.3) Compute xgboost for full database for our optimal # of iterations and hyperparameters ################
# This step could seem useless. And for predictive or CV reasons it is, unless you have some more data
# were you could want to test its performance.
# But, I perform this step for three reasons:
  # 1) To compute feature importance in a further step.
  # 2) To know how much time the "final" model takes to run.
  # 3) To know its in-sample prediction properties.

t0 = Sys.time()
t0
set.seed(1)
xgb_res_dfull <- xgboost(data=XF,
                         label=yF,
                         objective = "reg:squarederror",
                         eval_metric = evalerror.dv1, 
                         #eval_metric = "rmse", 
                         params = best.tuned_params, 
                         nthread=total_cores, 
                         nrounds=best.iter,
                         verbose = T,
                         cb.early.stop(stopping_rounds = best.iter), # The number of rounds with no improvement in the evaluation metric in order to stop the training.
                         maximize=T)
t1 = Sys.time()
t1-t0
  #1.546265 mins

pred.y.xgb.1 <- predict(xgb_res_dfull, dfull)


# So: to know it's in-sample prediction properties
summary(pred.y.xgb.1)
# In sample roc: 
pROC::roc(yF,pred.y.xgb.1)
# Best in sample F1-Score
tmp = sapply(seq(0,1,0.01), function(x)
  f1.score.dv(yF %>% as.numeric(),pred.y.xgb.1,x))
tmp[tmp==max(tmp)]

### If you want, you could save the model. To, for example, use it to predict
# our y-variable on other data (with the same Xs of course):
  #xgb.save(xgb_res_dfull, 'xgb_res_dfull')

#### 8.4) With a simple CV, test again predictive power ################

# If you are as suspicious as me, you would like to perform these computations.
# What I mean is: what if we did something wrong and xgb.cv and/or our hyperparameter tuning
# didn't gave us a real idea of our out-of-sample predictive power?

# Also, with many iterations of this random sub-sampling validation, we can have an idea
# of the distribution of the predictive measure we are handling. If we have too much dispersion
# we should try again all steps with bigger reps.par.tun for tuning, perhaps more folds for
# our xgb.cv or suspect we have omitted an important variable.

# We want this step to be parallelized, for this purpose we use doSnow package.
# If you don't want to parallelize, change line 505 from 
# %dopar% to %do%. In this scenario lines 449, 500 and 565 are irrelevant
gc()
cl = makeCluster(total_cores)
registerDoSNOW(cl)

.eval_measure.res = foreach (x.seed = 1:100, .combine = c,
                      .packages = c("magrittr", "data.table"),
                      .export = c("yF","XF","best.iter")
) %dopar% {
  t0 = Sys.time()
  #print(x.seed)
  set.seed(x.seed)
  nrowF = nrow(yF)
  
  id.test = sample(1:nrowF, nrowF*0.3)

  y1.test = yF[id.test]
  X1.test = XF[id.test,]
  d.test = xgboost::xgb.DMatrix(data = X1.test, 
                                label = y1.test)
  
  y1.train = yF[-id.test]
  X1.train = XF[-id.test,]
  
  suppressMessages(
    model.x <- xgboost::xgboost(data=X1.train,
                                label=y1.train,
                                nround = best.iter, 
                                objective = "reg:squarederror",
                                eval_metric = evalerror.dv1,
                                #eval_metric = "rmse",
                                nthread =1, 
                                early_stopping_rounds = best.iter, 
                                params = best.tuned_params,
                                maximize = T,
                                verbose = 0 #If 0, xgboost will stay silent
    ) 
  )
  # Prediction on test data
  pred.y.xgb.x <- predict(model.x, d.test)
  
  # Compute roc measure or whatever you want
  suppressMessages(
    .roc.x <- pROC::roc(y1.test,pred.y.xgb.x)
  )
  .roc.x = .roc.x$auc %>% as.numeric()

  f1.s.x = sapply(seq(0,1,0.01), function(x)
    f1.score.dv(y1.test %>% as.numeric(),pred.y.xgb.x,x))
  f1.s.x = f1.s.x[f1.s.x==max(f1.s.x)]
  
  
  # Measure total time for each iteration
  t1 = Sys.time()
  diff.ts = difftime(t1,t0,units = "mins") %>% as.numeric()
  # Save results in a list
  .n1 = paste0("time_",x.seed)
  .n2 = paste0("roc_",x.seed)
  .n3 = paste0("f1_score_",x.seed)
  .res = list()
  .res[[.n1]] = diff.ts;.res[[.n2]] =.roc.x
  .res[[.n3]] = f1.s.x
  # This step is weirdly needed. Sometimes I have seen that doSnow cleans RAM after
  # each iteration. Sometimes not, so, I'll leave it here just in case:
  gc()
  # Return our results
  return(.res)
}
stopCluster(cl)
gc()

# Subset our vector
.ts = names(.eval_measure.res) %>% str_which("time")
.rs = names(.eval_measure.res) %>% str_which("roc")
.fs = names(.eval_measure.res) %>% str_which("f1_score")

# Let's see the total amount of time and distribution of time of each xgboost
# In minutes:
.eval_measure.res[.ts] %>% unlist %>% sum
.eval_measure.res[.ts] %>% unlist %>% summary()

#Let's see the distribution of our predictive measure
.eval_measure.res[.rs] %>% unlist %>% summary()
.eval_measure.res[.fs] %>% unlist %>% summary()

######## 9) Feature importance #########################
# xgb.importance from xgboost decomposes the percentage that each feature
# contributes to the selected measure of performance. 
# This information is also useful for further feature selection.

model.f.importance =  xgb.importance(model = xgb_res_dfull)
model.f.importance %<>% .[order(-Gain)]
model.f.importance[,Gain] %>% sum
model.f.importance[,Gain.cumsum := cumsum(Gain)] 
model.f.importance %>% dim
  # In this stage you can see that from 78 variables (each category of our
  # categorical variables is a dummy, so a variable itself) only 31 are considered.

######## 10) Interpretable model #########################

# An xgboost at the end of the day could be expressed as a bunch of simpler models.
# Paraphrasing David Foster, author of xgboostExplainer package:
  # We could make our xgboost as simple to understand as a single decision tree.

# Here you can find the package description from author's own words:
  # https://medium.com/applied-data-science/new-r-package-the-xgboost-explainer-51dd7d1aa211
  # https://github.com/AppliedDataSciencePartners/xgboostExplainer

# This library isn't on CRAN, you have to install it as Foster suggests:
  # library(devtools) 
  # install_github("AppliedDataSciencePartners/xgboostExplainer")

library(xgboostExplainer)

set.seed(1)

prop.test = 0.3
n.yXf = nrow(XF)
id.test = sample(n.yXf, n.yXf*prop.test)
Xf.train = XF[-id.test,]
yf.train = yF[-id.test]

Xf.test = XF[id.test,]
yf.test = yF[id.test]

xgb.train.data <- xgb.DMatrix(Xf.train, label = yf.train)
xgb.test.data <- xgb.DMatrix(Xf.test, label = yf.test )



t0 = Sys.time()
t0
explainer = buildExplainer(xgb_res_dfull,xgb.train.data, 
                           type="regression")
t1 = Sys.time()
t1-t0
  #25.0574 secs
  #With big models this step could take hours

t0 = Sys.time()
t0
pred.breakdown = explainPredictions(xgb_res_dfull, explainer, xgb.test.data)
t1 = Sys.time()
t1-t0
  #2.088038 secs

explainer %>% dim
  # 576  
  # We have as much nodes as rows has the explainer object


######## 11) Conclusions and remarks #########################
# Here you have it, a full walkthrough of xgboost, with parallelized parameter tuning
# using any personalized "loss function". There's still job to do to leave this
# script better. But I think you could use it as a power-horse for tons of predicting
# problems that arise on the day-to-day work, heavily out-performing the predicting power
# of simpler models and/or xgboost withouth proper tuning.

# I have been around modelling of variables such as child malnutrition since 2013 using 
# an econometric's approach both for prediction and causality's understanding. 
# And I could assure to you that xgboost out-performs, in the prediction world, 
# by far any econometric approach I ever tryed, even when based on tons of time on feature computation, 
# thinking, re-thinking and selection, even with simple xgboosts as is the present one.
# ML-based approaches are also getting more popular within the causality world, with big names
# such as Dufflo using them. So, it's worth it to any statistician or econometrician
# to get it's hands on ML.

# I have to state that after research about Hyperparameter-Tuning many authors suggest
# to use bayesian or genetic-optimization-based algorithms to get better and faster results.
# Those algorithms are only implemented on Python. So, if you want even better results, it's
# time to take hands-on Python.