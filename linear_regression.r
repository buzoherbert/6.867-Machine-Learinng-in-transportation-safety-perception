install.packages("caTools", dependencies = TRUE)
install.packages("magrittr")
install.packages("caret")
install.packages('e1071', dependencies=TRUE)
install.packages("data.table")

# For sample splitting
library("caTools")

# For removing unused levels in test data
library(magrittr)

# For confusion matrices
library("caret")
library("e1071")
library(data.table)

interpret_variables <- function(data_frame){
  # Remove unused
  data_frame[["X"]] = NULL
  data_frame[["latitude"]] = NULL
  data_frame[["longitude"]] = NULL
  
  
  # Making sure variables are treated properly
  for(i in names(data_frame)){
    if(i == "total_female_count"){
      data_frame[["total_female_count"]] = as.numeric(as.character(data_frame[["total_female_count"]]))
    } else if(i == "total_passenger_count"){
      data_frame[["total_passenger_count"]] = as.numeric(as.character(data_frame[["total_passenger_count"]]))
    } else if(i == "empty_seats") {
      data_frame[["empty_seats"]] = as.numeric(as.character(data_frame[["empty_seats"]]))
    } else if(i == "point_security") {
      data_frame[["point_security"]] = as.numeric(as.character(data_frame[["point_security"]]))
    } else if(i == "mode_security") {
      data_frame[["mode_security"]] = as.numeric(as.character(data_frame[["mode_security"]]))
    } else if(i == "importance_safety") {
      data_frame[["importance_safety"]] = as.numeric(as.character(data_frame[["importance_safety"]]))
    } else if(i == "haversine") {
      data_frame[["haversine"]] = as.numeric(as.character(data_frame[["haversine"]]))      
    } else if(i == "hour_sin") {
      data_frame[["hour_sin"]] = as.numeric(as.character(data_frame[["hour_sin"]]))      
    } else if(i == "hour_cos") {
      data_frame[["hour_cos"]] = as.numeric(as.character(data_frame[["hour_cos"]]))      
    }else {
      data_frame[[i]] <- as.factor(data_frame[[i]])
    } 
  }
  return(data_frame)
}

# Removing categories not on the training set
# Residual standard error: 1.092 on 550 degrees of freedom
# Multiple R-squared:  0.416, Adjusted R-squared:  0.2727 
# F-statistic: 2.902 on 135 and 550 DF,  p-value: < 2.2e-16

# Funtion to remove misssing levels from fields in a testing dataframe
# Source: https://stackoverflow.com/a/44316204/3128369
#' @title remove_missing_levels
#' @description Accounts for missing factor levels present only in test data
#' but not in train data by setting values to NA
#'
#' @import magrittr
#' @importFrom gdata unmatrix
#' @importFrom stringr str_split
#'
#' @param fit fitted model on training data
#'
#' @param test_data data to make predictions for
#'
#' @return data.frame with matching factor levels to fitted model
#'
#' @keywords internal
#'
#' @export
remove_missing_levels <- function(fit, test_data) {
  
  # https://stackoverflow.com/a/39495480/4185785
  
  # drop empty factor levels in test data
  test_data %>%
    droplevels() %>%
    as.data.frame() -> test_data
  
  # 'fit' object structure of 'lm' and 'glmmPQL' is different so we need to
  # account for it
  if (any(class(fit) == "glmmPQL")) {
    # Obtain factor predictors in the model and their levels
    factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                     names(unlist(fit$contrasts))))
    # do nothing if no factors are present
    if (length(factors) == 0) {
      return(test_data)
    }
    
    map(fit$contrasts, function(x) names(unmatrix(x))) %>%
      unlist() -> factor_levels
    factor_levels %>% str_split(":", simplify = TRUE) %>%
      extract(, 1) -> factor_levels
    
    model_factors <- as.data.frame(cbind(factors, factor_levels))
  } else {
    # Obtain factor predictors in the model and their levels
    factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                     names(unlist(fit$xlevels))))
    # do nothing if no factors are present
    if (length(factors) == 0) {
      return(test_data)
    }
    
    factor_levels <- unname(unlist(fit$xlevels))
    model_factors <- as.data.frame(cbind(factors, factor_levels))
  }
  
  # Select column names in test data that are factor predictors in
  # trained model
  
  predictors <- names(test_data[names(test_data) %in% factors])
  
  # For each factor predictor in your data, if the level is not in the model,
  # set the value to NA
  
  for (i in 1:length(predictors)) {
    found <- test_data[, predictors[i]] %in% model_factors[
      model_factors$factors == predictors[i], ]$factor_levels
    if (any(!found)) {
      # track which variable
      var <- predictors[i]
      # set to NA
      test_data[!found, predictors[i]] <- NA
      # drop empty factor levels in test data
      test_data %>%
        droplevels() -> test_data
      # issue warning to console
      message(sprintf(paste0("Setting missing levels in '%s', only present",
                             " in test data but missing in train data,",
                             " to 'NA'."),
                      var))
    }
  }
  return(test_data)
}


# Calculating f1 score
# Source: https://stackoverflow.com/a/36843900
f1_score <- function(predicted, expected, positive.class="1") {
  predicted <- as.factor(predicted)
  expected  <- as.factor(expected)
  cm = as.matrix(table(expected, predicted))
  
  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <-  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  
  #Assuming that F1 is zero when it's not possible compute it
  f1[is.na(f1)] <- 0
  
  #Binary F1 or Multi-class macro-averaged F1
  ifelse(nlevels(expected) == 2, f1[positive.class], mean(f1))
}


# Function to get the metrics of each model and add it to the summaries table
# It returns the summaries table passed to it with added data about the new model
# If summaries table doesn't exist, it creates it
getModelMetrics <- function(file_num, model_name, linear_model, summaries_table, train_data, test_data, 
                            significant_variables_only, confusion_matrix) {
  # If the summaries table is not a data frame, it gets initialized
  if(!is.data.frame(summaries_table)){
    summaries_table <- data.frame(matrix(ncol = 13, nrow = 0))
    names <- c("category", "file_num", "r2", "sse", "pred_means", "sst", "osr2", "rmse", "mae", 
               "var_num", "significant", "accuracy", "f1_score")
    colnames(summaries_table) <- names
  }
  
  pred_data = getPredictionVector(linear_model, test_data)
  SSE = sum((pred_data - test_data$point_security)^2)
  pred_mean = mean(train_data$point_security)
  SST = sum((pred_mean - test_data$point_security)^2)
  OSR2 = 1-SSE/SST
  RMSE = sqrt(sum((pred_data - test_data$point_security)^2)/nrow(test_data))
  MAE = sum(abs(pred_data - test_data$point_security))/nrow(test_data)
  
  i = nrow(summaries_table) + 1
  summaries_table[i, "category"] = model_name
  summaries_table[i, "file_num"] = file_num
  summaries_table[i, "r2"] = summary(linear_model)$r.squared
  summaries_table[i, "sse"] = SSE
  summaries_table[i, "pred_means"] = pred_mean
  summaries_table[i, "sst"] = SST
  summaries_table[i, "osr2"] = OSR2
  summaries_table[i, "rmse"] = RMSE
  summaries_table[i, "mae"] = MAE
  summaries_table[i,"var_num"] = length(names(linear_model$coefficients))
  if(significant_variables_only){
    summaries_table[i, "significant"] = TRUE;
  } else {
    summaries_table[i, "significant"] = FALSE;
  }
  summaries_table[i, "accuracy"] = confusion_matrix$overall['Accuracy']
  #Making test and predicted vector same size
  expected = as.factor(na.omit(
    remove_missing_levels(fit=linear_model, test_data=test_data))$point_security)
  summaries_table[i, "f1_score"] = f1_score(pred_data, expected)
  return(summaries_table)
}


cleanSignificantVariables <- function(original_list, list_to_clean) {
  # Removing intercept from list
  list_to_clean = list_to_clean[list_to_clean != "(Intercept)"]
  for(i in 1:(length(list_to_clean))){
    for(j in 1:(length(original_list))){
      if(grepl(original_list[[j]], list_to_clean[[i]])){
        list_to_clean[[i]] = original_list[[j]]
      }
    }
  }
  return(list_to_clean) 
}


getPredictionVector <- function(model, test_data){
  test_data = na.omit(remove_missing_levels (fit=model, test_data=test_data))
  pred_data = predict(model, newdata = test_data)
  return(pred_data)
}

getFactorVectorFromFloat <- function(data){
  for(i in 1:(length(data))){
    if(data[[i]] > 5){
      data[[i]] <- 5
    }
    if(data[[i]] < 1){
      data[[i]] <- 1
    }
    data[[i]] = round(data[[i]])
  }
  return(lapply(data, as.integer))
}

getConfusionMatrix <- function(pred, test, model){
  predicted = as.factor(unlist(pred, use.names=FALSE))
  real = as.factor(na.omit(
    remove_missing_levels(fit=model, test_data=test))$point_security)
  
  real <- ordered(real, levels = c("1","2","3","4","5"))
  predicted <- ordered(predicted, levels = c("1","2","3","4","5"))
  
  u = union(predicted, real)
  conf = table(ordered(predicted, c("1","2","3","4","5")), ordered(real, c("1","2","3","4","5")))
  confusion_matrix = confusionMatrix(conf)
  return(confusion_matrix)
}

aggregateConfusionMatrices <- function(conf_mat_list){
  #Assuming the list is not empty, that all the matrices have the same dimensions and that they are all square matrices
  dim = nrow(as.table(conf_mat_list[[1]]))
  sum_matrix = as.matrix(conf_mat_list[[1]])
  for(i in 2:(length(conf_mat_list))){
    sum_matrix = sum_matrix + as.matrix(conf_mat_list[[i]])
  }
  total = sum(sum_matrix)
  weighted_matrix = (1/total) * sum_matrix
  return(weighted_matrix)
}


outputConfusionMatrices <- function(conf_mat_list, filename){
  append = FALSE
  for(i in 1:(length(conf_mat_list))){
    write.table(conf_mat_list[[i]], file = filename, append = append, quote = TRUE, sep = ",",
                eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                col.names = FALSE, qmethod = c("escape", "double"),
                fileEncoding = "")
    append = TRUE
  }
}



##################
#Initial varibles

data_tables = list()
data_test = list()
data_train = list()
models = list()
models_significant = list()
summaries = NULL
summaries_sig = NULL

#Prediction vectors
predictions = list()
predictions_sig = list()
cat_pred = list()
cat_pred_sig = list()

conf_mat = list()
conf_mat_sig = list()

conf_mat_agg = list()
conf_mat_agg_sig = list()

# Variable lists

all_variables = c("age", "gender", "education","origin",
                  "destination", "companions", "trip_purpose",
                  "mode_security", "importance_safety", "most_safe", "least_safe",
                  "bus_or_ped", "base_study_zone", "busdestination", "total_seats",
                  "haversine", "urban_typology", "total_passenger_count", "total_female_count",
                  "empty_seats", "hour_sin", "hour_cos", "week_day")
sociodemographics = c("age", "gender", "education")
personal_trip = c("origin", "destination", "companions", "trip_purpose")
perception = c("mode_security", "importance_safety", "most_safe", "least_safe")
trip_context = c("bus_or_ped", "base_study_zone", "busdestination", "total_seats")
time_context = c("haversine", "urban_typology", "total_passenger_count", "total_female_count",
                 "empty_seats", "hour_sin", "hour_cos", "week_day")

model_variables = list(all_variables, sociodemographics, personal_trip, perception, trip_context, time_context)

model_names = c("All variables", "Sociodemographic","Personal trip", "Perception", 
                "Trip context", "Time context")


# loading files
files_number = 5
for (i in 1:(files_number)) {
  file_name = "final_data_"
  file_name = paste(file_name, i-1, sep="")
  file_name = paste(file_name, ".csv", sep="")
  data_tables[[i]] <- read.table(file=file_name, header = TRUE, na.strings=c("", "NA"), sep=",")
  data_tables[[i]] <- interpret_variables(data_tables[[i]])
  
  # Dividing the datasets
  train_smp_size <- floor(0.6 * nrow(data_tables[[i]]))
  test_smp_size <- floor(0.2 * nrow(data_tables[[i]]))
  data_train[[i]] <- data_tables[[i]][1:(train_smp_size-1),]
  data_test[[i]] <- data_tables[[i]][(train_smp_size + (test_smp_size + 1)):nrow(data_tables[[i]]),]
  #data_test[[i]] <- data_tables[[i]][(train_smp_size ):nrow(data_tables[[i]]),]
}


# Generating the models
for(i in 1:(length(model_names))){
  for (j in 1:(files_number)) {
    ########################
    ## Linear regression model with all the variables
    variables_to_add = paste("+", paste(model_variables[[i]], collapse =" +"), sep="")
    model_def = paste("data_train[[j]]$point_security ~",variables_to_add, sep="")
    if(j==1){ models[[i]] = list()}
    models[[i]][[j]] = lm(model_def, data = data_train[[j]])
   
    ########################
    ## Linear regression model with significant variables
    # Getting significant variables
    sig_variables = row.names(data.frame(summary(models[[i]][[j]])$coef[summary(models[[i]][[j]])$coef[,4] <= .1, 4]))
    
    # Cleaning the significant variable list
    sig_variables = cleanSignificantVariables(model_variables[[i]], sig_variables)
    
    variables_to_add = paste("+", paste(sig_variables, collapse =" +"), sep="")
    model_def = paste("data_train[[j]]$point_security ~",variables_to_add, sep="")
    if(j==1){ models_significant[[i]] = list()}
    models_significant[[i]][[j]] = lm(model_def, data = data_train[[j]])
    
    
    ############################
    #Getting prediction vectors
    
    # For normal models
    if(j==1){
      predictions[[i]] = list()
      cat_pred[[i]] = list()
      predictions_sig[[i]] = list()
      cat_pred_sig[[i]] = list()
      conf_mat[[i]] = list()
      conf_mat_sig[[i]] = list()
    }
    predictions[[i]][[j]] = getPredictionVector(models[[i]][[j]], data_test[[j]])
    cat_pred[[i]][[j]] = getFactorVectorFromFloat(predictions[[i]][[j]])
    
    predictions_sig[[i]][[j]] = getPredictionVector(models_significant[[i]][[j]], data_test[[j]])
    cat_pred_sig[[i]][[j]] = getFactorVectorFromFloat(predictions_sig[[i]][[j]])
    
    conf_mat[[i]][[j]] = getConfusionMatrix(cat_pred[[i]][[j]], data_test[[j]], models[[i]][[j]])
    conf_mat_sig[[i]][[j]] = getConfusionMatrix(cat_pred_sig[[i]][[j]], data_test[[j]], models_significant[[i]][[j]])
    # Adding to summaries
    summaries = getModelMetrics(j, model_names[i] ,models[[i]][[j]], summaries, data_train[[j]], data_test[[j]], 
                                FALSE, conf_mat[[i]][[j]])
    summaries_sig = getModelMetrics(j, model_names[i] ,models_significant[[i]][[j]], summaries, data_train[[j]], data_test[[j]], 
                                TRUE, conf_mat_sig[[i]][[j]])
  }
  conf_mat_agg[[i]] = aggregateConfusionMatrices(conf_mat[[i]])
  conf_mat_agg_sig[[i]] = aggregateConfusionMatrices(conf_mat_sig[[i]])
}

# Aggregating data
summaries_aggregated = aggregate(summaries, list(summaries$category), mean)
summaries_aggregated$category = summaries_aggregated$Group.1
summaries_aggregated$Group.1 = NULL
summaries_sig_aggregated = aggregate(summaries_sig, list(summaries_sig$category), mean)
summaries_sig_aggregated$category = summaries_sig_aggregated$Group.1
summaries_sig_aggregated$Group.1 = NULL

#Writing output to files
conf_folder_name = "./confusions/"
outputConfusionMatrices(conf_mat_agg, paste(conf_folder_name, "linear_regression.csv", sep=""))
outputConfusionMatrices(conf_mat_agg_sig, paste(folder_name, "linear_regression_stepwise.csv", sep=""))

metrics_folder_name = "./metrics/"

write.table(summaries_aggregated, file = paste(metrics_folder_name, "linear_regression.csv", sep=""), 
            append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")

write.table(summaries_sig_aggregated, file = paste(metrics_folder_name, "linear_regression_stepwise.csv", sep=""), 
            append = FALSE, quote = TRUE, sep = ",",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")



# Some plots
library(ggplot2)
qplot(var_num, osr2, colour = significant, shape = model, 
      +       data = summaries)

qplot(var_num, r2, colour = significant, shape = model, 
      data = summaries)


qplot(var_num, f1_score, data = summaries_aggregated, shape = Group.1)
qplot(var_num, accuracy, data = summaries_aggregated, shape = Group.1)
qplot(var_num, sse, data = summaries_aggregated, shape = Group.1)
