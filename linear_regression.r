install.packages("caTools", dependencies = TRUE)
install.packages("magrittr")
install.packages("class")

# For sample splitting
library("caTools")

# For removing unused levels in test data
library(magrittr)

# For knn model
library("class")

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


# Function to get the metrics of each model and add it to the summaries table
# It returns the summaries table passed to it with added data about the new model
# If summaries table doesn't exist, it creates it
getModelMetrics <- function(model_name, linear_model, summaries_table, train_data, test_data) {
  test_data = na.omit(remove_missing_levels (fit=linear_model, test_data=test_data))
  
  # If the summaries table is not a data frame, it gets initialized
  if(!is.data.frame(summaries_table)){
    summaries_table <- data.frame(matrix(ncol = 9, nrow = 0))
    names <- c("model", "r2", "sse", "pred_means", "sst", "osr2", "rmse", "mae", "var_num")
    colnames(summaries_table) <- names
  }
  
  pred_data = predict(linear_model, newdata = test_data)
  SSE = sum((pred_data - test_data$point_security)^2)
  pred_mean = mean(train_data$point_security)
  SST = sum((pred_mean - test_data$point_security)^2)
  OSR2 = 1-SSE/SST
  RMSE = sqrt(sum((pred_data - test_data$point_security)^2)/nrow(test_data))
  MAE = sum(abs(pred_data - test_data$point_security))/nrow(test_data)
  
  i = nrow(summaries_table) + 1
  summaries_table[i, "model"] = model_name
  summaries_table[i, "r2"] = summary(linear_model)$r.squared
  summaries_table[i, "sse"] = SSE
  summaries_table[i, "pred_means"] = pred_mean
  summaries_table[i, "sst"] = SST
  summaries_table[i, "osr2"] = OSR2
  summaries_table[i, "rmse"] = RMSE
  summaries_table[i, "mae"] = MAE
  summaries_table[i,"var_num"] = length(names(linear_model$coefficients))
  
  return(summaries_table)
}



# loading files
files_number = 5

data_tables = list()
data_test = list()
data_train = list()
models = list()
summaries = NULL
for (i in 1:(files_number)) {
  file_name = "final_data_"
  file_name = paste(file_name, i-1, sep="")
  file_name = paste(file_name, ".csv", sep="")
  data_tables[[i]] <- read.table(file=file_name, header = TRUE, na.strings=c("", "NA"), sep=",")
  data_tables[[i]] <- interpret_variables(data_tables[[i]])
  
  # Dividing the datasets
  train_smp_size <- floor(0.6 * nrow(data_tables[[i]]))
  data_train[[i]] <- data_tables[[i]][1:(train_smp_size-1),]
  data_test[[i]] <- data_tables[[i]][(train_smp_size):nrow(data_tables[[i]]),]
  
  
  ########################
  ## Linear regression model with all the variables
  models[[i]] = lm(data_train[[i]]$point_security ~., data = data_train[[i]])
  summary(models[[i]])
  model_name = paste("Initial model ", i, sep="")
  summaries = getModelMetrics(model_name ,models[[i]], summaries, data_train[[i]], data_test[[i]])
  
  sig_variables = row.names(data.frame(summary(models[[i]])$coef[summary(models[[i]])$coef[,4] <= .1, 4]))
  
  
  model_info = paste("data_train[[i]]$point_security ~",sig_variables, sep="")
  models[[i]] = lm(model_info, data = data_train[[i]])
  model_name = paste("Only relevant variables ", i, sep="")
  summaries = getModelMetrics(model_name ,models[[i]], summaries, data_train[[i]], data_test[[i]])
  #sig_variables_list = as.vector(sig_variables$Symbol)
  
  
}







########################
## Linear regression model


linear_model = lm(train$point_security~., data = train)
summary(linear_model)
summaries = getModelMetrics("Initial model",linear_model, NULL, train, test)


# Taking out the least relevant variables
linear_model2 = lm(train$point_security~. 
                   -inside_or_outside -gender -age -education
                   -trip_purpose -most_safe -least_safe -bus_or_ped
                   -base_study_zone -busdestination -total_seats -haversine
                   -total_passenger_count -total_female_count -empty_seats
                   , data = train)
summary(linear_model2)
summaries = getModelMetrics("Relevant variables",linear_model2, summaries, train, test)

# Only trip variables

linear_model3 = lm(train$point_security~ +origin +destination +companions +trip_purpose, data = train)
summary(linear_model3)
summaries = getModelMetrics("Trip variables",linear_model3, summaries, train, test)


# Only significant variables for the trip

linear_model3a = lm(train$point_security~ +origin +destination, data = train)
summary(linear_model3a)
summaries = getModelMetrics("Trip variables - Relevant",linear_model3a, summaries, train, test)

# Only perception variables

linear_model4 = lm(train$point_security~ +mode_security +importance_safety +most_safe +least_safe, data = train)
summary(linear_model4)
summaries = getModelMetrics("Perception variables",linear_model4, summaries, train, test)

# Only perception variables

linear_model4a = lm(train$point_security~ +mode_security +importance_safety, data = train)
summary(linear_model4a)
summaries = getModelMetrics("Perception variables - relevant",linear_model4a, summaries, train, test)


#Only  instant contextual information
linear_model5 = lm(train$point_security~ +haversine +urban_typology +total_passenger_count
                   +total_female_count +empty_seats +hour +week_day, data = train)
summary(linear_model5)
summaries = getModelMetrics("Instant contextual information",linear_model5, summaries, train, test)

#Only  instant contextual information
linear_model5a = lm(train$point_security~ +urban_typology, data = train)
summary(linear_model5a)
summaries = getModelMetrics("Instant contextual information - relevant",linear_model5a, summaries, train, test)

# Only sociodemographic data
linear_model6 = lm(train$point_security~ +age +gender +education, data = train)
summary(linear_model6)
summaries = getModelMetrics("Sociodemographic data",linear_model6, summaries, train, test)


# Only personal trip information
linear_model7 = lm(train$point_security~ +origin +destination +companions +trip_purpose, data = train)
summary(linear_model7)
summaries = getModelMetrics("Personal trip information",linear_model7, summaries, train, test)


linear_model7a = lm(train$point_security~ +origin +destination, data = train)
summary(linear_model7a)
summaries = getModelMetrics("Personal trip information - relevant",linear_model7a, summaries, train, test)