
install.packages("GGally", dependencies = TRUE)
install.packages("ggplot2", dependencies = TRUE)
install.packages("caTools", dependencies = TRUE)
install.packages("magrittr")
install.packages("class")


# Graphing library
library(GGally)
library(ggplot2)


# For sample splitting
library("caTools")

# For removing unused levels in test data
library(magrittr)

# For knn model
library("class");

# loading file
plot_data <- read.table(file="safety_data_clean.csv", header = TRUE, na.strings=c("", "NA"), sep=",")

# Making sure variables are treated properly
for(i in names(plot_data)){
  if(i == "total_female_count"){
    print(i)
    plot_data[["total_female_count"]] = as.numeric(as.character(plot_data[["total_female_count"]]))
  } else if(i == "total_passenger_count"){
    print(i)
    plot_data[["total_passenger_count"]] = as.numeric(as.character(plot_data[["total_passenger_count"]]))
  } else if(i == "empty_seats") {
    print(i)
    plot_data[["empty_seats"]] = as.numeric(as.character(plot_data[["empty_seats"]]))
  } else if(i == "point_security") {
    print(i)
    plot_data[["point_security"]] = as.numeric(as.character(plot_data[["point_security"]]))
  } else if(i == "haversine") {
    print(i)
    plot_data[["haversine"]] = as.numeric(plot_data[["haversine"]])      
  } else {
    print("default")
    plot_data[[i]] <- as.factor(plot_data[[i]])
  } 
}

# Some fields have to be treated as numeric.
# TODO improve this so we don't convert the data twice.
plot_data[["total_female_count"]] = as.numeric(as.character(plot_data[["total_female_count"]]))
plot_data[["total_passenger_count"]] = as.numeric(as.character(plot_data[["total_passenger_count"]]))
plot_data[["empty_seats"]] = as.numeric(as.character(plot_data[["empty_seats"]]))
plot_data[["point_security"]] = as.numeric(as.character(plot_data[["point_security"]]))
plot_data[["haversine"]] = as.numeric(plot_data[["haversine"]])


# Getting a summary of the data
# summary(plot_data)

# plotting the data
#to_plot = plot_data
#to_plot$point_security = as.factor(to_plot$point_security)
#ggpairs(to_plot, mapping = aes(color = point_security))

########################
## Creating train and testing sets
## 70% of the sample size
smp_size <- floor(0.7 * nrow(plot_data))
## set the seed to make your partition reproductible
set.seed(888)
train_ind <- sample(seq_len(nrow(plot_data)), size = smp_size)

train <- plot_data[train_ind, ]
test <- plot_data[-train_ind, ]

train = subset(train, select = -c(X))
test = subset(test, select = -c(X))



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


########################
## Linear regression model


linear_model = lm(train$point_security~., data = train)
summary(linear_model)
summaries = getModelMetrics("Initial model",linear_model, NULL, train, test)

# Call:
# lm(formula = train$point_security ~ ., data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.7961 -0.7175  0.0039  0.6678  3.2401 

# Coefficients: (5 not defined because of singularities)
#                                    Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                       1.341e+00  1.323e+00   1.014  0.31116    
# inside_or_outsideFuera CETRAM    -1.962e-01  1.335e-01  -1.470  0.14213    
# gendermale                        3.610e-02  1.054e-01   0.342  0.73217    
# age18-24                         -9.187e-02  2.092e-01  -0.439  0.66078    
# age25-44                          1.432e-01  2.056e-01   0.696  0.48648    
# age45-64                         -8.963e-02  2.267e-01  -0.395  0.69273    
# age65+                            1.166e-01  3.193e-01   0.365  0.71522    
# educationMaestria y Doctorado     9.371e-02  6.532e-01   0.143  0.88597    
# educationPreparatoria             1.301e-01  1.271e-01   1.024  0.30619    
# educationPrimaria                -2.182e-01  2.258e-01  -0.966  0.33439    
# educationSecundaria               9.016e-02  1.495e-01   0.603  0.54683    
# educationSin estudios            -5.722e-01  4.548e-01  -1.258  0.20895    
# originAtenco                      7.648e-01  1.280e+00   0.597  0.55050    
# originAtizapan de Zaragoza       -1.503e-01  3.926e-01  -0.383  0.70191    
# originAzcapotzalco                1.963e-01  3.952e-01   0.497  0.61956    
# originBenito Juarez               4.174e-01  5.897e-01   0.708  0.47931    
# originChalco                      1.620e+00  8.183e-01   1.980  0.04822 *  
# originChimalhuacï¿½ï¿½n     5.263e-01  1.444e+00   0.364  0.71571    
# originChimalhuacan                1.021e-01  1.367e+00   0.075  0.94048    
# originCoacalco de Berriozal       2.459e-01  5.805e-01   0.424  0.67204    
# originCoyoacan                    2.888e-01  7.237e-01   0.399  0.69003    
# originCuajimalpa de Morelos       3.978e-01  9.338e-01   0.426  0.67031    
# originCuauhtemoc                  6.084e-01  5.353e-01   1.137  0.25623    
# originCuauhtlmoc                  1.222e+00  1.304e+00   0.937  0.34910    
# originCuautitlan                  3.468e-01  1.244e+00   0.279  0.78052    
# originCuautitlan Izcalli          7.074e-01  9.357e-01   0.756  0.45002    
# originEcatepec de Morelos        -4.679e-02  4.507e-01  -0.104  0.91735    
# originGustavo A. Madero           2.035e-01  5.354e-01   0.380  0.70404    
# originIztacalco                  -5.823e-01  9.327e-01  -0.624  0.53265    
# originIztapalapa                  1.312e+00  7.804e-01   1.681  0.09337 .  
# originMagdalena Contreras         1.944e+00  1.265e+00   1.537  0.12502    
# originMiguel Hidalgo              4.254e-02  5.431e-01   0.078  0.93760    
# originNaucalpan de Juarez         3.918e-01  6.198e-01   0.632  0.52758    
# originNextlalplan                -9.873e-03  1.269e+00  -0.008  0.99379    
# originNezahualcoyotl             -2.351e-01  8.555e-01  -0.275  0.78359    
# originNicolas Romero              3.756e-01  4.830e-01   0.778  0.43710    
# originOtro                        7.068e-01  5.612e-01   1.259  0.20846    
# originTecamec                     4.111e-01  4.652e-01   0.884  0.37725    
# originTexcoco                    -5.644e-03  9.187e-01  -0.006  0.99510    
# originTlalnepantla de Baz         1.245e-01  4.131e-01   0.301  0.76322    
# originTlalpan                     8.042e-01  9.381e-01   0.857  0.39170    
# originTultitlan                   4.729e-01  7.289e-01   0.649  0.51680    
# originVenustiano Carranza         6.058e-02  7.916e-01   0.077  0.93903    
# originZumpango                    1.877e-01  1.243e+00   0.151  0.88000    
# destinationAtenco                -8.118e-01  2.007e+00  -0.405  0.68598    
# destinationAtizapan de Zaragoza   7.560e-01  8.687e-01   0.870  0.38460    
# destinationAzcapotzalco           1.067e+00  8.724e-01   1.223  0.22189    
# destinationBenito Juarez         -3.889e-02  1.047e+00  -0.037  0.97038    
# destinationChimalhuacan          -7.695e-01  1.638e+00  -0.470  0.63875    
# destinationCoacalco de Berriozal  2.590e-01  9.787e-01   0.265  0.79142    
# destinationCocotitlan            -6.461e-01  1.478e+00  -0.437  0.66222    
# destinationCoyoacan               1.030e+00  1.505e+00   0.685  0.49391    
# destinationCuajimalpa de Morelos  2.267e+00  1.082e+00   2.094  0.03670 *  
# destinationCuauhtemoc             4.795e-01  9.668e-01   0.496  0.62011    
# destinationCuautitlan            -5.841e-01  1.536e+00  -0.380  0.70381    
# destinationCuautitlan Izcalli    -2.888e-01  1.133e+00  -0.255  0.79888    
# destinationEcatepec de Morelos    1.963e-01  9.259e-01   0.212  0.83217    
# destinationGustavo A. Madero     -3.368e-01  1.038e+00  -0.324  0.74575    
# destinationIztapalapa             1.265e+00  1.444e+00   0.876  0.38125    
# destinationMiguel Hidalgo         7.637e-01  1.057e+00   0.723  0.47017    
# destinationNaucalpan de Juarez    3.374e-01  9.392e-01   0.359  0.71959    
# destinationNezahualcoyotl        -5.972e-02  1.145e+00  -0.052  0.95842    
# destinationNicolas Romero         1.041e-01  9.421e-01   0.110  0.91208    
# destinationOtro                   4.141e-01  9.682e-01   0.428  0.66905    
# destinationTecamec                5.418e-01  9.303e-01   0.582  0.56054    
# destinationTemamatla             -1.654e+00  1.419e+00  -1.165  0.24454    
# destinationTizayuca               2.010e+00  1.516e+00   1.326  0.18548    
# destinationTlalnepantla de Baz    8.398e-01  8.780e-01   0.956  0.33930    
# destinationTlalpan                2.441e+00  1.445e+00   1.689  0.09184 .  
# destinationTultitlan              1.799e+00  1.494e+00   1.204  0.22918    
# destinationVenustiano Carranza    7.714e-01  1.100e+00   0.701  0.48338    
# companions3 to 4                 -1.446e-01  2.286e-01  -0.633  0.52719    
# companionsMas                    -1.201e+00  5.374e-01  -2.234  0.02590 *  
# companionsNone                   -1.774e-01  1.187e-01  -1.495  0.13563    
# trip_purposeEstudio               2.672e-01  2.471e-01   1.081  0.28009    
# trip_purposeOtro                  4.481e-03  1.778e-01   0.025  0.97990    
# trip_purposeRecreacion            9.278e-02  2.017e-01   0.460  0.64575    
# trip_purposeTrabajo              -1.036e-01  1.656e-01  -0.626  0.53191    
# mode_security2                    5.583e-01  1.782e-01   3.132  0.00183 ** 
# mode_security3                    9.741e-01  1.379e-01   7.066 5.15e-12 ***
# mode_security4                    1.239e+00  1.709e-01   7.248 1.54e-12 ***
# mode_security5                    1.684e+00  2.119e-01   7.946 1.20e-14 ***
# importance_safety2                6.966e-01  6.434e-01   1.083  0.27946    
# importance_safety3                5.970e-01  4.736e-01   1.260  0.20810    
# importance_safety4                5.476e-01  4.518e-01   1.212  0.22601    
# importance_safety5                4.510e-01  4.181e-01   1.079  0.28122    
# importance_safetyI                2.395e+00  1.262e+00   1.898  0.05822 .  
# most_safeMetro                   -2.112e-02  1.395e-01  -0.151  0.87970    
# most_safePeseros                  2.897e-02  2.513e-01   0.115  0.90827    
# most_safeTaxi                     1.400e-02  1.694e-01   0.083  0.93415    
# most_safeTrolebus                 2.210e-01  2.474e-01   0.893  0.37216    
# least_safeMetro                   3.150e-01  3.137e-01   1.004  0.31576    
# least_safePeseros                 9.721e-02  2.535e-01   0.384  0.70146    
# least_safeTaxi                    4.669e-02  2.778e-01   0.168  0.86663    
# least_safeTrolebus               -5.329e-02  3.855e-01  -0.138  0.89011    
# bus_or_pedVan                    -8.975e-02  2.122e-01  -0.423  0.67256    
# base_study_zoneEl Rosario         7.317e-02  3.530e-01   0.207  0.83588    
# busdestinationHeroes Tecamec      1.843e-01  2.037e-01   0.905  0.36604    
# busdestinationMexico Nueva        3.089e-02  1.808e-01   0.171  0.86439    
# busdestinationMexipuerto                 NA         NA      NA       NA    
# busdestinationTacuba             -6.080e-01  4.961e-01  -1.226  0.22090    
# busdestinationTexcoco                    NA         NA      NA       NA    
# total_seats35                            NA         NA      NA       NA    
# haversine                         7.858e-05  3.576e-04   0.220  0.82616    
# urban_typology1                   4.126e-01  3.842e-01   1.074  0.28329    
# urban_typology2                   4.621e-01  2.411e-01   1.917  0.05577 .  
# urban_typology3                   8.002e-01  4.221e-01   1.896  0.05854 .  
# urban_typology4                   6.190e-01  4.960e-01   1.248  0.21259    
# urban_typology5                   1.190e-01  2.459e-01   0.484  0.62879    
# urban_typology6                   4.460e-01  2.541e-01   1.755  0.07982 .  
# urban_typology7                   3.088e-01  3.683e-01   0.839  0.40211    
# urban_typology9                   4.731e-01  2.869e-01   1.649  0.09983 .  
# total_passenger_count             8.119e-04  6.546e-03   0.124  0.90135    
# total_female_count               -1.844e-02  2.212e-02  -0.834  0.40482    
# empty_seats                      -1.343e-03  6.216e-03  -0.216  0.82899    
# hour7                                    NA         NA      NA       NA    
# hour12                           -1.031e+00  4.015e-01  -2.568  0.01051 *  
# hour13                           -1.151e+00  3.520e-01  -3.271  0.00114 ** 
# hour14                           -9.602e-01  3.419e-01  -2.808  0.00517 ** 
# hour15                           -1.058e+00  4.048e-01  -2.614  0.00921 ** 
# hour16                           -8.907e-01  3.722e-01  -2.393  0.01707 *  
# hour17                           -1.094e+00  3.456e-01  -3.166  0.00164 ** 
# hour18                           -9.899e-01  3.486e-01  -2.839  0.00470 ** 
# hour19                                   NA         NA      NA       NA    
# week_day2                         3.426e-01  2.047e-01   1.674  0.09480 .  
# week_day3                         4.161e-01  2.185e-01   1.905  0.05737 .  
# week_day4                         2.077e-01  2.341e-01   0.887  0.37533    
# week_day5                        -3.688e-02  2.123e-01  -0.174  0.86215    
# week_day6                         3.993e-01  2.066e-01   1.933  0.05375 .  
# week_day7                        -4.198e-01  2.175e-01  -1.931  0.05409 .  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.132 on 520 degrees of freedom
# Multiple R-squared:  0.4058,  Adjusted R-squared:  0.2641 
# F-statistic: 2.864 on 124 and 520 DF,  p-value: < 2.2e-16

# Taking out the least relevant variables
linear_model2 = lm(train$point_security~. 
                   -inside_or_outside -gender -age -education
                   -trip_purpose -most_safe -least_safe -bus_or_ped
                   -base_study_zone -busdestination -total_seats -haversine
                   -total_passenger_count -total_female_count -empty_seats
                   , data = train)
summary(linear_model2)
summaries = getModelMetrics("Relevant variables",linear_model2, summaries, train, test)


# Call:
# lm(formula = train$point_security ~ . - inside_or_outside - gender - 
#     age - education - trip_purpose - most_safe - least_safe - 
#     bus_or_ped - base_study_zone - busdestination - total_seats - 
#     haversine - total_passenger_count - total_female_count - 
#     empty_seats, data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.7498 -0.7347  0.0267  0.7169  3.5152 

# Coefficients: (2 not defined because of singularities)
#                                   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                       1.643865   1.088766   1.510 0.131656    
# originAtenco                      0.979259   1.206717   0.812 0.417425    
# originAtizapan de Zaragoza       -0.111396   0.371074  -0.300 0.764139    
# originAzcapotzalco                0.250690   0.379084   0.661 0.508692    
# originBenito Juarez               0.492045   0.566117   0.869 0.385138    
# originChalco                      1.274909   0.791412   1.611 0.107767    
# originChimalhuacï¿½ï¿½n     0.694904   1.385436   0.502 0.616165    
# originChimalhuacan                0.013475   1.254001   0.011 0.991430    
# originCoacalco de Berriozal       0.091676   0.546502   0.168 0.866841    
# originCoyoacan                    0.326179   0.699528   0.466 0.641196    
# originCuajimalpa de Morelos       0.422156   0.898985   0.470 0.638833    
# originCuauhtemoc                  0.470304   0.509906   0.922 0.356757    
# originCuauhtlmoc                  1.080275   1.255609   0.860 0.389965    
# originCuautitlan                  0.389037   1.214002   0.320 0.748742    
# originCuautitlan Izcalli          0.829418   0.913444   0.908 0.364268    
# originEcatepec de Morelos        -0.143333   0.428366  -0.335 0.738051    
# originGustavo A. Madero           0.210962   0.517140   0.408 0.683477    
# originIztacalco                  -0.478164   0.891406  -0.536 0.591887    
# originIztapalapa                  1.464897   0.753664   1.944 0.052439 .  
# originMagdalena Contreras         1.647572   1.240645   1.328 0.184728    
# originMiguel Hidalgo              0.048617   0.526183   0.092 0.926418    
# originNaucalpan de Juarez         0.575851   0.586730   0.981 0.326797    
# originNextlalplan                -0.563674   1.227455  -0.459 0.646256    
# originNezahualcoyotl             -0.463355   0.821278  -0.564 0.572855    
# originNicolas Romero              0.293640   0.455994   0.644 0.519872    
# originOtro                        0.917971   0.537156   1.709 0.088023 .  
# originTecamec                     0.364551   0.432470   0.843 0.399621    
# originTexcoco                     0.049386   0.888758   0.056 0.955706    
# originTlalnepantla de Baz         0.163419   0.394497   0.414 0.678854    
# originTlalpan                     0.874425   0.887833   0.985 0.325106    
# originTultitlan                   0.587433   0.696419   0.844 0.399311    
# originVenustiano Carranza        -0.004329   0.763279  -0.006 0.995477    
# originZumpango                    0.489664   1.215766   0.403 0.687280    
# destinationAtenco                -1.239409   1.888710  -0.656 0.511956    
# destinationAtizapan de Zaragoza   0.557121   0.841097   0.662 0.508008    
# destinationAzcapotzalco           0.834720   0.841306   0.992 0.321549    
# destinationBenito Juarez         -0.314957   0.996805  -0.316 0.752147    
# destinationChimalhuacan          -1.003572   1.546338  -0.649 0.516609    
# destinationCoacalco de Berriozal  0.106361   0.934207   0.114 0.909397    
# destinationCocotitlan            -0.788187   1.429000  -0.552 0.581470    
# destinationCoyoacan               0.548198   1.458092   0.376 0.707084    
# destinationCuajimalpa de Morelos  2.300784   1.033202   2.227 0.026360 *  
# destinationCuauhtemoc             0.351624   0.920672   0.382 0.702667    
# destinationCuautitlan            -0.460503   1.475692  -0.312 0.755113    
# destinationCuautitlan Izcalli    -0.081552   1.098474  -0.074 0.940845    
# destinationEcatepec de Morelos    0.086385   0.869163   0.099 0.920866    
# destinationGustavo A. Madero     -0.420416   0.995992  -0.422 0.673111    
# destinationIztapalapa             1.101183   1.414569   0.778 0.436632    
# destinationMiguel Hidalgo         0.677219   0.968706   0.699 0.484786    
# destinationNaucalpan de Juarez    0.144256   0.912771   0.158 0.874482    
# destinationNezahualcoyotl        -0.335009   1.086792  -0.308 0.758005    
# destinationNicolas Romero        -0.214073   0.901632  -0.237 0.812413    
# destinationOtro                   0.176351   0.927710   0.190 0.849307    
# destinationTecamec                0.496343   0.876460   0.566 0.571417    
# destinationTemamatla             -2.181717   1.336187  -1.633 0.103083    
# destinationTizayuca               1.812967   1.445098   1.255 0.210168    
# destinationTlalnepantla de Baz    0.627228   0.847764   0.740 0.459699    
# destinationTlalpan                2.452382   1.403576   1.747 0.081152 .  
# destinationTultitlan              2.036457   1.429643   1.424 0.154881    
# destinationVenustiano Carranza    0.591013   1.063192   0.556 0.578514    
# companions3 to 4                 -0.016265   0.217405  -0.075 0.940390    
# companionsMas                    -1.250517   0.516698  -2.420 0.015833 *  
# companionsNone                   -0.163402   0.108764  -1.502 0.133577    
# mode_security2                    0.584622   0.170528   3.428 0.000653 ***
# mode_security3                    1.012683   0.127997   7.912  1.4e-14 ***
# mode_security4                    1.304934   0.153138   8.521  < 2e-16 ***
# mode_security5                    1.686074   0.198780   8.482  < 2e-16 ***
# importance_safety2                0.541190   0.623279   0.868 0.385610    
# importance_safety3                0.507298   0.451884   1.123 0.262083    
# importance_safety4                0.447732   0.434826   1.030 0.303610    
# importance_safety5                0.368036   0.400399   0.919 0.358407    
# importance_safetyI                2.111643   1.220735   1.730 0.084223 .  
# urban_typology1                   0.366253   0.271030   1.351 0.177141    
# urban_typology2                   0.404214   0.214221   1.887 0.059698 .  
# urban_typology3                   0.639625   0.350632   1.824 0.068662 .  
# urban_typology4                   0.623811   0.456694   1.366 0.172518    
# urban_typology5                   0.115212   0.228825   0.503 0.614817    
# urban_typology6                   0.328155   0.228485   1.436 0.151505    
# urban_typology7                   0.246049   0.341879   0.720 0.472016    
# urban_typology9                   0.447640   0.253294   1.767 0.077735 .  
# hour7                                   NA         NA      NA       NA    
# hour12                           -0.894528   0.372982  -2.398 0.016802 *  
# hour13                           -0.978456   0.331900  -2.948 0.003333 ** 
# hour14                           -0.928182   0.328223  -2.828 0.004856 ** 
# hour15                           -0.882030   0.380376  -2.319 0.020768 *  
# hour16                           -0.752429   0.345896  -2.175 0.030031 *  
# hour17                           -1.039501   0.328649  -3.163 0.001648 ** 
# hour18                           -0.921586   0.334581  -2.754 0.006073 ** 
# hour19                                  NA         NA      NA       NA    
# week_day2                         0.342536   0.183897   1.863 0.063043 .  
# week_day3                         0.353165   0.199985   1.766 0.077956 .  
# week_day4                         0.112569   0.207189   0.543 0.587132    
# week_day5                        -0.066419   0.192254  -0.345 0.729867    
# week_day6                         0.346991   0.191084   1.816 0.069927 .  
# week_day7                        -0.379511   0.200781  -1.890 0.059258 .  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.122 on 552 degrees of freedom
# Multiple R-squared:  0.3809,  Adjusted R-squared:  0.2777 
# F-statistic: 3.692 on 92 and 552 DF,  p-value: < 2.2e-16


# Only trip variables

linear_model3 = lm(train$point_security~ +origin +destination +companions +trip_purpose, data = train)
summary(linear_model3)
summaries = getModelMetrics("Trip variables",linear_model3, summaries, train, test)


# Call:
# lm(formula = train$point_security ~ +origin + destination + companions + 
#     trip_purpose, data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.9250 -1.0278  0.0000  0.8448  2.9722 

# Coefficients:
#                                   Estimate Std. Error t value Pr(>|t|)   
# (Intercept)                       2.611822   0.980897   2.663  0.00797 **
# originAtenco                      0.881880   1.334245   0.661  0.50890   
# originAtizapan de Zaragoza        0.178951   0.402545   0.445  0.65681   
# originAzcapotzalco                0.507278   0.413073   1.228  0.21993   
# originBenito Juarez               0.776108   0.617109   1.258  0.20903   
# originChalco                      2.093300   0.863850   2.423  0.01569 * 
# originChimalhuacï¿½ï¿½n     1.053309   1.533971   0.687  0.49258   
# originChimalhuacan                1.117159   1.322921   0.844  0.39876   
# originCoacalco de Berriozal       0.234221   0.595043   0.394  0.69401   
# originCoyoacan                    0.986337   0.745096   1.324  0.18610   
# originCuajimalpa de Morelos       1.228008   0.983115   1.249  0.21213   
# originCuauhtemoc                  0.173926   0.562196   0.309  0.75715   
# originCuauhtlmoc                  2.386781   1.317742   1.811  0.07062 . 
# originCuautitlan                 -0.428701   1.346560  -0.318  0.75032   
# originCuautitlan Izcalli          1.670535   0.994937   1.679  0.09369 . 
# originEcatepec de Morelos         0.145352   0.466227   0.312  0.75533   
# originGustavo A. Madero           0.171865   0.566031   0.304  0.76152   
# originIztacalco                  -1.673964   0.985028  -1.699  0.08978 . 
# originIztapalapa                  2.142445   0.830062   2.581  0.01009 * 
# originMagdalena Contreras         2.197924   1.353649   1.624  0.10498   
# originMiguel Hidalgo              0.353976   0.576227   0.614  0.53926   
# originNaucalpan de Juarez         0.724791   0.641771   1.129  0.25921   
# originNextlalplan                -0.882501   1.338698  -0.659  0.51002   
# originNezahualcoyotl             -0.691868   0.889433  -0.778  0.43696   
# originNicolas Romero              0.579448   0.493014   1.175  0.24035   
# originOtro                        1.227535   0.591691   2.075  0.03846 * 
# originTecamec                     0.566570   0.475659   1.191  0.23409   
# originTexcoco                    -0.625663   0.986717  -0.634  0.52628   
# originTlalnepantla de Baz         0.239641   0.431327   0.556  0.57871   
# originTlalpan                     0.876996   0.974301   0.900  0.36843   
# originTultitlan                   0.675264   0.762397   0.886  0.37614   
# originVenustiano Carranza         0.673116   0.839896   0.801  0.42321   
# originZumpango                    1.104653   1.328464   0.832  0.40602   
# destinationAtenco                -0.510302   2.021175  -0.252  0.80076   
# destinationAtizapan de Zaragoza   0.271019   0.907153   0.299  0.76523   
# destinationAzcapotzalco           0.565001   0.915242   0.617  0.53726   
# destinationBenito Juarez         -0.217462   1.079339  -0.201  0.84040   
# destinationChimalhuacan          -0.446377   1.568585  -0.285  0.77607   
# destinationCoacalco de Berriozal -0.476367   1.012370  -0.471  0.63814   
# destinationCocotitlan            -1.446377   1.568585  -0.922  0.35687   
# destinationCoyoacan               0.221136   1.561107   0.142  0.88740   
# destinationCuajimalpa de Morelos  2.013941   1.125304   1.790  0.07403 . 
# destinationCuauhtemoc             0.389961   1.009321   0.386  0.69937   
# destinationCuautitlan            -0.877095   1.649447  -0.532  0.59510   
# destinationCuautitlan Izcalli    -0.092341   1.206156  -0.077  0.93900   
# destinationEcatepec de Morelos   -0.418524   0.941456  -0.445  0.65681   
# destinationGustavo A. Madero     -0.412855   1.072420  -0.385  0.70040   
# destinationIztapalapa             1.191698   1.539804   0.774  0.43929   
# destinationMiguel Hidalgo         0.824194   1.049322   0.785  0.43251   
# destinationNaucalpan de Juarez    0.278143   0.984933   0.282  0.77774   
# destinationNezahualcoyotl        -0.827811   1.185878  -0.698  0.48542   
# destinationNicolas Romero        -0.498948   0.971875  -0.513  0.60788   
# destinationOtro                  -0.067092   1.008852  -0.067  0.94700   
# destinationTecamec               -0.004186   0.942987  -0.004  0.99646   
# destinationTemamatla             -2.643419   1.465154  -1.804  0.07172 . 
# destinationTizayuca               1.524302   1.578632   0.966  0.33466   
# destinationTlalnepantla de Baz    0.312194   0.917711   0.340  0.73384   
# destinationTlalpan                2.046547   1.566511   1.306  0.19192   
# destinationTultitlan              1.490703   1.560220   0.955  0.33975   
# destinationVenustiano Carranza    0.167887   1.167855   0.144  0.88574   
# companions3 to 4                  0.016600   0.237972   0.070  0.94441   
# companionsMas                    -0.704801   0.551077  -1.279  0.20143   
# companionsNone                   -0.072599   0.121991  -0.595  0.55200   
# trip_purposeEstudio               0.235279   0.245394   0.959  0.33807   
# trip_purposeOtro                 -0.110849   0.184496  -0.601  0.54820   
# trip_purposeRecreacion           -0.208877   0.201985  -1.034  0.30151   
# trip_purposeTrabajo              -0.238199   0.169932  -1.402  0.16153   
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.253 on 578 degrees of freedom
# Multiple R-squared:  0.1919,  Adjusted R-squared:  0.09965 
# F-statistic:  2.08 on 66 and 578 DF,  p-value: 4.667e-06

# Only significant variables for the trip

linear_model3a = lm(train$point_security~ +origin +destination, data = train)
summary(linear_model3a)
summaries = getModelMetrics("Trip variables - Relevant",linear_model3a, summaries, train, test)

# Call:
# lm(formula = train$point_security ~ +origin + destination, data = train)

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -3.00524 -1.13644  0.04278  0.86356  2.86356 

# Coefficients:
#                                   Estimate Std. Error t value Pr(>|t|)   
# (Intercept)                       2.381297   0.977120   2.437  0.01510 * 
# originAtenco                      1.402828   1.315408   1.066  0.28666   
# originAtizapan de Zaragoza        0.247646   0.399445   0.620  0.53552   
# originAzcapotzalco                0.618703   0.409910   1.509  0.13175   
# originBenito Juarez               0.832470   0.614825   1.354  0.17626   
# originChalco                      2.060671   0.861346   2.392  0.01705 * 
# originChimalhuacï¿½ï¿½n     1.487914   1.520598   0.979  0.32823   
# originChimalhuacan                1.402828   1.315408   1.066  0.28666   
# originCoacalco de Berriozal       0.277673   0.588309   0.472  0.63711   
# originCoyoacan                    1.067750   0.739269   1.444  0.14918   
# originCuajimalpa de Morelos       1.402828   0.971370   1.444  0.14923   
# originCuauhtemoc                  0.321402   0.556437   0.578  0.56375   
# originCuauhtlmoc                  2.371107   1.316332   1.801  0.07217 . 
# originCuautitlan                  0.062513   1.319923   0.047  0.96224   
# originCuautitlan Izcalli          1.686095   0.994824   1.695  0.09063 . 
# originEcatepec de Morelos         0.197109   0.463168   0.426  0.67058   
# originGustavo A. Madero           0.235214   0.563522   0.417  0.67654   
# originIztacalco                  -1.597172   0.971370  -1.644  0.10066   
# originIztapalapa                  2.288776   0.826806   2.768  0.00582 **
# originMagdalena Contreras         2.192064   1.353732   1.619  0.10593   
# originMiguel Hidalgo              0.388954   0.573093   0.679  0.49760   
# originNaucalpan de Juarez         0.937914   0.632279   1.483  0.13851   
# originNextlalplan                -0.939329   1.338245  -0.702  0.48301   
# originNezahualcoyotl             -0.642585   0.881439  -0.729  0.46628   
# originNicolas Romero              0.634450   0.490447   1.294  0.19631   
# originOtro                        1.325897   0.587923   2.255  0.02449 * 
# originTecamec                     0.633317   0.473190   1.338  0.18129   
# originTexcoco                    -0.438408   0.978665  -0.448  0.65434   
# originTlalnepantla de Baz         0.328331   0.427249   0.768  0.44251   
# originTlalpan                     0.902828   0.971370   0.929  0.35305   
# originTultitlan                   0.785214   0.757491   1.037  0.30035   
# originVenustiano Carranza         0.694594   0.837303   0.830  0.40713   
# originZumpango                    1.062513   1.319923   0.805  0.42116   
# destinationAtenco                -0.784125   1.990202  -0.394  0.69373   
# destinationAtizapan de Zaragoza   0.215875   0.902196   0.239  0.81097   
# destinationAzcapotzalco           0.556190   0.911312   0.610  0.54189   
# destinationBenito Juarez         -0.336456   1.073487  -0.313  0.75407   
# destinationChimalhuacan          -0.578405   1.564822  -0.370  0.71179   
# destinationCoacalco de Berriozal -0.452612   1.006656  -0.450  0.65315   
# destinationCocotitlan            -1.578405   1.564822  -1.009  0.31354   
# destinationCoyoacan               0.290372   1.555409   0.187  0.85197   
# destinationCuajimalpa de Morelos  1.994880   1.114904   1.789  0.07409 . 
# destinationCuauhtemoc             0.309027   0.998531   0.309  0.75707   
# destinationCuautitlan            -1.628943   1.548474  -1.052  0.29325   
# destinationCuautitlan Izcalli    -0.332199   1.188105  -0.280  0.77988   
# destinationEcatepec de Morelos   -0.441968   0.936790  -0.472  0.63725   
# destinationGustavo A. Madero     -0.309620   1.070728  -0.289  0.77256   
# destinationIztapalapa             1.000000   1.536298   0.651  0.51536   
# destinationMiguel Hidalgo         0.666120   1.043074   0.639  0.52333   
# destinationNaucalpan de Juarez    0.176664   0.977474   0.181  0.85664   
# destinationNezahualcoyotl        -0.869211   1.182771  -0.735  0.46270   
# destinationNicolas Romero        -0.573361   0.967629  -0.593  0.55372   
# destinationOtro                  -0.186160   1.003785  -0.185  0.85293   
# destinationTecamec               -0.001579   0.937264  -0.002  0.99866   
# destinationTemamatla             -2.670073   1.458933  -1.830  0.06774 . 
# destinationTizayuca               1.421595   1.564822   0.908  0.36400   
# destinationTlalnepantla de Baz    0.247596   0.912481   0.271  0.78622   
# destinationTlalpan                2.371057   1.548474   1.531  0.12626   
# destinationTultitlan              1.371057   1.548474   0.885  0.37629   
# destinationVenustiano Carranza    0.027675   1.160652   0.024  0.98099   
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.254 on 585 degrees of freedom
# Multiple R-squared:  0.1798,  Adjusted R-squared:  0.09711 
# F-statistic: 2.174 on 59 and 585 DF,  p-value: 3.17e-06


# Only perception variables

linear_model4 = lm(train$point_security~ +mode_security +importance_safety +most_safe +least_safe, data = train)
summary(linear_model4)
summaries = getModelMetrics("Perception variables",linear_model4, summaries, train, test)

# Call:
# lm(formula = train$point_security ~ +mode_security + importance_safety + 
#     most_safe + least_safe, data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.7510 -0.8983  0.0639  0.8704  3.2580 

# Coefficients:
#                     Estimate Std. Error t value Pr(>|t|)    
# (Intercept)         1.659411   0.482798   3.437 0.000627 ***
# mode_security2      0.779690   0.172109   4.530 7.06e-06 ***
# mode_security3      1.037814   0.127589   8.134 2.23e-15 ***
# mode_security4      1.456673   0.150837   9.657  < 2e-16 ***
# mode_security5      1.721050   0.202954   8.480  < 2e-16 ***
# importance_safety2  0.076036   0.614164   0.124 0.901510    
# importance_safety3  0.291793   0.466742   0.625 0.532086    
# importance_safety4  0.449080   0.442021   1.016 0.310035    
# importance_safety5  0.175661   0.411965   0.426 0.669965    
# importance_safetyI  2.364184   1.275010   1.854 0.064173 .  
# most_safeMetro     -0.008451   0.131015  -0.065 0.948587    
# most_safePeseros   -0.090450   0.236986  -0.382 0.702838    
# most_safeTaxi      -0.164775   0.160337  -1.028 0.304497    
# most_safeTrolebus   0.247853   0.232310   1.067 0.286425    
# least_safeMetro     0.256697   0.305610   0.840 0.401257    
# least_safePeseros   0.071656   0.246980   0.290 0.771813    
# least_safeTaxi     -0.052958   0.271399  -0.195 0.845354    
# least_safeTrolebus  0.213335   0.362584   0.588 0.556493    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.203 on 627 degrees of freedom
# Multiple R-squared:  0.1917,  Adjusted R-squared:  0.1697 
# F-statistic: 8.744 on 17 and 627 DF,  p-value: < 2.2e-16

# Only perception variables

linear_model4a = lm(train$point_security~ +mode_security +importance_safety, data = train)
summary(linear_model4a)
summaries = getModelMetrics("Perception variables - relevant",linear_model4a, summaries, train, test)

# Call:
# lm(formula = train$point_security ~ +mode_security + importance_safety, 
#     data = train)

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -2.59899 -0.87478  0.07675  1.04344  3.12522 

# Coefficients:
#                    Estimate Std. Error t value Pr(>|t|)    
# (Intercept)         1.71488    0.40983   4.184 3.26e-05 ***
# mode_security2      0.77644    0.17128   4.533 6.94e-06 ***
# mode_security3      1.04847    0.12568   8.342 4.55e-16 ***
# mode_security4      1.46003    0.14711   9.925  < 2e-16 ***
# mode_security5      1.72421    0.19714   8.746  < 2e-16 ***
# importance_safety2  0.02312    0.60560   0.038    0.970    
# importance_safety3  0.24168    0.45881   0.527    0.599    
# importance_safety4  0.42522    0.43327   0.981    0.327    
# importance_safety5  0.15991    0.40349   0.396    0.692    
# importance_safetyI  2.23666    1.26705   1.765    0.078 .  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.2 on 635 degrees of freedom
# Multiple R-squared:  0.1847,  Adjusted R-squared:  0.1731 
# F-statistic: 15.98 on 9 and 635 DF,  p-value: < 2.2e-16


#Only  instant contextual information
linear_model5 = lm(train$point_security~ +haversine +urban_typology +total_passenger_count
                   +total_female_count +empty_seats +hour +week_day, data = train)
summary(linear_model5)
summaries = getModelMetrics("Instant contextual information",linear_model5, summaries, train, test)

# Call:
# lm(formula = train$point_security ~ +haversine + urban_typology + 
#     total_passenger_count + total_female_count + empty_seats + 
#     hour + week_day, data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.7142 -1.1428  0.1241  0.9798  2.7329 

# Coefficients:
#                         Estimate Std. Error t value Pr(>|t|)   
# (Intercept)            4.0065846  1.3216420   3.032  0.00254 **
# haversine             -0.0002549  0.0003018  -0.844  0.39874   
# urban_typology1        0.8768780  0.3594168   2.440  0.01498 * 
# urban_typology2        0.5625094  0.2397615   2.346  0.01929 * 
# urban_typology3        0.8160782  0.4131167   1.975  0.04867 * 
# urban_typology4        0.8700722  0.4877395   1.784  0.07493 . 
# urban_typology5       -0.0457901  0.2537344  -0.180  0.85685   
# urban_typology6        0.5059705  0.2503294   2.021  0.04369 * 
# urban_typology7        0.2937721  0.3795157   0.774  0.43919   
# urban_typology9        0.2687391  0.2939634   0.914  0.36097   
# total_passenger_count  0.0034328  0.0055269   0.621  0.53476   
# total_female_count    -0.0021647  0.0216472  -0.100  0.92038   
# empty_seats            0.0016853  0.0053526   0.315  0.75298   
# hour7                 -0.8888707  1.8345360  -0.485  0.62819   
# hour12                -1.3622035  1.3199809  -1.032  0.30248   
# hour13                -1.5799178  1.3061750  -1.210  0.22690   
# hour14                -1.6047736  1.3050877  -1.230  0.21930   
# hour15                -1.7387245  1.3244313  -1.313  0.18974   
# hour16                -1.4441592  1.3109584  -1.102  0.27106   
# hour17                -1.7824967  1.3062091  -1.365  0.17287   
# hour18                -1.7574324  1.3081193  -1.343  0.17961   
# hour19                -0.7359468  1.3361469  -0.551  0.58197   
# week_day2              0.2607241  0.1938933   1.345  0.17922   
# week_day3              0.1889647  0.2053683   0.920  0.35787   
# week_day4              0.1469295  0.2122748   0.692  0.48909   
# week_day5              0.0103530  0.2071330   0.050  0.96015   
# week_day6              0.2911934  0.2027110   1.436  0.15137   
# week_day7             -0.3019793  0.2125151  -1.421  0.15583   
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.275 on 617 degrees of freedom
# Multiple R-squared:  0.1063,  Adjusted R-squared:  0.06718 
# F-statistic: 2.718 on 27 and 617 DF,  p-value: 8.852e-06

#Only  instant contextual information
linear_model5a = lm(train$point_security~ +urban_typology, data = train)
summary(linear_model5a)
summaries = getModelMetrics("Instant contextual information - relevant",linear_model5a, summaries, train, test)

# Call:
# lm(formula = train$point_security ~ +urban_typology, data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.4118 -1.2990  0.1257  1.1206  2.7010 

# Coefficients:
#                 Estimate Std. Error t value Pr(>|t|)    
# (Intercept)      2.30952    0.19804  11.662  < 2e-16 ***
# urban_typology1  1.10224    0.25188   4.376 1.41e-05 ***
# urban_typology2  0.56482    0.21873   2.582  0.01004 *  
# urban_typology3  1.09957    0.33778   3.255  0.00119 ** 
# urban_typology4  1.09048    0.45160   2.415  0.01603 *  
# urban_typology5 -0.01055    0.23707  -0.045  0.96450    
# urban_typology6  0.56991    0.22562   2.526  0.01178 *  
# urban_typology7  0.35714    0.36157   0.988  0.32365    
# urban_typology9  0.36905    0.26198   1.409  0.15942    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.283 on 636 degrees of freedom
# Multiple R-squared:  0.06652, Adjusted R-squared:  0.05478 
# F-statistic: 5.665 on 8 and 636 DF,  p-value: 5.726e-07

# Only sociodemographic data
linear_model6 = lm(train$point_security~ +age +gender +education, data = train)
summary(linear_model6)
summaries = getModelMetrics("Sociodemographic data",linear_model6, summaries, train, test)

# Call:
# lm(formula = train$point_security ~ +age + gender + education, 
#     data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -1.9927 -1.1553  0.1364  1.1364  2.7081 

# Coefficients:
#                               Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                    2.63036    0.22309  11.790   <2e-16 ***
# age18-24                       0.07986    0.21684   0.368    0.713    
# age25-44                       0.12916    0.20681   0.625    0.533    
# age45-64                      -0.10663    0.22341  -0.477    0.633    
# age65+                         0.03000    0.33292   0.090    0.928    
# gendermale                     0.08985    0.10560   0.851    0.395    
# educationMaestria y Doctorado  0.75450    0.66946   1.127    0.260    
# educationPreparatoria          0.14337    0.13183   1.087    0.277    
# educationPrimaria             -0.09788    0.23224  -0.421    0.674    
# educationSecundaria            0.13115    0.15255   0.860    0.390    
# educationSin estudios         -0.45832    0.48814  -0.939    0.348    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.321 on 634 degrees of freedom
# Multiple R-squared:  0.01446, Adjusted R-squared:  -0.001085 
# F-statistic: 0.9302 on 10 and 634 DF,  p-value: 0.5046

# Only personal trip information
linear_model7 = lm(train$point_security~ +origin +destination +companions +trip_purpose, data = train)
summary(linear_model7)
summaries = getModelMetrics("Personal trip information",linear_model7, summaries, train, test)

# Call:
# lm(formula = train$point_security ~ +origin + destination + companions + 
#     trip_purpose, data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.9250 -1.0278  0.0000  0.8448  2.9722 

# Coefficients:
#                                   Estimate Std. Error t value Pr(>|t|)   
# (Intercept)                       2.611822   0.980897   2.663  0.00797 **
# originAtenco                      0.881880   1.334245   0.661  0.50890   
# originAtizapan de Zaragoza        0.178951   0.402545   0.445  0.65681   
# originAzcapotzalco                0.507278   0.413073   1.228  0.21993   
# originBenito Juarez               0.776108   0.617109   1.258  0.20903   
# originChalco                      2.093300   0.863850   2.423  0.01569 * 
# originChimalhuacï¿½ï¿½n     1.053309   1.533971   0.687  0.49258   
# originChimalhuacan                1.117159   1.322921   0.844  0.39876   
# originCoacalco de Berriozal       0.234221   0.595043   0.394  0.69401   
# originCoyoacan                    0.986337   0.745096   1.324  0.18610   
# originCuajimalpa de Morelos       1.228008   0.983115   1.249  0.21213   
# originCuauhtemoc                  0.173926   0.562196   0.309  0.75715   
# originCuauhtlmoc                  2.386781   1.317742   1.811  0.07062 . 
# originCuautitlan                 -0.428701   1.346560  -0.318  0.75032   
# originCuautitlan Izcalli          1.670535   0.994937   1.679  0.09369 . 
# originEcatepec de Morelos         0.145352   0.466227   0.312  0.75533   
# originGustavo A. Madero           0.171865   0.566031   0.304  0.76152   
# originIztacalco                  -1.673964   0.985028  -1.699  0.08978 . 
# originIztapalapa                  2.142445   0.830062   2.581  0.01009 * 
# originMagdalena Contreras         2.197924   1.353649   1.624  0.10498   
# originMiguel Hidalgo              0.353976   0.576227   0.614  0.53926   
# originNaucalpan de Juarez         0.724791   0.641771   1.129  0.25921   
# originNextlalplan                -0.882501   1.338698  -0.659  0.51002   
# originNezahualcoyotl             -0.691868   0.889433  -0.778  0.43696   
# originNicolas Romero              0.579448   0.493014   1.175  0.24035   
# originOtro                        1.227535   0.591691   2.075  0.03846 * 
# originTecamec                     0.566570   0.475659   1.191  0.23409   
# originTexcoco                    -0.625663   0.986717  -0.634  0.52628   
# originTlalnepantla de Baz         0.239641   0.431327   0.556  0.57871   
# originTlalpan                     0.876996   0.974301   0.900  0.36843   
# originTultitlan                   0.675264   0.762397   0.886  0.37614   
# originVenustiano Carranza         0.673116   0.839896   0.801  0.42321   
# originZumpango                    1.104653   1.328464   0.832  0.40602   
# destinationAtenco                -0.510302   2.021175  -0.252  0.80076   
# destinationAtizapan de Zaragoza   0.271019   0.907153   0.299  0.76523   
# destinationAzcapotzalco           0.565001   0.915242   0.617  0.53726   
# destinationBenito Juarez         -0.217462   1.079339  -0.201  0.84040   
# destinationChimalhuacan          -0.446377   1.568585  -0.285  0.77607   
# destinationCoacalco de Berriozal -0.476367   1.012370  -0.471  0.63814   
# destinationCocotitlan            -1.446377   1.568585  -0.922  0.35687   
# destinationCoyoacan               0.221136   1.561107   0.142  0.88740   
# destinationCuajimalpa de Morelos  2.013941   1.125304   1.790  0.07403 . 
# destinationCuauhtemoc             0.389961   1.009321   0.386  0.69937   
# destinationCuautitlan            -0.877095   1.649447  -0.532  0.59510   
# destinationCuautitlan Izcalli    -0.092341   1.206156  -0.077  0.93900   
# destinationEcatepec de Morelos   -0.418524   0.941456  -0.445  0.65681   
# destinationGustavo A. Madero     -0.412855   1.072420  -0.385  0.70040   
# destinationIztapalapa             1.191698   1.539804   0.774  0.43929   
# destinationMiguel Hidalgo         0.824194   1.049322   0.785  0.43251   
# destinationNaucalpan de Juarez    0.278143   0.984933   0.282  0.77774   
# destinationNezahualcoyotl        -0.827811   1.185878  -0.698  0.48542   
# destinationNicolas Romero        -0.498948   0.971875  -0.513  0.60788   
# destinationOtro                  -0.067092   1.008852  -0.067  0.94700   
# destinationTecamec               -0.004186   0.942987  -0.004  0.99646   
# destinationTemamatla             -2.643419   1.465154  -1.804  0.07172 . 
# destinationTizayuca               1.524302   1.578632   0.966  0.33466   
# destinationTlalnepantla de Baz    0.312194   0.917711   0.340  0.73384   
# destinationTlalpan                2.046547   1.566511   1.306  0.19192   
# destinationTultitlan              1.490703   1.560220   0.955  0.33975   
# destinationVenustiano Carranza    0.167887   1.167855   0.144  0.88574   
# companions3 to 4                  0.016600   0.237972   0.070  0.94441   
# companionsMas                    -0.704801   0.551077  -1.279  0.20143   
# companionsNone                   -0.072599   0.121991  -0.595  0.55200   
# trip_purposeEstudio               0.235279   0.245394   0.959  0.33807   
# trip_purposeOtro                 -0.110849   0.184496  -0.601  0.54820   
# trip_purposeRecreacion           -0.208877   0.201985  -1.034  0.30151   
# trip_purposeTrabajo              -0.238199   0.169932  -1.402  0.16153   
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.253 on 578 degrees of freedom
# Multiple R-squared:  0.1919,  Adjusted R-squared:  0.09965 
# F-statistic:  2.08 on 66 and 578 DF,  p-value: 4.667e-06

linear_model7a = lm(train$point_security~ +origin +destination, data = train)
summary(linear_model7a)
summaries = getModelMetrics("Personal trip information - relevant",linear_model7a, summaries, train, test)

# Call:
# lm(formula = train$point_security ~ +origin + destination, data = train)

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -3.00524 -1.13644  0.04278  0.86356  2.86356 

# Coefficients:
#                                   Estimate Std. Error t value Pr(>|t|)   
# (Intercept)                       2.381297   0.977120   2.437  0.01510 * 
# originAtenco                      1.402828   1.315408   1.066  0.28666   
# originAtizapan de Zaragoza        0.247646   0.399445   0.620  0.53552   
# originAzcapotzalco                0.618703   0.409910   1.509  0.13175   
# originBenito Juarez               0.832470   0.614825   1.354  0.17626   
# originChalco                      2.060671   0.861346   2.392  0.01705 * 
# originChimalhuacï¿½ï¿½n     1.487914   1.520598   0.979  0.32823   
# originChimalhuacan                1.402828   1.315408   1.066  0.28666   
# originCoacalco de Berriozal       0.277673   0.588309   0.472  0.63711   
# originCoyoacan                    1.067750   0.739269   1.444  0.14918   
# originCuajimalpa de Morelos       1.402828   0.971370   1.444  0.14923   
# originCuauhtemoc                  0.321402   0.556437   0.578  0.56375   
# originCuauhtlmoc                  2.371107   1.316332   1.801  0.07217 . 
# originCuautitlan                  0.062513   1.319923   0.047  0.96224   
# originCuautitlan Izcalli          1.686095   0.994824   1.695  0.09063 . 
# originEcatepec de Morelos         0.197109   0.463168   0.426  0.67058   
# originGustavo A. Madero           0.235214   0.563522   0.417  0.67654   
# originIztacalco                  -1.597172   0.971370  -1.644  0.10066   
# originIztapalapa                  2.288776   0.826806   2.768  0.00582 **
# originMagdalena Contreras         2.192064   1.353732   1.619  0.10593   
# originMiguel Hidalgo              0.388954   0.573093   0.679  0.49760   
# originNaucalpan de Juarez         0.937914   0.632279   1.483  0.13851   
# originNextlalplan                -0.939329   1.338245  -0.702  0.48301   
# originNezahualcoyotl             -0.642585   0.881439  -0.729  0.46628   
# originNicolas Romero              0.634450   0.490447   1.294  0.19631   
# originOtro                        1.325897   0.587923   2.255  0.02449 * 
# originTecamec                     0.633317   0.473190   1.338  0.18129   
# originTexcoco                    -0.438408   0.978665  -0.448  0.65434   
# originTlalnepantla de Baz         0.328331   0.427249   0.768  0.44251   
# originTlalpan                     0.902828   0.971370   0.929  0.35305   
# originTultitlan                   0.785214   0.757491   1.037  0.30035   
# originVenustiano Carranza         0.694594   0.837303   0.830  0.40713   
# originZumpango                    1.062513   1.319923   0.805  0.42116   
# destinationAtenco                -0.784125   1.990202  -0.394  0.69373   
# destinationAtizapan de Zaragoza   0.215875   0.902196   0.239  0.81097   
# destinationAzcapotzalco           0.556190   0.911312   0.610  0.54189   
# destinationBenito Juarez         -0.336456   1.073487  -0.313  0.75407   
# destinationChimalhuacan          -0.578405   1.564822  -0.370  0.71179   
# destinationCoacalco de Berriozal -0.452612   1.006656  -0.450  0.65315   
# destinationCocotitlan            -1.578405   1.564822  -1.009  0.31354   
# destinationCoyoacan               0.290372   1.555409   0.187  0.85197   
# destinationCuajimalpa de Morelos  1.994880   1.114904   1.789  0.07409 . 
# destinationCuauhtemoc             0.309027   0.998531   0.309  0.75707   
# destinationCuautitlan            -1.628943   1.548474  -1.052  0.29325   
# destinationCuautitlan Izcalli    -0.332199   1.188105  -0.280  0.77988   
# destinationEcatepec de Morelos   -0.441968   0.936790  -0.472  0.63725   
# destinationGustavo A. Madero     -0.309620   1.070728  -0.289  0.77256   
# destinationIztapalapa             1.000000   1.536298   0.651  0.51536   
# destinationMiguel Hidalgo         0.666120   1.043074   0.639  0.52333   
# destinationNaucalpan de Juarez    0.176664   0.977474   0.181  0.85664   
# destinationNezahualcoyotl        -0.869211   1.182771  -0.735  0.46270   
# destinationNicolas Romero        -0.573361   0.967629  -0.593  0.55372   
# destinationOtro                  -0.186160   1.003785  -0.185  0.85293   
# destinationTecamec               -0.001579   0.937264  -0.002  0.99866   
# destinationTemamatla             -2.670073   1.458933  -1.830  0.06774 . 
# destinationTizayuca               1.421595   1.564822   0.908  0.36400   
# destinationTlalnepantla de Baz    0.247596   0.912481   0.271  0.78622   
# destinationTlalpan                2.371057   1.548474   1.531  0.12626   
# destinationTultitlan              1.371057   1.548474   0.885  0.37629   
# destinationVenustiano Carranza    0.027675   1.160652   0.024  0.98099   
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.254 on 585 degrees of freedom
# Multiple R-squared:  0.1798,  Adjusted R-squared:  0.09711 
# F-statistic: 2.174 on 59 and 585 DF,  p-value: 3.17e-06

library(rpart)
library(rpart.plot)
CART = rpart(train$point_security~.,
             data = train, 
             method="class")

rpart.plot(CART)


# KNN model
new_safety_data <- read.table(file="safety_data_clean.csv", header = TRUE, na.strings=c("", "NA"), sep=",")

for(i in names(new_safety_data)){
  new_safety_data[[i]] <- as.numeric(new_safety_data[[i]])
}

new_smp_size <- floor(0.7 * nrow(new_safety_data))
## set the seed to make your partition reproductible
set.seed(888)
new_train_ind <- sample(seq_len(nrow(new_safety_data)), size = new_smp_size)

new_train <- new_safety_data[new_train_ind, ]
new_test <- new_safety_data[-new_train_ind, ]

nn3 <- knn (new_train, new_test, new_train$point_security, k=3)
table(nn3, new_test$point_security)
prop.table(table(nn3, new_test$point_security))

nn5 <- knn (new_train, new_test, new_train$point_security, k=5)
table(nn5, new_test$point_security)
prop.table(table(nn5, new_test$point_security))

# seems to be best
nn7 <- knn (new_train, new_test, new_train$point_security, k=7)
table(nn7, new_test$point_security)
prop.table(table(nn7, new_test$point_security))

nn9 <- knn (new_train, new_test, new_train$point_security, k=9)
table(nn9, new_test$point_security)
prop.table(table(nn9, new_test$point_security))

nn11 <- knn (new_train, new_test, new_train$point_security, k=11)
table(nn11, new_test$point_security)
prop.table(table(nn11, new_test$point_security))