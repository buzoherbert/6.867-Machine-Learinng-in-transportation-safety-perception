
install.packages("GGally", dependencies = TRUE);
install.packages("ggplot2", dependencies = TRUE);
install.packages("caTools", dependencies = TRUE);
install.packages("magrittr")
install.packages("class")


# Graphing library
 library(GGally);
 library(ggplot2);


# For sample splitting
library("caTools");

# For removing unused levels in test data
library(magrittr);

# For knn model
library("class");

# loading file
plot_data <- read.table(file="safety_data_clean.csv", header = TRUE, na.strings=c("", "NA"), sep=",")

# Making sure variables are treated properly
for(i in names(plot_data)){
  if(i == "total_female_count"){
    print(i)
    plot_data[["total_female_count"]] = as.numeric(as.character(plot_data[["total_female_count"]]));
  } else if(i == "total_passenger_count"){
    print(i)
    plot_data[["total_passenger_count"]] = as.numeric(as.character(plot_data[["total_passenger_count"]]));
  } else if(i == "empty_seats") {
    print(i)
    plot_data[["empty_seats"]] = as.numeric(as.character(plot_data[["empty_seats"]]));
  } else if(i == "point_security") {
    print(i)
    plot_data[["point_security"]] = as.numeric(as.character(plot_data[["point_security"]]));
  } else if(i == "haversine") {
    print(i)
    plot_data[["haversine"]] = as.numeric(plot_data[["haversine"]]);      
  } else {
    print("default")
    plot_data[[i]] <- as.factor(plot_data[[i]])
  } 
}

# Some fields have to be treated as numeric.
# TODO improve this so we don't convert the data twice.
plot_data[["total_female_count"]] = as.numeric(as.character(plot_data[["total_female_count"]]));
plot_data[["total_passenger_count"]] = as.numeric(as.character(plot_data[["total_passenger_count"]]));
plot_data[["empty_seats"]] = as.numeric(as.character(plot_data[["empty_seats"]]));
plot_data[["point_security"]] = as.numeric(as.character(plot_data[["point_security"]]));
plot_data[["haversine"]] = as.numeric(plot_data[["haversine"]]);


# Getting a summary of the data
# summary(plot_data)

# plotting the data
#to_plot = plot_data;
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
# Removing categories not on the training set
# Residual standard error: 1.092 on 550 degrees of freedom
# Multiple R-squared:  0.416,	Adjusted R-squared:  0.2727 
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


########################
## Linear regression model


linear_model = lm(train$point_security~., data = train)
summary(linear_model)

#Call:
  # lm(formula = train$point_security ~ ., data = train)
  
  # Residuals:
#     Min      1Q  Median      3Q     Max 
# -3.3654 -0.7034  0.0000  0.6053  2.9270 
  
# Coefficients: (5 not defined because of singularities)
#                                     Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                        2.806e+00  1.760e+00   1.594  0.11147    
# bus_or_pedVan                      3.207e-02  1.963e-01   0.163  0.87027    
# inside_or_outsideFuera CETRAM     -1.010e-01  1.249e-01  -0.809  0.41913    
# gendermale                        -6.937e-03  9.816e-02  -0.071  0.94368    
# age0-17                            1.127e+00  7.030e-01   1.603  0.10948    
# age18-24                           9.922e-01  6.855e-01   1.447  0.14834    
# age25-44                           1.129e+00  6.870e-01   1.643  0.10100    
# age45-64                           8.774e-01  6.936e-01   1.265  0.20642    
# age65+                             9.814e-01  7.135e-01   1.375  0.16954    
# educationLicenciatura             -7.413e-01  7.196e-01  -1.030  0.30340    
# educationMaestria y Doctorado     -1.044e+00  8.311e-01  -1.256  0.20973    
# educationPreparatoria             -5.926e-01  7.150e-01  -0.829  0.40759    
# educationPrimaria                 -7.363e-01  7.375e-01  -0.998  0.31854    
# educationSecundaria               -5.953e-01  7.143e-01  -0.833  0.40503    
# educationSin estudios             -9.708e-01  8.422e-01  -1.153  0.24952    
# originAcolman                     -1.351e+00  1.441e+00  -0.938  0.34884    
# originAlvaro Obregon              -1.763e+00  9.084e-01  -1.941  0.05278 .  
# originAtenco                      -1.099e+00  1.431e+00  -0.768  0.44275    
# originAtizapan de Zaragoza        -2.355e+00  8.420e-01  -2.797  0.00535 ** 
# originAzcapotzalco                -1.954e+00  8.424e-01  -2.320  0.02073 *  
# originBenito Juarez               -1.565e+00  9.237e-01  -1.694  0.09084 .  
# originChimalhuacï¿½ï¿½n     -1.208e+00  1.524e+00  -0.793  0.42828    
# originCoacalco de Berriozal       -1.892e+00  9.231e-01  -2.050  0.04083 *  
# originCoyoacan                    -2.394e+00  1.148e+00  -2.085  0.03749 *  
# originCuajimalpa de Morelos       -1.887e+00  1.412e+00  -1.336  0.18200    
# originCuauhtemoc                  -2.118e+00  8.948e-01  -2.367  0.01828 *  
# originCuauhtlmoc                  -1.110e+00  1.460e+00  -0.760  0.44730    
# originCuautitlan                  -1.908e+00  1.400e+00  -1.362  0.17361    
# originCuautitlan Izcalli          -1.179e+00  1.207e+00  -0.977  0.32910    
# originEcatepec de Morelos         -2.097e+00  8.641e-01  -2.427  0.01553 *  
# originGustavo A. Madero           -1.742e+00  9.045e-01  -1.926  0.05461 .  
# originIxtapaluca                  -1.692e+00  1.414e+00  -1.197  0.23178    
# originIztacalco                   -2.955e+00  1.198e+00  -2.467  0.01393 *  
# originIztapalapa                  -1.009e+00  1.069e+00  -0.944  0.34558    
# originMagdalena Contreras         -3.244e-01  1.430e+00  -0.227  0.82068    
# originMiguel Hidalgo              -1.939e+00  9.371e-01  -2.069  0.03903 *  
# originNaucalpan de Juarez         -2.003e+00  9.383e-01  -2.135  0.03322 *  
# originNextlalplan                 -2.604e+00  1.436e+00  -1.814  0.07030 .  
# originNezahualcoyotl              -1.617e+00  1.583e+00  -1.021  0.30763    
# originNicolas Romero              -2.295e+00  8.825e-01  -2.601  0.00955 ** 
# originOtro                        -1.827e+00  9.441e-01  -1.935  0.05348 .  
# originTecamec                     -1.747e+00  8.797e-01  -1.986  0.04749 *  
# originTexcoco                     -2.296e+00  1.160e+00  -1.980  0.04822 *  
# originTiahuac                     -1.633e+00  1.358e+00  -1.202  0.22974    
# originTlalnepantla de Baz         -1.981e+00  8.456e-01  -2.342  0.01951 *  
# originTlalpan                     -1.558e+00  1.162e+00  -1.341  0.18040    
# originTultitlan                   -1.668e+00  9.642e-01  -1.730  0.08412 .  
# originVenustiano Carranza         -2.201e+00  1.171e+00  -1.880  0.06062 .  
# originZumpango                    -1.709e+00  1.511e+00  -1.131  0.25867    
# destinationAlvaro Obregon          8.198e-01  1.371e+00   0.598  0.55022    
# destinationAtenco                 -1.848e+00  2.339e+00  -0.790  0.42978    
# destinationAtizapan de Zaragoza    1.523e+00  1.082e+00   1.408  0.15973    
# destinationAzcapotzalco            1.680e+00  1.076e+00   1.561  0.11917    
# destinationBenito Juarez           1.210e+00  1.217e+00   0.994  0.32051    
# destinationChimalhuacan            9.233e-01  1.754e+00   0.526  0.59886    
# destinationCoacalco de Berriozal   1.402e+00  1.133e+00   1.237  0.21665    
# destinationCoyoacan                1.501e+00  1.358e+00   1.105  0.26957    
# destinationCuajimalpa de Morelos   2.972e+00  1.233e+00   2.410  0.01630 *  
# destinationCuauhtemoc              1.284e+00  1.161e+00   1.106  0.26920    
# destinationCuautitlan              5.476e-01  1.370e+00   0.400  0.68946    
# destinationCuautitlan Izcalli      9.045e-01  1.562e+00   0.579  0.56268    
# destinationEcatepec de Morelos     1.235e+00  1.092e+00   1.131  0.25859    
# destinationGustavo A. Madero       8.477e-01  1.182e+00   0.717  0.47349    
# destinationIztacalco               8.463e-01  1.616e+00   0.524  0.60073    
# destinationIztapalapa              2.159e+00  1.566e+00   1.379  0.16861    
# destinationMiguel Hidalgo          1.992e+00  1.174e+00   1.696  0.09041 .  
# destinationNaucalpan de Juarez     1.501e+00  1.100e+00   1.364  0.17307    
# destinationNezahualcoyotl          7.135e-01  1.210e+00   0.589  0.55582    
# destinationNicolas Romero          9.871e-01  1.117e+00   0.884  0.37718    
# destinationOtro                    1.444e+00  1.148e+00   1.257  0.20919    
# destinationTecamec                 1.243e+00  1.096e+00   1.134  0.25733    
# destinationTemamatla              -8.183e-01  1.497e+00  -0.547  0.58486    
# destinationTeotihuacï¿½ï¿½n         NA         NA      NA       NA    
# destinationTizayuca                2.379e+00  1.596e+00   1.491  0.13663    
# destinationTlalnepantla de Baz     1.644e+00  1.092e+00   1.506  0.13273    
# destinationTlalpan                 2.776e+00  1.356e+00   2.047  0.04113 *  
# destinationTultitlan               1.873e+00  1.344e+00   1.394  0.16398    
# destinationVenustiano Carranza     1.531e+00  1.257e+00   1.218  0.22367    
# companions1 to 2                   2.232e-01  7.377e-01   0.303  0.76236    
# companions3 to 4                   7.408e-02  7.619e-01   0.097  0.92258    
# companionsMas                     -1.678e-01  8.517e-01  -0.197  0.84385    
# companionsNone                     2.991e-01  7.332e-01   0.408  0.68342    
# trip_purposeCompra                 7.626e-01  6.932e-01   1.100  0.27175    
# trip_purposeEstudio                9.176e-01  7.030e-01   1.305  0.19237    
# trip_purposeOtro                   8.046e-01  6.820e-01   1.180  0.23860    
# trip_purposeRecreacion             6.999e-01  6.919e-01   1.012  0.31218    
# trip_purposeTrabajo                5.863e-01  6.764e-01   0.867  0.38639    
# mode_security2                     4.309e-01  1.667e-01   2.585  0.01001 *  
# mode_security3                     8.843e-01  1.266e-01   6.983 8.39e-12 ***
# mode_security4                     1.302e+00  1.501e-01   8.673  < 2e-16 ***
# mode_security5                     1.313e+00  2.011e-01   6.530 1.50e-10 ***
# importance_safety1                -1.515e+00  1.303e+00  -1.163  0.24525    
# importance_safety2                -1.312e+00  1.426e+00  -0.920  0.35817    
# importance_safety3                -1.184e+00  1.279e+00  -0.926  0.35475    
# importance_safety4                -1.000e+00  1.270e+00  -0.788  0.43125    
# importance_safety5                -1.143e+00  1.261e+00  -0.907  0.36498    
# most_safeBRT                       3.825e-01  3.534e-01   1.082  0.27965    
# most_safeMetro                     3.892e-01  3.443e-01   1.130  0.25878    
# most_safePeseros                   7.382e-01  3.819e-01   1.933  0.05373 .  
# most_safeTaxi                      5.574e-01  3.554e-01   1.568  0.11741    
# most_safeTrolebus                  8.849e-01  3.835e-01   2.308  0.02140 *  
# least_safeBRT                     -2.127e-01  4.501e-01  -0.473  0.63669    
# least_safeMetro                    1.524e-02  4.303e-01   0.035  0.97176    
# least_safePeseros                 -3.709e-02  3.891e-01  -0.095  0.92409    
# least_safeTaxi                     1.786e-01  4.021e-01   0.444  0.65716    
# least_safeTrolebus                -7.255e-02  4.548e-01  -0.160  0.87332    
# base_study_zoneEl Rosario          4.091e-01  3.229e-01   1.267  0.20573    
# busdestinationHeroes Tecamec       3.434e-01  1.855e-01   1.852  0.06462 .  
# busdestinationMexico Nueva        -1.476e-02  1.712e-01  -0.086  0.93133    
# busdestinationMexipuerto                  NA         NA      NA       NA    
# busdestinationTacuba              -1.019e+00  4.554e-01  -2.238  0.02560 *  
# busdestinationTexcoco              1.134e+00  2.421e+00   0.468  0.63970    
# total_seats35                             NA         NA      NA       NA    
# haversine                          2.162e-04  3.442e-04   0.628  0.53011    
# urban_typology1                    9.677e-01  3.736e-01   2.590  0.00985 ** 
# urban_typology2                    4.867e-01  2.303e-01   2.113  0.03504 *  
# urban_typology3                    7.936e-01  3.753e-01   2.115  0.03490 *  
# urban_typology4                    1.129e+00  4.598e-01   2.456  0.01436 *  
# urban_typology5                    2.276e-01  2.338e-01   0.973  0.33077    
# urban_typology6                    6.421e-01  2.402e-01   2.673  0.00774 ** 
# urban_typology7                    4.916e-01  3.223e-01   1.525  0.12775    
# urban_typology9                    5.645e-01  2.868e-01   1.968  0.04956 *  
# total_passenger_count              4.501e-05  6.071e-03   0.007  0.99409    
# total_female_count                 3.689e-03  1.909e-02   0.193  0.84680    
# empty_seats                       -3.222e-04  5.772e-03  -0.056  0.95550    
# hour7                                     NA         NA      NA       NA    
# hour12                            -1.770e+00  1.187e+00  -1.491  0.13658    
# hour13                            -1.749e+00  1.172e+00  -1.492  0.13620    
# hour14                            -1.556e+00  1.173e+00  -1.327  0.18512    
# hour15                            -1.621e+00  1.196e+00  -1.355  0.17591    
# hour16                            -1.610e+00  1.179e+00  -1.365  0.17284    
# hour17                            -1.765e+00  1.173e+00  -1.505  0.13282    
# hour18                            -1.676e+00  1.173e+00  -1.428  0.15375    
# hour19                            -1.009e+00  1.186e+00  -0.850  0.39550    
# hour20                                    NA         NA      NA       NA    
# week_day2                          2.084e-01  1.981e-01   1.052  0.29328    
# week_day3                          1.455e-01  2.004e-01   0.726  0.46797    
# week_day4                          9.583e-02  2.122e-01   0.452  0.65175    
# week_day5                         -2.273e-01  1.958e-01  -1.161  0.24615    
# week_day6                          2.446e-01  2.025e-01   1.208  0.22766    
# week_day7                         -3.865e-01  2.122e-01  -1.821  0.06908 .  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Function to get the metrics of each model and add it to the summaries table
# It returns the summaries table passed to it with added data about the new model
# If summaries table doesn't exist, it creates it
getModelMetrics <- function(model_name, linear_model, summaries_table, train_data, test_data) {
  test_data = na.omit(remove_missing_levels (fit=linear_model, test_data=test_data));
  
  # If the summaries table is not a data frame, it gets initialized
  if(!is.data.frame(summaries_table)){
    summaries_table <- data.frame(matrix(ncol = 9, nrow = 0));
    names <- c("model", "r2", "sse", "pred_means", "sst", "osr2", "rmse", "mae", "var_num");
    colnames(summaries_table) <- names;
  }
  
  pred_data = predict(linear_model, newdata = test_data);
  SSE = sum((pred_data - test_data$point_security)^2);
  pred_mean = mean(train_data$point_security);
  SST = sum((pred_mean - test_data$point_security)^2);
  OSR2 = 1-SSE/SST;
  RMSE = sqrt(sum((pred_data - test_data$point_security)^2)/nrow(test_data));
  MAE = sum(abs(pred_data - test_data$point_security))/nrow(test_data);
  
  i = nrow(summaries_table) + 1;
  summaries_table[i, "model"] = model_name;
  summaries_table[i, "r2"] = summary(linear_model)$r.squared;
  summaries_table[i, "sse"] = SSE;
  summaries_table[i, "pred_means"] = pred_mean;
  summaries_table[i, "sst"] = SST;
  summaries_table[i, "osr2"] = OSR2;
  summaries_table[i, "rmse"] = RMSE;
  summaries_table[i, "mae"] = MAE;
  summaries_table[i,"var_num"] = length(names(linear_model$coefficients));
  
  return(summaries_table);
}

summaries = getModelMetrics("Initial model",linear_model, summaries, train, test);

# Taking out the least relevant variables
linear_model2 = lm(train$point_security~. 
                   -bus_or_ped -inside_or_outside -gender -age -education 
                   -companions - trip_purpose -importance_safety -least_safe
                   -total_passenger_count -total_female_count -empty_seats
                   -base_study_zone -total_seats -haversine
                   -hour, data = train)
summary(linear_model2);
summaries = getModelMetrics("Relevant variables",linear_model2, summaries, train, test);


#Call:
  # lm(formula = train$point_security ~ . - bus_or_ped - inside_or_outside - 
  #     gender - age - education - companions - trip_purpose - importance_safety - 
  #     least_safe - total_passenger_count - total_female_count - 
  #     empty_seats - base_study_zone - total_seats - haversine - 
  #     hour, data = train)
  
  # Residuals:
  #     Min      1Q  Median      3Q     Max 
  # -3.0145 -0.7063  0.0000  0.6799  3.3167 
  
  # Coefficients: (1 not defined because of singularities)
#                                   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                        1.11420    0.51865   2.148 0.032096 *  
# originAcolman                     -1.07145    1.40904  -0.760 0.447307    
# originAlvaro Obregon              -1.60651    0.88166  -1.822 0.068936 .  
# originAtenco                      -0.61575    1.37045  -0.449 0.653378    
# originAtizapan de Zaragoza        -1.96831    0.80772  -2.437 0.015107 *  
# originAzcapotzalco                -1.74585    0.81415  -2.144 0.032407 *  
# originBenito Juarez               -1.33130    0.89172  -1.493 0.135981    
# originChimalhuacï¿½ï¿½n     -0.71219    1.47218  -0.484 0.628728    
# originCoacalco de Berriozal       -1.75886    0.88776  -1.981 0.048026 *  
# originCoyoacan                    -2.27339    1.11843  -2.033 0.042529 *  
# originCuajimalpa de Morelos       -1.16158    1.36232  -0.853 0.394197    
# originCuauhtemoc                  -1.92170    0.86354  -2.225 0.026431 *  
# originCuauhtlmoc                  -1.00658    1.41855  -0.710 0.478241    
# originCuautitlan                  -1.61318    1.36364  -1.183 0.237284    
# originCuautitlan Izcalli          -0.99574    1.14724  -0.868 0.385775    
# originEcatepec de Morelos         -1.88351    0.83931  -2.244 0.025191 *  
# originGustavo A. Madero           -1.45615    0.87385  -1.666 0.096165 .  
# originIxtapaluca                  -1.81199    1.37502  -1.318 0.188081    
# originIztacalco                   -2.75860    1.13038  -2.440 0.014961 *  
# originIztapalapa                  -0.70059    1.03950  -0.674 0.500589    
# originMagdalena Contreras         -0.14982    1.39622  -0.107 0.914583    
# originMiguel Hidalgo              -1.74357    0.90726  -1.922 0.055108 .  
# originNaucalpan de Juarez         -1.64099    0.89666  -1.830 0.067734 .  
# originNextlalplan                 -2.48058    1.39467  -1.779 0.075813 .  
# originNezahualcoyotl              -1.73486    1.39538  -1.243 0.214252    
# originNicolas Romero              -2.00213    0.85094  -2.353 0.018955 *  
# originOtro                        -1.58399    0.91240  -1.736 0.083071 .  
# originTecamec                     -1.40580    0.84484  -1.664 0.096642 .  
# originTexcoco                     -1.85636    1.12226  -1.654 0.098628 .  
# originTiahuac                     -0.59596    1.17811  -0.506 0.613139    
# originTlalnepantla de Baz         -1.69397    0.81573  -2.077 0.038264 *  
# originTlalpan                     -1.25214    1.11815  -1.120 0.263239    
# originTultitlan                   -1.17114    0.93059  -1.258 0.208709    
# originVenustiano Carranza         -1.95097    1.13839  -1.714 0.087088 .  
# originZumpango                    -1.01452    1.45298  -0.698 0.485303    
# destinationAlvaro Obregon          1.39970    1.21232   1.155 0.248733    
# destinationAtenco                  0.27535    1.83207   0.150 0.880581    
# destinationAtizapan de Zaragoza    2.19091    0.89872   2.438 0.015068 *  
# destinationAzcapotzalco            2.15907    0.90085   2.397 0.016851 *  
# destinationBenito Juarez           1.75233    1.03509   1.693 0.090992 .  
# destinationChimalhuacan            1.77936    1.44803   1.229 0.219628    
# destinationCoacalco de Berriozal   2.01993    0.95572   2.114 0.034972 *  
# destinationCoyoacan                1.88367    1.19554   1.576 0.115653    
# destinationCuajimalpa de Morelos   3.62295    1.06702   3.395 0.000731 ***
# destinationCuauhtemoc              1.77857    1.00321   1.773 0.076760 .  
# destinationCuautitlan              0.96965    1.19179   0.814 0.416196    
# destinationCuautitlan Izcalli      1.55492    1.41854   1.096 0.273460    
# destinationEcatepec de Morelos     1.91256    0.91707   2.086 0.037447 *  
# destinationGustavo A. Madero       1.56421    1.00357   1.559 0.119614    
# destinationIztacalco               1.18561    1.43254   0.828 0.408212    
# destinationIztapalapa              2.90540    1.42432   2.040 0.041807 *  
# destinationMiguel Hidalgo          2.56499    0.98249   2.611 0.009263 ** 
# destinationNaucalpan de Juarez     1.97191    0.90369   2.182 0.029495 *  
# destinationNezahualcoyotl          1.25428    1.03855   1.208 0.227633    
# destinationNicolas Romero          1.51013    0.94023   1.606 0.108775    
# destinationOtro                    2.11537    0.97530   2.169 0.030481 *  
# destinationTecamec                 1.95043    0.91857   2.123 0.034139 *  
# destinationTemamatla              -0.26220    1.35327  -0.194 0.846435    
# destinationTeotihuacï¿½ï¿½n       NA         NA      NA       NA    
# destinationTizayuca                2.80327    1.45144   1.931 0.053912 .  
# destinationTlalnepantla de Baz     2.17246    0.90798   2.393 0.017037 *  
# destinationTlalpan                 3.28544    1.19900   2.740 0.006326 ** 
# destinationTultitlan               2.37324    1.19819   1.981 0.048086 *  
# destinationVenustiano Carranza     2.22002    1.10047   2.017 0.044108 *  
# mode_security2                     0.44706    0.15933   2.806 0.005183 ** 
# mode_security3                     0.94348    0.11874   7.946 9.71e-15 ***
# mode_security4                     1.38436    0.13992   9.894  < 2e-16 ***
# mode_security5                     1.35588    0.19191   7.065 4.51e-12 ***
# most_safeBRT                       0.36547    0.29071   1.257 0.209189    
# most_safeMetro                     0.37813    0.28199   1.341 0.180458    
# most_safePeseros                   0.62921    0.33362   1.886 0.059785 .  
# most_safeTaxi                      0.51449    0.29292   1.756 0.079527 .  
# most_safeTrolebus                  0.78705    0.32615   2.413 0.016115 *  
# busdestinationHeroes Tecamec      -0.08652    0.28603  -0.302 0.762392    
# busdestinationMexico Nueva        -0.07865    0.15203  -0.517 0.605113    
# busdestinationMexipuerto          -0.37165    0.28028  -1.326 0.185342    
# busdestinationTacuba              -0.92155    0.41656  -2.212 0.027326 *  
# busdestinationTexcoco              0.87726    2.23261   0.393 0.694512    
# urban_typology1                    0.86383    0.25988   3.324 0.000942 ***
# urban_typology2                    0.38534    0.21042   1.831 0.067557 .  
# urban_typology3                    0.68098    0.31545   2.159 0.031271 *  
# urban_typology4                    1.02457    0.41333   2.479 0.013458 *  
# urban_typology5                    0.18459    0.22007   0.839 0.401949    
# urban_typology6                    0.51555    0.21951   2.349 0.019166 *  
# urban_typology7                    0.38537    0.29810   1.293 0.196597    
# urban_typology9                    0.47988    0.25313   1.896 0.058480 .  
# week_day2                          0.09795    0.16937   0.578 0.563274    
# week_day3                          0.02108    0.17206   0.123 0.902538    
# week_day4                          0.03999    0.18792   0.213 0.831566    
# week_day5                         -0.30501    0.17112  -1.782 0.075187 .  
# week_day6                          0.09364    0.17847   0.525 0.599994    
# week_day7                         -0.36155    0.18900  -1.913 0.056238 .  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.088 on 595 degrees of freedom
# Multiple R-squared:  0.373,	Adjusted R-squared:  0.2781 
# F-statistic: 3.932 on 90 and 595 DF,  p-value: < 2.2e-16


# Only trip variables

linear_model3 = lm(train$point_security~ +origin +destination +companions +trip_purpose, data = train)
summary(linear_model3)
summaries = getModelMetrics("Trip variables",linear_model3, summaries, train, test);


#Call:
  # lm(formula = train$point_security ~ +origin + destination + companions + 
  #     trip_purpose, data = train)
  
  # Residuals:
  #      Min       1Q   Median       3Q      Max 
  # -2.64649 -1.05939  0.05793  0.78433  2.84390 
  
  # Coefficients: (1 not defined because of singularities)
  #                                    Estimate Std. Error t value Pr(>|t|)    
  # (Intercept)                        2.640083   0.422911   6.243 8.03e-10 ***
  # originAcolman                     -1.122587   1.406023  -0.798  0.42494    
# originAlvaro Obregon              -1.528978   0.989054  -1.546  0.12264    
# originAtenco                      -0.666260   1.400353  -0.476  0.63440    
# originAtizapan de Zaragoza        -1.715412   0.906295  -1.893  0.05886 .  
# originAzcapotzalco                -1.452473   0.909727  -1.597  0.11087    
# originBenito Juarez               -1.056001   0.998062  -1.058  0.29045    
# originChimalhuacï¿½ï¿½n     -0.780358   1.645876  -0.474  0.63558    
# originCoacalco de Berriozal       -1.563134   0.988300  -1.582  0.11425    
# originCoyoacan                    -2.105880   1.246498  -1.689  0.09164 .  
# originCuajimalpa de Morelos       -0.747200   1.528449  -0.489  0.62511    
# originCuauhtemoc                  -1.890462   0.963912  -1.961  0.05030 .  
# originCuauhtlmoc                   0.488137   1.530605   0.319  0.74990    
# originCuautitlan                  -2.031709   1.533796  -1.325  0.18579    
# originCuautitlan Izcalli           0.300633   1.298375   0.232  0.81697    
# originEcatepec de Morelos         -1.695531   0.931492  -1.820  0.06921 .  
# originGustavo A. Madero           -1.362023   0.977434  -1.393  0.16398    
# originIxtapaluca                  -1.529380   1.523665  -1.004  0.31589    
# originIztacalco                   -3.706179   1.266553  -2.926  0.00356 ** 
# originIztapalapa                   0.238311   1.155476   0.206  0.83667    
# originMagdalena Contreras         -0.005112   1.555441  -0.003  0.99738    
# originMiguel Hidalgo              -1.537718   1.008270  -1.525  0.12775    
# originNaucalpan de Juarez         -1.363759   1.012280  -1.347  0.17841    
# originNextlalplan                 -2.851635   1.539693  -1.852  0.06449 .  
# originNezahualcoyotl              -2.233500   1.543253  -1.447  0.14833    
# originNicolas Romero              -1.686004   0.948339  -1.778  0.07592 .  
# originOtro                        -1.206654   1.015882  -1.188  0.23538    
# originTecamec                     -1.286903   0.935724  -1.375  0.16954    
# originTexcoco                     -2.589459   1.263329  -2.050  0.04082 *  
# originTiahuac                     -1.357259   1.401099  -0.969  0.33307    
# originTlalnepantla de Baz         -1.354082   0.914202  -1.481  0.13908    
# originTlalpan                     -1.147916   1.257449  -0.913  0.36166    
# originTultitlan                   -0.853119   1.032342  -0.826  0.40890    
# originVenustiano Carranza         -1.050430   1.268473  -0.828  0.40793    
# originZumpango                    -1.718201   1.589868  -1.081  0.28025    
# destinationAlvaro Obregon          0.970280   1.443401   0.672  0.50170    
# destinationAtenco                  0.006085   1.683355   0.004  0.99712    
# destinationAtizapan de Zaragoza    1.409192   1.131746   1.245  0.21355    
# destinationAzcapotzalco            1.501920   1.113557   1.349  0.17791    
# destinationBenito Juarez           1.094869   1.257730   0.871  0.38436    
# destinationChimalhuacan            0.515071   1.681965   0.306  0.75953    
# destinationCoacalco de Berriozal   0.477741   1.186463   0.403  0.68734    
# destinationCoyoacan                1.220653   1.422242   0.858  0.39108    
# destinationCuajimalpa de Morelos   2.942860   1.304782   2.255  0.02446 *  
# destinationCuauhtemoc              0.641215   1.222402   0.525  0.60008    
# destinationCuautitlan              0.478702   1.434584   0.334  0.73873    
# destinationCuautitlan Izcalli      1.534952   1.665674   0.922  0.35714    
# destinationEcatepec de Morelos     0.671175   1.148941   0.584  0.55932    
# destinationGustavo A. Madero       0.852780   1.238375   0.689  0.49132    
# destinationIztacalco              -0.641847   1.662802  -0.386  0.69963    
# destinationIztapalapa              2.272013   1.674271   1.357  0.17527    
# destinationMiguel Hidalgo          1.828052   1.209234   1.512  0.13111    
# destinationNaucalpan de Juarez     1.316209   1.129418   1.165  0.24431    
# destinationNezahualcoyotl          0.131793   1.273954   0.103  0.91764    
# destinationNicolas Romero          0.824652   1.170329   0.705  0.48131    
# destinationOtro                    1.330747   1.205780   1.104  0.27018    
# destinationTecamec                 0.895493   1.151453   0.778  0.43704    
# destinationTemamatla              -1.576318   1.599260  -0.986  0.32469    
# destinationTeotihuacï¿½ï¿½n        NA         NA      NA       NA    
# destinationTizayuca                2.338272   1.680509   1.391  0.16461    
# destinationTlalnepantla de Baz     1.331403   1.139590   1.168  0.24313    
# destinationTlalpan                 2.800899   1.422449   1.969  0.04939 *  
# destinationTultitlan               0.857705   1.436084   0.597  0.55056    
# destinationVenustiano Carranza     1.015615   1.330763   0.763  0.44565    
# companions1 to 2                  -0.361395   0.569563  -0.635  0.52598    
# companions3 to 4                  -0.419899   0.595327  -0.705  0.48088    
# companionsMas                     -0.583242   0.717135  -0.813  0.41637    
# companionsNone                    -0.301123   0.561214  -0.537  0.59177    
# trip_purposeCompra                 1.108767   0.661638   1.676  0.09429 .  
# trip_purposeEstudio                1.309605   0.668854   1.958  0.05068 .  
# trip_purposeOtro                   1.059320   0.653803   1.620  0.10569    
# trip_purposeRecreacion             1.018299   0.660156   1.543  0.12346    
# trip_purposeTrabajo                0.841500   0.644516   1.306  0.19217    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.225 on 614 degrees of freedom
# Multiple R-squared:  0.1793,	Adjusted R-squared:  0.08438 
# F-statistic: 1.889 on 71 and 614 DF,  p-value: 4.017e-05

# Only perception variables

linear_model4 = lm(train$point_security~ +mode_security +importance_safety +most_safe +least_safe, data = train)
summary(linear_model4)
summaries = getModelMetrics("Perception variables",linear_model4, summaries, train, test);

# Call:
# lm(formula = train$point_security ~ +mode_security + importance_safety + 
#     most_safe + least_safe, data = train)

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -2.54500 -0.94949  0.04758  0.75233  3.05051 

# Coefficients:
#                    Estimate Std. Error t value Pr(>|t|)    
# (Intercept)         2.02445    1.16016   1.745 0.081450 .  
# mode_security2      0.61195    0.16035   3.816 0.000148 ***
# mode_security3      0.97555    0.11856   8.228  1.0e-15 ***
# mode_security4      1.42279    0.13725  10.366  < 2e-16 ***
# mode_security5      1.37518    0.19299   7.126  2.7e-12 ***
# importance_safety1 -1.20002    1.24944  -0.960 0.337180    
# importance_safety2 -1.25096    1.32083  -0.947 0.343931    
# importance_safety3 -0.87664    1.23148  -0.712 0.476799    
# importance_safety4 -0.46762    1.21704  -0.384 0.700934    
# importance_safety5 -0.79388    1.21045  -0.656 0.512141    
# most_safeBRT        0.58868    0.33668   1.748 0.080845 .  
# most_safeMetro      0.56572    0.32755   1.727 0.084611 .  
# most_safePeseros    0.62947    0.36706   1.715 0.086832 .  
# most_safeTaxi       0.59311    0.33943   1.747 0.081040 .  
# most_safeTrolebus   1.08362    0.37255   2.909 0.003751 ** 
# least_safeBRT      -0.07938    0.43080  -0.184 0.853860    
# least_safeMetro     0.27475    0.40853   0.673 0.501474    
# least_safePeseros   0.15320    0.37181   0.412 0.680445    
# least_safeTaxi      0.27024    0.38600   0.700 0.484114    
# least_safeTrolebus  0.26218    0.43287   0.606 0.544937    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.154 on 666 degrees of freedom
# Multiple R-squared:  0.2097,	Adjusted R-squared:  0.1871 
# F-statistic: 9.299 on 19 and 666 DF,  p-value: < 2.2e-16

#Only  instant contextual information
linear_model5 = lm(train$point_security~ +haversine +urban_typology +total_passenger_count
                   +total_female_count +empty_seats +hour +week_day, data = train)
summary(linear_model5);
summaries = getModelMetrics("Instant contextual information",linear_model5, summaries, train, test);

# Call:
# lm(formula = train$point_security ~ +haversine + urban_typology + 
#     total_passenger_count + total_female_count + empty_seats + 
#     hour + week_day, data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.7444 -1.0317  0.1521  0.9661  2.9047 

# Coefficients:
#                         Estimate Std. Error t value Pr(>|t|)    
# (Intercept)            2.1073482  1.2747748   1.653 0.098785 .  
# haversine             -0.0001917  0.0002944  -0.651 0.515015    
# urban_typology1        1.1974795  0.3548987   3.374 0.000784 ***
# urban_typology2        0.4006267  0.2296962   1.744 0.081599 .  
# urban_typology3        0.6549039  0.3735016   1.753 0.079997 .  
# urban_typology4        1.2111943  0.4510928   2.685 0.007436 ** 
# urban_typology5        0.0494331  0.2401763   0.206 0.836995    
# urban_typology6        0.5341091  0.2389423   2.235 0.025732 *  
# urban_typology7        0.1599721  0.3360665   0.476 0.634223    
# urban_typology9        0.2603907  0.2933263   0.888 0.375018    
# total_passenger_count  0.0040604  0.0049804   0.815 0.415202    
# total_female_count     0.0106283  0.0189837   0.560 0.575762    
# empty_seats            0.0028120  0.0047517   0.592 0.554195    
# hour7                  0.8714548  1.7639821   0.494 0.621452    
# hour12                 0.5570743  1.2702622   0.439 0.661131    
# hour13                 0.4964576  1.2577358   0.395 0.693175    
# hour14                 0.4849624  1.2559390   0.386 0.699521    
# hour15                 0.2710430  1.2735499   0.213 0.831530    
# hour16                 0.5430102  1.2613921   0.430 0.666984    
# hour17                 0.3196015  1.2569271   0.254 0.799365    
# hour18                 0.1875036  1.2584401   0.149 0.881602    
# hour19                 0.8144224  1.2796806   0.636 0.524720    
# hour20                 3.1336444  1.7806084   1.760 0.078895 .  
# week_day2              0.0287591  0.1874540   0.153 0.878115    
# week_day3             -0.0556924  0.1942437  -0.287 0.774422    
# week_day4             -0.0591611  0.1999702  -0.296 0.767438    
# week_day5             -0.2464980  0.1904140  -1.295 0.195935    
# week_day6              0.0787109  0.2002074   0.393 0.694339    
# week_day7             -0.3664954  0.2050543  -1.787 0.074348 .  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.229 on 657 degrees of freedom
# Multiple R-squared:  0.116,	Adjusted R-squared:  0.07831 
# F-statistic: 3.079 on 28 and 657 DF,  p-value: 2.74e-07

# Only sociodemographic data
linear_model6 = lm(train$point_security~ +age +gender +education, data = train)
summary(linear_model6)
summaries = getModelMetrics("Sociodemographic data",linear_model6, summaries, train, test);

# Call:
# lm(formula = train$point_security ~ +age + gender + education, 
#     data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.1679 -0.9911  0.1380  1.0963  2.6736 

# Coefficients:
#                               Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                    2.52808    0.36019   7.019 5.48e-12 ***
# age0-17                        0.63980    0.66053   0.969    0.333    
# age18-24                       0.70321    0.64697   1.087    0.277    
# age25-44                       0.70992    0.64728   1.097    0.273    
# age45-64                       0.50461    0.65440   0.771    0.441    
# age65+                         0.54439    0.69437   0.784    0.433    
# gendermale                     0.13244    0.09938   1.333    0.183    
# educationLicenciatura         -0.46671    0.58918  -0.792    0.429    
# educationMaestria y Doctorado -0.83285    0.72253  -1.153    0.249    
# educationPreparatoria         -0.37262    0.58460  -0.637    0.524    
# educationPrimaria             -0.42186    0.61125  -0.690    0.490    
# educationSecundaria           -0.39475    0.58612  -0.673    0.501    
# educationSin estudios         -0.70630    0.76124  -0.928    0.354    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.283 on 673 degrees of freedom
# Multiple R-squared:  0.01225,	Adjusted R-squared:  -0.005363 
# F-statistic: 0.6955 on 12 and 673 DF,  p-value: 0.7567

# Only personal trip information
linear_model7 = lm(train$point_security~ +origin +destination +companions +trip_purpose, data = train)
summary(linear_model7)
summaries = getModelMetrics("Personal trip information",linear_model7, summaries, train, test);

# Call:
# lm(formula = train$point_security ~ +origin + destination + companions + 
#     trip_purpose, data = train)

# Residuals:
#      Min       1Q   Median       3Q      Max 
# -2.64649 -1.05939  0.05793  0.78433  2.84390 

# Coefficients: (1 not defined because of singularities)
#                                    Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                        2.640083   0.422911   6.243 8.03e-10 ***
# originAcolman                     -1.122587   1.406023  -0.798  0.42494    
# originAlvaro Obregon              -1.528978   0.989054  -1.546  0.12264    
# originAtenco                      -0.666260   1.400353  -0.476  0.63440    
# originAtizapan de Zaragoza        -1.715412   0.906295  -1.893  0.05886 .  
# originAzcapotzalco                -1.452473   0.909727  -1.597  0.11087    
# originBenito Juarez               -1.056001   0.998062  -1.058  0.29045    
# originChimalhuacï¿½ï¿½n     -0.780358   1.645876  -0.474  0.63558    
# originCoacalco de Berriozal       -1.563134   0.988300  -1.582  0.11425    
# originCoyoacan                    -2.105880   1.246498  -1.689  0.09164 .  
# originCuajimalpa de Morelos       -0.747200   1.528449  -0.489  0.62511    
# originCuauhtemoc                  -1.890462   0.963912  -1.961  0.05030 .  
# originCuauhtlmoc                   0.488137   1.530605   0.319  0.74990    
# originCuautitlan                  -2.031709   1.533796  -1.325  0.18579    
# originCuautitlan Izcalli           0.300633   1.298375   0.232  0.81697    
# originEcatepec de Morelos         -1.695531   0.931492  -1.820  0.06921 .  
# originGustavo A. Madero           -1.362023   0.977434  -1.393  0.16398    
# originIxtapaluca                  -1.529380   1.523665  -1.004  0.31589    
# originIztacalco                   -3.706179   1.266553  -2.926  0.00356 ** 
# originIztapalapa                   0.238311   1.155476   0.206  0.83667    
# originMagdalena Contreras         -0.005112   1.555441  -0.003  0.99738    
# originMiguel Hidalgo              -1.537718   1.008270  -1.525  0.12775    
# originNaucalpan de Juarez         -1.363759   1.012280  -1.347  0.17841    
# originNextlalplan                 -2.851635   1.539693  -1.852  0.06449 .  
# originNezahualcoyotl              -2.233500   1.543253  -1.447  0.14833    
# originNicolas Romero              -1.686004   0.948339  -1.778  0.07592 .  
# originOtro                        -1.206654   1.015882  -1.188  0.23538    
# originTecamec                     -1.286903   0.935724  -1.375  0.16954    
# originTexcoco                     -2.589459   1.263329  -2.050  0.04082 *  
# originTiahuac                     -1.357259   1.401099  -0.969  0.33307    
# originTlalnepantla de Baz         -1.354082   0.914202  -1.481  0.13908    
# originTlalpan                     -1.147916   1.257449  -0.913  0.36166    
# originTultitlan                   -0.853119   1.032342  -0.826  0.40890    
# originVenustiano Carranza         -1.050430   1.268473  -0.828  0.40793    
# originZumpango                    -1.718201   1.589868  -1.081  0.28025    
# destinationAlvaro Obregon          0.970280   1.443401   0.672  0.50170    
# destinationAtenco                  0.006085   1.683355   0.004  0.99712    
# destinationAtizapan de Zaragoza    1.409192   1.131746   1.245  0.21355    
# destinationAzcapotzalco            1.501920   1.113557   1.349  0.17791    
# destinationBenito Juarez           1.094869   1.257730   0.871  0.38436    
# destinationChimalhuacan            0.515071   1.681965   0.306  0.75953    
# destinationCoacalco de Berriozal   0.477741   1.186463   0.403  0.68734    
# destinationCoyoacan                1.220653   1.422242   0.858  0.39108    
# destinationCuajimalpa de Morelos   2.942860   1.304782   2.255  0.02446 *  
# destinationCuauhtemoc              0.641215   1.222402   0.525  0.60008    
# destinationCuautitlan              0.478702   1.434584   0.334  0.73873    
# destinationCuautitlan Izcalli      1.534952   1.665674   0.922  0.35714    
# destinationEcatepec de Morelos     0.671175   1.148941   0.584  0.55932    
# destinationGustavo A. Madero       0.852780   1.238375   0.689  0.49132    
# destinationIztacalco              -0.641847   1.662802  -0.386  0.69963    
# destinationIztapalapa              2.272013   1.674271   1.357  0.17527    
# destinationMiguel Hidalgo          1.828052   1.209234   1.512  0.13111    
# destinationNaucalpan de Juarez     1.316209   1.129418   1.165  0.24431    
# destinationNezahualcoyotl          0.131793   1.273954   0.103  0.91764    
# destinationNicolas Romero          0.824652   1.170329   0.705  0.48131    
# destinationOtro                    1.330747   1.205780   1.104  0.27018    
# destinationTecamec                 0.895493   1.151453   0.778  0.43704    
# destinationTemamatla              -1.576318   1.599260  -0.986  0.32469    
# destinationTeotihuacï¿½ï¿½n        NA         NA      NA       NA    
# destinationTizayuca                2.338272   1.680509   1.391  0.16461    
# destinationTlalnepantla de Baz     1.331403   1.139590   1.168  0.24313    
# destinationTlalpan                 2.800899   1.422449   1.969  0.04939 *  
# destinationTultitlan               0.857705   1.436084   0.597  0.55056    
# destinationVenustiano Carranza     1.015615   1.330763   0.763  0.44565    
# companions1 to 2                  -0.361395   0.569563  -0.635  0.52598    
# companions3 to 4                  -0.419899   0.595327  -0.705  0.48088    
# companionsMas                     -0.583242   0.717135  -0.813  0.41637    
# companionsNone                    -0.301123   0.561214  -0.537  0.59177    
# trip_purposeCompra                 1.108767   0.661638   1.676  0.09429 .  
# trip_purposeEstudio                1.309605   0.668854   1.958  0.05068 .  
# trip_purposeOtro                   1.059320   0.653803   1.620  0.10569    
# trip_purposeRecreacion             1.018299   0.660156   1.543  0.12346    
# trip_purposeTrabajo                0.841500   0.644516   1.306  0.19217    
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.225 on 614 degrees of freedom
# Multiple R-squared:  0.1793,	Adjusted R-squared:  0.08438 
# F-statistic: 1.889 on 71 and 614 DF,  p-value: 4.017e-05




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