install.packages("NISTunits", dependencies = TRUE);
install.packages("GGally", dependencies = TRUE);
install.packages("ggplot2", dependencies = TRUE);
install.packages("caTools", dependencies = TRUE);
install.packages("magrittr")


library(NISTunits);
# Graphing library
 library(GGally);
 library(ggplot2);
# For parsing dates
library(lubridate);

# For sample splitting
library("caTools");

# For removing unused levels in test data
library(magrittr);

# loading file
safety_data = read.csv("safety_data.csv");

# Removing rows with no safety perception measurement
completeFun <- function(data, desiredCols) {
  completeVec <- complete.cases(data[, desiredCols])
  return(data[completeVec, ])
}

safety_data = completeFun(safety_data, "pointsecurity")

# Based on
# https://stackoverflow.com/a/365853/3128369
haversine <- function(data, lat1, lon1, lat2, lon2){
  earthRadiusKm = 6371;
  
  dLat = NISTdegTOradian(data[[lat2]]-data[[lat1]]);
  dLon = NISTdegTOradian(data[[lon2]]-data[[lon1]]);
  
  lat1 = NISTdegTOradian(data[[lat1]]);
  lat2 = NISTdegTOradian(data[[lat2]]);
  
  a = sin(dLat/2) * sin(dLat/2) +
    sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2); 
  c = 2 * atan2(sqrt(a), sqrt(1-a)); 
  distance = earthRadiusKm * c;
  return (distance);
}

safety_data[["haversine"]] = haversine(safety_data, "cetram_lat", "cetram_long", "latitude", "longitude")


# Get day of the week
times = strptime(safety_data$date, "%m/%d/%Y");
safety_data$wday = wday(as.Date(times), label=FALSE);

# Making a basic plot of some potentially relevant variables

plot_data <- data.frame(
  # Type of survey
  inside_or_outside = safety_data[["inside_or_outside"]],
  
  # Sociodemographic data
  gender = safety_data[["gender"]],
  age = safety_data[["age"]],
  education = safety_data[["educational_attainment"]],
  
  # Personal trip related data
  origin = safety_data[["origin"]],
  destination = safety_data[["destinations"]],
  companions = safety_data[["companions"]],  
  trip_purpose = safety_data[["trip_purpose"]],

  # Perception data
  mode_security = safety_data[["modesecurity"]],
  point_security = safety_data[["pointsecurity"]],
  importance_safety = safety_data[["Importance_safety_digit"]],
  most_safe = safety_data[["mostsafe"]],
  least_safe = safety_data[["leastsafe"]],
  
  # Context data
  bus_or_ped = safety_data[["bus_or_ped"]],
  base_study_zone = safety_data[["base_study_zone"]],
  busdestination = safety_data[["busdestination"]],  
  total_seats = safety_data[["totalseats"]],
  

  # Time related contextual information
  haversine = safety_data[["haversine"]],
  urban_typology = safety_data[["urban.typology"]],
  total_passenger_count = safety_data[["totalpassengercount"]],
  total_female_count = safety_data[["totalfemalecount"]],
  empty_seats = safety_data[["emptyseats"]],
  hour = safety_data[["hour"]],
  week_day = safety_data[["wday"]]
  
  
);

# Treating all the variables as categorical
for(i in names(plot_data)){
  plot_data[[i]] <- as.factor(plot_data[[i]])
}

# Some fields have to be treated as numeric.
# TODO improve this so we don't convert the data twice.
plot_data[["total_female_count"]] = as.numeric(as.character(plot_data[["total_female_count"]]));
plot_data[["total_passenger_count"]] = as.numeric(as.character(plot_data[["total_passenger_count"]]));
plot_data[["empty_seats"]] = as.numeric(as.character(plot_data[["empty_seats"]]));
plot_data[["point_security"]] = as.numeric(as.character(plot_data[["point_security"]]));
plot_data[["haversine"]] = as.numeric(plot_data[["haversine"]]);

#Removing incomplete cases
plot_data = na.omit(plot_data)


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
test = na.omit(remove_missing_levels (fit=linear_model, test_data=test));


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

pred_data = predict(linear_model, newdata = test);

# Data frame for summaries
summaries <- data.frame(matrix(ncol = 9, nrow = 0))
names <- c("model", "r2", "sse", "pred_means", "sst", "osr2", "rmse", "mae", "var_num")
colnames(summaries) <- names

SSE = sum((pred_data - test$point_security)^2);
pred_mean = mean(train$point_security);
SST = sum((pred_mean - test$point_security)^2);
OSR2 = 1-SSE/SST;
RMSE = sqrt(sum((pred_data - test$point_security)^2)/nrow(test));
MAE = sum(abs(pred_data - test$point_security))/nrow(test);

current_model = linear_model;
i = 1;
summaries[i, "model"] = "Initial model";
summaries[i, "r2"] = summary(current_model)$r.squared
summaries[i, "sse"] = SSE;
summaries[i, "pred_means"] = pred_mean;
summaries[i, "sst"] = SST;
summaries[i, "osr2"] = OSR2;
summaries[i, "rmse"] = RMSE;
summaries[i, "mae"] = MAE;
summaries[i,"var_num"] = length(names(current_model$coefficients));


# Taking out the least relevant variables
linear_model2 = lm(train$point_security~. 
                   -bus_or_ped -inside_or_outside -gender -age -education 
                   -companions - trip_purpose -importance_safety -least_safe
                   -total_passenger_count -total_female_count -empty_seats
                   -base_study_zone -total_seats -haversine
                   -hour, data = train)
summary(linear_model2);

pred_data = predict(linear_model2, newdata = test);

SSE = sum((pred_data - test$point_security)^2);
pred_mean = mean(train$point_security);
SST = sum((pred_mean - test$point_security)^2);
OSR2 = 1-SSE/SST;
RMSE = sqrt(sum((pred_data - test$point_security)^2)/nrow(test));
MAE = sum(abs(pred_data - test$point_security))/nrow(test);

current_model = linear_model2;
#i = i+1;
summaries[i, "model"] = "Only significant variables";
summaries[i, "r2"] = summary(current_model)$r.squared
summaries[i, "sse"] = SSE;
summaries[i, "pred_means"] = pred_mean;
summaries[i, "sst"] = SST;
summaries[i, "osr2"] = OSR2;
summaries[i, "rmse"] = RMSE;
summaries[i, "mae"] = MAE;
summaries[i,"var_num"] = length(names(current_model$coefficients));






linear_model3 = lm(train$point_security~ +gender +age, data = train)
summary(linear_model3)



library(rpart)
library(rpart.plot)
CART = rpart(train$point_security~.,
             data = train, 
             method="class")

rpart.plot(CART)







