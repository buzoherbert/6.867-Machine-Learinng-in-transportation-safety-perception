install.packages("NISTunits", dependencies = TRUE);
install.packages("GGally", dependencies = TRUE);
install.packages("ggplot2", dependencies = TRUE);
install.packages("caTools", dependencies = TRUE);

library(NISTunits);
# Graphing library
 library(GGally);
 library(ggplot2);
# For parsing dates
library(lubridate);

# For sample splitting
library("caTools");

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
  bus_or_ped = safety_data[["bus_or_ped"]],
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
to_plot = plot_data;
to_plot$point_security = as.factor(to_plot$point_security)
ggpairs(to_plot, mapping = aes(color = point_security))

########################
## Creating train and testing sets
## 70% of the sample size
smp_size <- floor(0.7 * nrow(plot_data))
## set the seed to make your partition reproductible
set.seed(888)
train_ind <- sample(seq_len(nrow(plot_data)), size = smp_size)

train <- plot_data[train_ind, ]
test <- plot_data[-train_ind, ]



########################
## Linear regression model


linear_model = lm(train$point_security~., data = train)
summary(linear_model)

# Call:
# lm(formula = train$point_security ~ ., data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -2.8728 -0.6383  0.0000  0.5877  3.1479 

# Coefficients: (5 not defined because of singularities)
#                                     Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                        2.8270222  1.5094371   1.873 0.061618 .  
# bus_or_pedVan                      0.0203469  0.1911199   0.106 0.915255    
# base_study_zoneEl Rosario         -0.0948897  0.2999468  -0.316 0.751854    
# busdestinationHeroes Tecamec       0.1916852  0.1895445   1.011 0.312325    
# busdestinationMexico Nueva         0.1547675  0.1593616   0.971 0.331893    
# busdestinationMexipuerto                  NA         NA      NA       NA    
# busdestinationTacuba              -0.3262205  0.4838981  -0.674 0.500501    
# busdestinationTexcoco             -2.3717381  2.3089908  -1.027 0.304794    
# inside_or_outsideFuera CETRAM     -0.1506041  0.1245169  -1.210 0.226992    
# total_seats35                             NA         NA      NA       NA    
# total_passenger_count             -0.0030489  0.0060517  -0.504 0.614594    
# total_female_count                -0.0062752  0.0193802  -0.324 0.746217    
# empty_seats                       -0.0027631  0.0057702  -0.479 0.632239    
# gendermale                         0.0749514  0.0949846   0.789 0.430402    
# age0-17                            1.3901689  0.9887764   1.406 0.160309    
# age18-24                           1.1571326  0.9807743   1.180 0.238588    
# age25-44                           1.2619706  0.9838321   1.283 0.200139    
# age45-64                           1.1057430  0.9860728   1.121 0.262628    
# age65+                             1.2092174  1.0188522   1.187 0.235807    
# companions1 to 2                  -0.8864589  0.9996069  -0.887 0.375574    
# companions3 to 4                  -0.8858910  1.0140653  -0.874 0.382719    
# companionsMas                     -1.3222728  1.0904253  -1.213 0.225800    
# companionsNone                    -0.9950413  0.9961424  -0.999 0.318289    
# educationLicenciatura             -0.0971708  0.7484087  -0.130 0.896744    
# educationMaestria y Doctorado     -0.4528590  0.9147449  -0.495 0.620753    
# educationPreparatoria             -0.0343374  0.7443504  -0.046 0.963223    
# educationPrimaria                 -0.3261120  0.7648786  -0.426 0.670015    
# educationSecundaria               -0.1009344  0.7443174  -0.136 0.892182    
# educationSin estudios             -1.1998907  0.9336109  -1.285 0.199263    
# originAcolman                      0.0141938  1.2841933   0.011 0.991185    
# originAlvaro Obregon              -0.6290032  0.6671195  -0.943 0.346168    
# originAtenco                      -0.4458452  1.2682511  -0.352 0.725317    
# originAtizapan de Zaragoza        -1.1052231  0.5573583  -1.983 0.047873 *  
# originAzcapotzalco                -0.6797609  0.5616550  -1.210 0.226695    
# originBenito Juarez               -0.4729937  0.6636233  -0.713 0.476309    
# originChalco                      -0.0281019  0.9987295  -0.028 0.977563    
# originChimalhuacan                -2.3887951  1.7309862  -1.380 0.168146    
# originCoacalco de Berriozal       -1.3068063  0.6967252  -1.876 0.061239 .  
# originCoyoacan                     0.1768242  0.9656788   0.183 0.854781    
# originCuauhtemoc                  -0.6309162  0.6286032  -1.004 0.315979    
# originCuautitlan                  -0.7170443  1.2323609  -0.582 0.560911    
# originCuautitlan Izcalli           0.1845736  0.8174805   0.226 0.821454    
# originEcatepec de Morelos         -0.9343164  0.5987949  -1.560 0.119263    
# originGustavo A. Madero           -0.4567990  0.6706215  -0.681 0.496061    
# originIxtapaluca                  -0.0729796  1.2446229  -0.059 0.953264    
# originIztacalco                   -1.8652136  1.2425183  -1.501 0.133894    
# originIztapalapa                   0.0138619  0.9627215   0.014 0.988517    
# originMagdalena Contreras          1.1970321  1.2782541   0.936 0.349452    
# originMiguel Hidalgo              -1.1911508  0.7053784  -1.689 0.091855 .  
# originNaucalpan de Juarez         -0.6554726  0.6990620  -0.938 0.348842    
# originNezahualcoyotl              -1.5043727  0.9954224  -1.511 0.131294    
# originNicolas Romero              -0.4631798  0.6151862  -0.753 0.451829    
# originOtro                        -0.6399965  0.6769480  -0.945 0.344866    
# originTecamec                     -0.6769102  0.6119796  -1.106 0.269172    
# originTiahuac                     -0.7443927  1.3113722  -0.568 0.570510    
# originTlalnepantla de Baz         -0.7042484  0.5691244  -1.237 0.216462    
# originTultitlan                   -0.3659034  0.6798090  -0.538 0.590628    
# originVenustiano Carranza         -0.6740469  0.7283263  -0.925 0.355129    
# originZumpango                    -0.7427148  1.3318607  -0.558 0.577311    
# destinationAlvaro Obregon          0.2871001  1.0333146   0.278 0.781238    
# destinationAtenco                  1.1807967  1.3620819   0.867 0.386375    
# destinationAtizapan de Zaragoza    0.7482239  0.7975178   0.938 0.348562    
# destinationAzcapotzalco            0.9774307  0.7837257   1.247 0.212875    
# destinationBenito Juarez          -0.0891360  0.9857291  -0.090 0.927982    
# destinationChalco                  0.6016299  1.1339690   0.531 0.595945    
# destinationCoacalco de Berriozal   0.4697864  0.8850459   0.531 0.595770    
# destinationCocotitlan              2.8465613  1.4534011   1.959 0.050675 .  
# destinationCoyoacan                0.6992502  1.3996394   0.500 0.617563    
# destinationCuajimalpa de Morelos   1.3059790  1.1381767   1.147 0.251707    
# destinationCuauhtemoc              1.1542596  0.8523698   1.354 0.176241    
# destinationCuautitlan              1.4090963  1.3549718   1.040 0.298827    
# destinationEcatepec de Morelos     0.2048338  0.8163125   0.251 0.801966    
# destinationGustavo A. Madero       0.7335057  0.9490030   0.773 0.439903    
# destinationIztacalco               0.5342542  1.4318282   0.373 0.709199    
# destinationIztapalapa              1.0121655  1.3584134   0.745 0.456527    
# destinationLa Paz                  1.3657466  1.3405322   1.019 0.308746    
# destinationMiguel Hidalgo          1.8439239  0.9294072   1.984 0.047759 *  
# destinationNaucalpan de Juarez     1.2040818  0.8301059   1.451 0.147490    
# destinationNezahualcoyotl          0.2435004  0.9597564   0.254 0.799815    
# destinationNicolas Romero         -0.1148202  0.8488608  -0.135 0.892453    
# destinationOtro                    0.1449920  0.8740487   0.166 0.868309    
# destinationTecamec                 0.4414890  0.8286823   0.533 0.594417    
# destinationTemamatla              -1.1334605  1.3901368  -0.815 0.415223    
# destinationTeotihuacï¿½ï¿½n         NA         NA      NA       NA    
# destinationTiahuac                 1.5251584  1.3849516   1.101 0.271280    
# destinationTizayuca                1.7846016  1.4017095   1.273 0.203504    
# destinationTlalnepantla de Baz     0.7933499  0.8075591   0.982 0.326336    
# destinationTlalpan                 1.9400992  1.1223210   1.729 0.084438 .  
# destinationTultitlan               0.1816278  0.9864882   0.184 0.853991    
# destinationVenustiano Carranza     0.9106799  1.1077623   0.822 0.411385    
# trip_purposeCompra                 1.1674448  0.7090505   1.646 0.100239    
# trip_purposeEstudio                1.3121652  0.7152596   1.835 0.067120 .  
# trip_purposeOtro                   0.9118282  0.6981411   1.306 0.192076    
# trip_purposeRecreacion             1.1095484  0.7051367   1.574 0.116178    
# trip_purposeTrabajo                0.9821229  0.6930514   1.417 0.157025    
# mode_security2                     0.4155943  0.1647442   2.523 0.011931 *  
# mode_security3                     0.8871575  0.1265899   7.008 7.16e-12 ***
# mode_security4                     1.1192759  0.1489572   7.514 2.37e-13 ***
# mode_security5                     1.3153411  0.2004026   6.563 1.23e-10 ***
# importance_safety1                -1.2664514  0.9395221  -1.348 0.178227    
# importance_safety2                -0.5607715  0.9703512  -0.578 0.563566    
# importance_safety3                -1.0762494  0.8738148  -1.232 0.218604    
# importance_safety4                -0.6288037  0.8633587  -0.728 0.466729    
# importance_safety5                -0.8743319  0.8455944  -1.034 0.301602    
# importance_safetyI                 0.7450754  1.4125529   0.527 0.598084    
# most_safeBRT                       0.5187247  0.3457954   1.500 0.134170    
# most_safeMetro                     0.5516626  0.3368730   1.638 0.102083    
# most_safePeseros                   0.7552033  0.3677478   2.054 0.040492 *  
# most_safeTaxi                      0.6438357  0.3470424   1.855 0.064106 .  
# most_safeTrolebus                  0.5001989  0.3815666   1.311 0.190441    
# least_safeBRT                     -0.4644479  0.4590253  -1.012 0.312076    
# least_safeMetro                   -0.1590200  0.4187432  -0.380 0.704275    
# least_safePeseros                 -0.3420166  0.3880629  -0.881 0.378521    
# least_safeTaxi                    -0.2252404  0.4047586  -0.556 0.578111    
# least_safeTrolebus                 0.1114304  0.4700020   0.237 0.812680    
# urban_typology1                    1.0580826  0.3514787   3.010 0.002730 ** 
# urban_typology2                    0.7984412  0.2279846   3.502 0.000499 ***
# urban_typology3                    1.2397867  0.3707505   3.344 0.000883 ***
# urban_typology4                    0.8670396  0.4116192   2.106 0.035625 *  
# urban_typology5                    0.4402933  0.2350968   1.873 0.061629 .  
# urban_typology6                    0.6173886  0.2350433   2.627 0.008864 ** 
# urban_typology7                    0.4909565  0.3346442   1.467 0.142926    
# urban_typology9                    0.6743029  0.2764449   2.439 0.015038 *  
# haversine                          0.0002490  0.0003381   0.736 0.461754    
# hour7                                     NA         NA      NA       NA    
# hour12                            -2.0383960  1.1455655  -1.779 0.075735 .  
# hour13                            -2.1483300  1.1311235  -1.899 0.058054 .  
# hour14                            -2.0336801  1.1333679  -1.794 0.073308 .  
# hour15                            -2.2374531  1.1468825  -1.951 0.051581 .  
# hour16                            -1.7982945  1.1370477  -1.582 0.114333    
# hour17                            -1.8981804  1.1326711  -1.676 0.094342 .  
# hour18                            -1.8868159  1.1326130  -1.666 0.096309 .  
# hour19                            -1.2426596  1.1530360  -1.078 0.281632    
# hour20                                    NA         NA      NA       NA    
# week_day2                          0.2640807  0.1901834   1.389 0.165534    
# week_day3                          0.1980619  0.1890271   1.048 0.295197    
# week_day4                          0.3877325  0.2047587   1.894 0.058807 .  
# week_day5                          0.0887067  0.1952455   0.454 0.649769    
# week_day6                          0.5308139  0.1953382   2.717 0.006789 ** 
# week_day7                         -0.3543144  0.1947240  -1.820 0.069373 .  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.063 on 545 degrees of freedom
#   (340 observations deleted due to missingness)
# Multiple R-squared:  0.4347,	Adjusted R-squared:  0.2957 
# F-statistic: 3.127 on 134 and 545 DF,  p-value: < 2.2e-16

pred_data = predict(linear_model, newdata = test)
SSE = sum((pred_data - test$point_security)^2)
pred_mean = mean(train$point_security)
SST = sum((pred_mean - test$point_security)^2)
OSR2 = 1-SSE/SST
RMSE = sqrt(sum((pred_data - test$point_security)^2)/nrow(test))
MAE = sum(abs(pred_data - test$point_security))/nrow(test)





# Taking out the least relevant variables
linear_model2 = lm(train$point_security~. 
                   -bus_or_ped -busdestination -inside_or_outside -total_seats -total_female_count 
                   -empty_seats -companions -education -origin -destination -hour, data = train)
summary(linear_model2)

pred_data2 = predict(linear_model2, newdata = test)
SSE2 = sum((pred_data2 - test$point_security)^2)
pred_mean2 = mean(train$point_security)
SST2 = sum((pred_mean2 - test$point_security)^2)
OSR22 = 1-SSE2/SST2
RMSE2 = sqrt(sum((pred_data2 - test$point_security)^2)/nrow(test))
MAE2 = sum(abs(pred_data2 - test$point_security))/nrow(test)




linear_model3 = lm(train$point_security~ +gender +age, data = train)
summary(linear_model3)



library(rpart)
library(rpart.plot)
CART = rpart(train$point_security~.,
             data = train, 
             method="class")

rpart.plot(CART)







