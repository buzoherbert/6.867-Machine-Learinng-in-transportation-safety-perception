install.packages("NISTunits", dependencies = TRUE);
install.packages("GGally", dependencies = TRUE);
install.packages("ggplot2", dependencies = TRUE);


library(NISTunits);
# Graphing library
 library(GGally);
 library(ggplot2);
# For parsing dates
library(lubridate);

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
  bus_or_ped = safety_data[["bus_or_ped"]],
  base_study_zone = safety_data[["base_study_zone"]],
  busdestination = safety_data[["busdestination"]],
  inside_or_outside = safety_data[["inside_or_outside"]],
  total_seats = safety_data[["totalseats"]],
  total_passenger_count = safety_data[["totalpassengercount"]],
  total_female_count = safety_data[["totalfemalecount"]],
  empty_seats = safety_data[["emptyseats"]],
  gender = safety_data[["gender"]],
  age = safety_data[["age"]],
  companions = safety_data[["companions"]],  
  education = safety_data[["educational_attainment"]],
  origin = safety_data[["origin"]],
  destination = safety_data[["destinations"]],
  trip_purpose = safety_data[["trip_purpose"]],
  mode_security = safety_data[["modesecurity"]],
  point_security = safety_data[["pointsecurity"]],
  importance_safety = safety_data[["Importance_safety_digit"]],
  most_safe = safety_data[["mostsafe"]],
  least_safe = safety_data[["leastsafe"]],
  urban_typology = safety_data[["urban.typology"]],
  haversine = safety_data[["haversine"]],
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
plot_data[["mode_security"]] = as.numeric(as.character(plot_data[["mode_security"]]));
plot_data[["point_security"]] = as.numeric(as.character(plot_data[["point_security"]]));
plot_data[["haversine"]] = as.numeric(plot_data[["haversine"]]);


# Getting a summary of the data
# summary(plot_data)

# plotting the data
to_plot = plot_data;
to_plot$point_security = as.factor(to_plot$point_security)
ggpairs(to_plot, mapping = aes(color = point_security))


########################
## Linear regression model

set.seed(888)
smp_size <- floor(0.75*nrow(plot_data))
train_data <- sample(seq_len(nrow(plot_data)), size = smp_size)
train <- plot_data[train_data,]
test <- plot_data[-train_data,]
linear_model = lm(train$point_security~., data = train)
summary(linear_model)

# Call:
# lm(formula = train$point_security ~ ., data = train)

# Residuals:
#     Min      1Q  Median      3Q     Max 
# -3.3291 -0.6796  0.0129  0.5853  3.0867 

# Coefficients: (5 not defined because of singularities)
#                                     Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                        3.0179390  1.5217115   1.983 0.047802 *  
# bus_or_pedVan                     -0.0133388  0.1875629  -0.071 0.943329    
# base_study_zoneEl Rosario         -0.0708567  0.2939966  -0.241 0.809630    
# busdestinationHeroes Tecamec       0.1895869  0.1853732   1.023 0.306854    
# busdestinationMexico Nueva         0.1563916  0.1576934   0.992 0.321729    
# busdestinationMexipuerto                  NA         NA      NA       NA    
# busdestinationTacuba              -0.4396766  0.4653813  -0.945 0.345165    
# busdestinationTexcoco             -2.6836236  2.3391824  -1.147 0.251744    
# inside_or_outsideFuera CETRAM     -0.1571131  0.1217916  -1.290 0.197550    
# total_seats35                             NA         NA      NA       NA    
# total_passenger_count             -0.0027199  0.0059202  -0.459 0.646095    
# total_female_count                -0.0038300  0.0192593  -0.199 0.842438    
# empty_seats                       -0.0021760  0.0056115  -0.388 0.698324    
# gendermale                         0.0726585  0.0928939   0.782 0.434431    
# age0-17                            1.4047617  0.9919528   1.416 0.157257    
# age18-24                           1.0933758  0.9833075   1.112 0.266618    
# age25-44                           1.1939652  0.9869277   1.210 0.226848    
# age45-64                           1.0788406  0.9902511   1.089 0.276395    
# age65+                             1.4064337  1.0178677   1.382 0.167573    
# companions1 to 2                  -1.1721102  1.0015249  -1.170 0.242342    
# companions3 to 4                  -1.1664947  1.0175654  -1.146 0.252112    
# companionsMas                     -1.7057590  1.0911429  -1.563 0.118524    
# companionsNone                    -1.2589085  0.9992607  -1.260 0.208225    
# educationLicenciatura              0.0244719  0.7567346   0.032 0.974213    
# educationMaestria y Doctorado     -0.2938035  0.9257414  -0.317 0.751074    
# educationPreparatoria              0.0330509  0.7533628   0.044 0.965022    
# educationPrimaria                 -0.2391135  0.7740715  -0.309 0.757504    
# educationSecundaria                0.0063339  0.7537028   0.008 0.993298    
# educationSin estudios             -0.6428793  0.9103870  -0.706 0.480367    
# originAcolman                     -0.0048685  1.2849884  -0.004 0.996978    
# originAlvaro Obregon              -0.5182202  0.6438881  -0.805 0.421242    
# originAtenco                      -0.2642813  1.2709868  -0.208 0.835352    
# originAtizapan de Zaragoza        -1.0258182  0.5380763  -1.906 0.057077 .  
# originAzcapotzalco                -0.6203664  0.5380659  -1.153 0.249395    
# originBenito Juarez               -0.4221141  0.6402385  -0.659 0.509955    
# originChalco                      -0.0834854  0.9964879  -0.084 0.933260    
# originChimalhuacan                -2.4481159  1.7424732  -1.405 0.160558    
# originCoacalco de Berriozal       -1.3097165  0.6708788  -1.952 0.051382 .  
# originCoyoacan                    -0.7420867  0.8415439  -0.882 0.378236    
# originCuajimalpa de Morelos       -0.3812473  1.2496599  -0.305 0.760412    
# originCuauhtemoc                  -0.6405889  0.6095128  -1.051 0.293696    
# originCuautitlan                  -0.4898721  1.2402017  -0.395 0.692990    
# originCuautitlan Izcalli           0.2897810  0.8239481   0.352 0.725190    
# originEcatepec de Morelos         -0.9444549  0.5796023  -1.629 0.103744    
# originGustavo A. Madero           -0.4559519  0.6576759  -0.693 0.488408    
# originIxtapaluca                  -0.1179246  1.2478506  -0.095 0.924742    
# originIztacalco                   -1.6532290  0.9716752  -1.701 0.089391 .  
# originIztapalapa                  -0.0333831  0.8492142  -0.039 0.968656    
# originMagdalena Contreras          1.0772520  1.2763924   0.844 0.399022    
# originMiguel Hidalgo              -1.1570732  0.6955162  -1.664 0.096720 .  
# originNaucalpan de Juarez         -0.5897198  0.6651488  -0.887 0.375656    
# originNezahualcoyotl              -1.4284584  0.9932829  -1.438 0.150931    
# originNicolas Romero              -0.4035775  0.5942443  -0.679 0.497313    
# originOtro                        -0.5163290  0.6681919  -0.773 0.439994    
# originTecamec                     -0.6259896  0.5919813  -1.057 0.290740    
# originTexcoco                      0.1055453  1.2391242   0.085 0.932149    
# originTiahuac                     -0.7284474  1.2927384  -0.563 0.573314    
# originTlalnepantla de Baz         -0.6923058  0.5475658  -1.264 0.206610    
# originTultitlan                   -0.2852110  0.6671216  -0.428 0.669153    
# originVenustiano Carranza         -0.6330179  0.7189292  -0.881 0.378946    
# originZumpango                    -0.5813706  1.3374378  -0.435 0.663947    
# destinationAlvaro Obregon          0.0593474  0.9233019   0.064 0.948771    
# destinationAtenco                  0.8747343  1.2991064   0.673 0.500998    
# destinationAtizapan de Zaragoza    0.5122812  0.6553534   0.782 0.434712    
# destinationAzcapotzalco            0.7313662  0.6417843   1.140 0.254923    
# destinationBenito Juarez          -0.5359754  0.8388409  -0.639 0.523105    
# destinationChalco                  0.4145797  1.0504292   0.395 0.693224    
# destinationCoacalco de Berriozal   0.2809888  0.7526344   0.373 0.709029    
# destinationCocotitlan              2.7095934  1.3908998   1.948 0.051878 .  
# destinationCoyoacan                0.4248962  1.3317813   0.319 0.749806    
# destinationCuajimalpa de Morelos   1.0268034  1.0559972   0.972 0.331273    
# destinationCuauhtemoc              0.7062950  0.7238330   0.976 0.329578    
# destinationCuautitlan              0.0228971  1.0540270   0.022 0.982676    
# destinationCuautitlan Izcalli      0.0491914  1.5515885   0.032 0.974719    
# destinationEcatepec de Morelos     0.1021042  0.6884403   0.148 0.882147    
# destinationGustavo A. Madero       0.6055291  0.8320489   0.728 0.467051    
# destinationIztacalco               0.3226236  1.3640631   0.237 0.813114    
# destinationIztapalapa              0.7994201  1.2903702   0.620 0.535808    
# destinationLa Paz                  1.2539162  1.3205856   0.950 0.342747    
# destinationMiguel Hidalgo          1.5969839  0.8164659   1.956 0.050940 .  
# destinationNaucalpan de Juarez     0.9941446  0.7083231   1.404 0.160988    
# destinationNezahualcoyotl         -0.1338343  0.8364946  -0.160 0.872940    
# destinationNicolas Romero         -0.2408478  0.7172823  -0.336 0.737157    
# destinationOtro                   -0.0533236  0.7488519  -0.071 0.943257    
# destinationTecamec                 0.3930347  0.6970102   0.564 0.573046    
# destinationTemamatla              -1.3060961  1.2526627  -1.043 0.297535    
# destinationTeotihuacï¿½ï¿½n         NA         NA      NA       NA    
# destinationTiahuac                 1.2838818  1.3730582   0.935 0.350144    
# destinationTizayuca                1.8303790  1.3404127   1.366 0.172606    
# destinationTlalnepantla de Baz     0.5536975  0.6669462   0.830 0.406763    
# destinationTlalpan                 1.7847695  1.0287004   1.735 0.083267 .  
# destinationTultitlan               0.0157175  0.8901712   0.018 0.985919    
# destinationVenustiano Carranza     0.7398667  1.0177147   0.727 0.467521    
# trip_purposeCompra                 1.3743206  0.6861821   2.003 0.045650 *  
# trip_purposeEstudio                1.3991514  0.6968835   2.008 0.045128 *  
# trip_purposeOtro                   1.1432984  0.6799045   1.682 0.093184 .  
# trip_purposeRecreacion             1.2471757  0.6874261   1.814 0.070144 .  
# trip_purposeTrabajo                1.1984603  0.6744026   1.777 0.076072 .  
# mode_security2                     0.4893023  0.1643158   2.978 0.003022 ** 
# mode_security3                     0.9402247  0.1239384   7.586 1.29e-13 ***
# mode_security4                     1.2025236  0.1458877   8.243 1.09e-15 ***
# mode_security5                     1.3732395  0.1947878   7.050 5.02e-12 ***
# importance_safety1                -1.0106721  0.9361415  -1.080 0.280755    
# importance_safety2                -0.4135349  0.9781806  -0.423 0.672625    
# importance_safety3                -0.9403336  0.8793533  -1.069 0.285351    
# importance_safety4                -0.4713430  0.8680177  -0.543 0.587327    
# importance_safety5                -0.7472508  0.8515996  -0.877 0.380590    
# importance_safetyI                 1.0504730  1.4274261   0.736 0.462071    
# most_safeBRT                       0.5832244  0.3467152   1.682 0.093070 .  
# most_safeMetro                     0.6366845  0.3376091   1.886 0.059804 .  
# most_safePeseros                   0.6423003  0.3661851   1.754 0.079944 .  
# most_safeTaxi                      0.6465148  0.3479419   1.858 0.063651 .  
# most_safeTrolebus                  0.6331444  0.3785764   1.672 0.094969 .  
# least_safeBRT                     -0.4376768  0.4554270  -0.961 0.336933    
# least_safeMetro                   -0.1838586  0.4204389  -0.437 0.662053    
# least_safePeseros                 -0.3812796  0.3887827  -0.981 0.327142    
# least_safeTaxi                    -0.2895874  0.4023483  -0.720 0.471968    
# least_safeTrolebus                 0.1489700  0.4729141   0.315 0.752870    
# urban_typology1                    1.0537196  0.3451731   3.053 0.002370 ** 
# urban_typology2                    0.7405528  0.2231685   3.318 0.000961 ***
# urban_typology3                    1.1195556  0.3584736   3.123 0.001877 ** 
# urban_typology4                    0.8853258  0.4122757   2.147 0.032167 *  
# urban_typology5                    0.4649577  0.2297780   2.024 0.043471 *  
# urban_typology6                    0.6135152  0.2308674   2.657 0.008087 ** 
# urban_typology7                    0.4896484  0.3303619   1.482 0.138832    
# urban_typology9                    0.5687679  0.2717607   2.093 0.036785 *  
# haversine                          0.0002650  0.0003317   0.799 0.424670    
# hour7                                     NA         NA      NA       NA    
# hour12                            -2.0507126  1.1605379  -1.767 0.077739 .  
# hour13                            -2.2634978  1.1463088  -1.975 0.048780 *  
# hour14                            -2.1727674  1.1478819  -1.893 0.058867 .  
# hour15                            -2.3870654  1.1613800  -2.055 0.040283 *  
# hour16                            -2.0016847  1.1511847  -1.739 0.082591 .  
# hour17                            -2.0483836  1.1474125  -1.785 0.074739 .  
# hour18                            -2.0665333  1.1473254  -1.801 0.072186 .  
# hour19                            -1.4884450  1.1637400  -1.279 0.201393    
# hour20                                    NA         NA      NA       NA    
# week_day2                          0.1777792  0.1849895   0.961 0.336934    
# week_day3                          0.1252071  0.1847341   0.678 0.498184    
# week_day4                          0.2919745  0.1992967   1.465 0.143447    
# week_day5                          0.0471027  0.1911021   0.246 0.805397    
# week_day6                          0.4098338  0.1901265   2.156 0.031520 *  
# week_day7                         -0.4238525  0.1907508  -2.222 0.026660 *  
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Residual standard error: 1.081 on 590 degrees of freedom
#   (365 observations deleted due to missingness)
# Multiple R-squared:  0.417,	Adjusted R-squared:  0.2816 
# F-statistic:  3.08 on 137 and 590 DF,  p-value: < 2.2e-16

# Taking out the least relevant variables
linear_model2 = lm(train$point_security~. 
                   -bus_or_ped -busdestination -inside_or_outside -total_seats -total_female_count 
                   -empty_seats -companions -education -origin -destination -hour, data = train)
summary(linear_model2)

linear_model3 = lm(train$point_security~ +gender +age, data = train)
summary(linear_model3)


pred_data = predict(linear_model2, newdata = test)
SSE = sum((pred_data - test$point_security)^2)
pred_mean = mean(train$point_security)
SST = sum((pred_mean - test$point_security)^2)
OSR2 = 1-SSE/SST
RMSE = sqrt(sum((pred_data - test$point_security)^2)/nrow(test))
MAE = sum(abs(pred_data - test$point_security))/nrow(test)