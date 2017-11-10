install.packages("NISTunits", dependencies = TRUE);
install.packages("GGally", dependencies = TRUE);

library(NISTunits);
# Graphing library
library(GGally)


# loading file
safety_data = read.csv("safety_data.csv");

# Removing rows with no safety perception measurement
completeFun <- function(data, desiredCols) {
  completeVec <- complete.cases(data[, desiredCols])
  return(data[completeVec, ])
}

#safety_data = completeFun(safety_data, "pointsecurity")

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

# Making a basic plot of some potentially relevant variables

plot_data <- data.frame(
  point_security = safety_data[["pointsecurity"]],
  haversine = safety_data[["haversine"]],
  gender = safety_data[["gender"]],
  age = safety_data[["age"]],
  companions = safety_data[["companions"]],  
  education = safety_data[["educational_attainment"]],
  trip_purpose = safety_data[["trip_purpose"]],
  time = safety_data[["time"]],
  base_study_zone = safety_data[["base_study_zone"]],
  busdestination = safety_data[["busdestination"]],
  inside_or_outside = safety_data[["inside_or_outside"]]
  
                        );
# Getting a summary of the data
summary(plot_data)

# plotting the data
ggpairs(plot_data,)

