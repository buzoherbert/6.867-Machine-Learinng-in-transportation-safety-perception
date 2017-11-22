install.packages("NISTunits", dependencies = TRUE);
install.packages("reshape2", dependencies = TRUE);
install.packages("plotly", dependencies = TRUE);
install.packages("ggplot2", dependencies = TRUE);
install.packages("epiDisplay", dependencies = TRUE);

library(epiDisplay);
library(plotly);
library(NISTunits);
library(reshape2);
library(ggplot2);
library(data.table);

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

safety_data[["haversine"]] = haversine(safety_data, "cetram_lat", "cetram_long", "latitude", "longitude");

ste <- function(x) sd(x)/sqrt(length(x))

#attach(.data);
field = "base_study_zone"
#myData = aggregate.plot(safety_data[["pointsecurity"]], list(field = safety_data[[field]]),  function(x){ c(Mean=mean(x), se=ste(x)) })
x=aggregate.plot(x=safety_data[["pointsecurity"]], by=list(field = safety_data[[field]]), error="ci",  legend = "none", main=title);
title(main=sprintf("safety perception vs %s", field))
num = aggregate(safety_data[["pointsecurity"]], list(field = safety_data[[field]]), length)
num;
#row.names(myData) <- myData[,1];
#myData <- subset(myData, select = -field );