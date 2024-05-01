
# R Studio - SparklyR (Spark, dplyr, Prophet-forecasting)

STEP1: Install all the required packages
```R
#install.packages('sparklyr')
#install.packages('aws.s3')
#library(aws.s3)
#bucketlist()
#install.packages("prophet")
#install.packages("dplyr")
```
STEP2: Activate all the packages and set up the environment for sparklyr
```R
library(sparklyr)
library(prophet)
library(dplyr)
Sys.setenv(SPARK_HOME="/usr/lib/spark")
config <- spark_config()
sc <- spark_connect(master = "yarn-client", config = config, version = '2.2.0')
```

STEP3: Get the file from S3 bucket
```R
daily_grocery <- spark_read_csv(sc,name = "file1",path = "s3://bigdataprjct/historical_data/date_lvl/date_lvl.csv")
```

STEP4: Use dplyr fucntions that can process the big data on spark cluster and reduce the data to do forecasting in R
```R
daily_grocery %>% top_n(743)
```

STE5: Conduct forecasting using facebook's developed forecasting package "prophet"
```R
daily_grocery <- as.data.frame(daily_grocery)
daily_grocery$ds <- as.Date(daily_grocery$date, "%m/%d/%Y")

daily_grocery$y <- daily_grocery$sales
forecastinData <- daily_grocery[c("ds","y")]
m <- prophet(forecastinData)
future <- make_future_dataframe(m, periods = 180)
fd <- predict(m, future)
fd_final <- copy_to(sc, fd)
```
STEP6: Send the forecasted data bact to S3 to plot in Amazon Quick Sight
```
spark_write_csv(fd_final,path = "s3://bigdataprjct/Miscellaneous/test.csv")
```
