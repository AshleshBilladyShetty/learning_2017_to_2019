# MaroonTeam3
## End-to-End Big-Data Cloud-Computing Solution for Ecuadorian Grocery Retailer using AWS

Codes were written in three main platforms 
1. Git Bash tunneled to EMR cluster - Hive
2. Jupyter - Pyspark
3. R Studio - SparklyR

### 1. Git Bash tunneled to EMR cluster - Hive
Creating External table in Hive to access the data present in “Stores” table on S3:
Stores:
```hive
CREATE EXTERNAL TABLE Stores (STORE_NBR INT, CITY STRING, STATE STRING, TYPE STRING, CLUSTER INT) 
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/InputFiles/stores/';
```
Creating External table in Hive to access the data present in “Items” table on S3:
Items:
```
CREATE EXTERNAL TABLE Items (ITEM_NBR INT, FAMILY STRING, CLASS INT, PERISHABLE INT) 
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/InputFiles/items/';
```
Creating External table in Hive to access the data present in “Holiday” table on S3:
Holiday:
```
CREATE EXTERNAL TABLE HOLIDAY (DATE DATE, TYPE STRING, LOCALE STRING, LOCALE_NAME STRING, DESCRIPTION STRING, TRANSFERRED STRING) 
FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION '';
```
Creating External table in Hive to access the data present in “Oil” table on S3:
OIL:
```
CREATE EXTERNAL TABLE OIL (DATE_NUM DATE, dcoilwtico FLOAT) 
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/oil/';
```
Creating External table in Hive to access the data present in “Transaction” table on S3:
Transaction Table :
```
CREATE EXTERNAL TABLE TRANSACTIONS (DATE_NUM DATE, STORE_NBR INT, TRANSACTIONS INT) 
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/transactions/';
```
#Aggregation of Transaction level data at day level
Transaction table at day level:
```
CREATE EXTERNAL TABLE TRANSACTIONS_DAY_LVL (DATE_NUM DATE, TRANSACTIONS INT) 
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/transactions/transactions_day_lvl/';

INSERT OVERWRITE TABLE TRANSACTIONS_DAY_LVL
SELECT DATE_NUM,SUM(TRANSACTIONS) AS TRANSACTIONS FROM TRANSACTIONS GROUP BY DATE_NUM;
Creating External table in Hive to access the data present in “Transaction-Line-Item file on S3:
TRANS_LINE_ITEM table
CREATE EXTERNAL TABLE TRANS_LINE_ITEM (ID INT, DATE_NUM DATE, STORE_NBR INT, ITEM_NBR INT, UNIT_SALES INT, ON_PROMOTION STRING) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/TRANS_LINE_ITEM/';
```
Aggregated Table at Line-Item table at day-item level
Line_Item_Aggregate table:
```
CREATE EXTERNAL TABLE LINE_ITEM_AGG (DATE_NUM DATE, DIST_STR INT, DIST_ITEM INT, SALES INT) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/line_item/';

INSERT OVERWRITE TABLE LINE_ITEM_AGG
SELECT DATE_NUM, COUNT(DISTINCT STORE_NBR) as DIST_STR, COUNT(DISTINCT item_nbr) as DIST_ITEM, SUM(UNIT_SALES) AS SALES from TRANS_LINE_ITEM GROUP BY DATE_NUM; 
```

Day level table with current day’s information 
Creating day level table for current day:
```
CREATE EXTERNAL TABLE DATE_LVL_TODAY (DATE_NUM DATE, DIST_STR INT, DIST_ITEM INT, SALES INT, dcoilwtico FLOAT, TOT_TRANS INT) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/DATE_LVL_TODAY/';
INSERT OVERWRITE TABLE DATE_LVL_TODAY
SELECT A.DATE_NUM, DIST_STR, DIST_ITEM,SALES, dcoilwtico, TRANSACTIONS FROM LINE_ITEM_AGG A LEFT JOIN TRANSACTIONS_DAY_LVL B ON A.DATE_NUM=B.DATE_NUM LEFT JOIN OIL C ON A.DATE_NUM=C.DATE_NUM
```
Top 5 and Bottom 5 Stores based on Sales
Creating Top_Bottom_Stores :
```
CREATE TABLE TOP_BOTTOM_STORES (
DATE_NUM DATE,
STORE_NBR INT,
TOT_SALES FLOAT,
TOP_BOTTOM STRING,
RANK_STORE INT) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/DATE_LVL_TOP_BOTTOM_STORES/';

INSERT OVERWRITE TABLE TOP_BOTTOM_STORES 
SELECT DATE_NUM,STORE_NBR,TOT_SALES,A.TOP_BOTTOM, A.RANK_STORE
FROM (SELECT DATE_NUM,STORE_NBR,T.TOT_SALES, 'top'  AS TOP_BOTTOM, RANK() OVER (ORDER BY T.TOT_SALES DESC) AS RANK_STORE FROM ( SELECT DATE_NUM, STORE_NBR, SUM(UNIT_SALES) AS TOT_SALES from TRANS_LINE_ITEM GROUP BY DATE_NUM,STORE_NBR ) T 
UNION
SELECT DATE_NUM,STORE_NBR,T.TOT_SALES, 'bottom'  AS TOP_BOTTOM, RANK() OVER (ORDER BY T.TOT_SALES) AS RANK_STORE FROM ( SELECT DATE_NUM, STORE_NBR, SUM(UNIT_SALES) AS TOT_SALES from TRANS_LINE_ITEM GROUP BY DATE_NUM,STORE_NBR HAVING TOT_SALES>0) T ) A WHERE A.RANK_STORE<=5
```
Top 5 and Bottom 5 Cities based on Sales
Creating Top_Bottom_Cities:
```
CREATE TABLE TOP_BOTTOM_CITIES (
DATE_NUM DATE,
CITY STRING,
TOT_SALES FLOAT,
TOP_BOTTOM STRING,
RANK_STORE INT) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/DATE_LVL_TOP_BOTTOM_CITIES/';

INSERT OVERWRITE TABLE TOP_BOTTOM_CITIES 
SELECT DATE_NUM,CITY,TOT_SALES,A.TOP_BOTTOM, A.RANK_CITY
FROM (SELECT DATE_NUM,CITY,T.TOT_SALES, 'top'  AS TOP_BOTTOM, RANK() OVER (ORDER BY T.TOT_SALES DESC) AS RANK_CITY FROM ( SELECT DATE_NUM, CITY, SUM(UNIT_SALES) AS TOT_SALES from TRANS_LINE_ITEM TR LEFT JOIN STORES S ON TR.STORE_NBR=S.STORE_NBR GROUP BY DATE_NUM,CITY ) T 
UNION
SELECT DATE_NUM,CITY,T.TOT_SALES, 'bottom'  AS TOP_BOTTOM, RANK() OVER (ORDER BY T.TOT_SALES) AS RANK_CITY FROM ( SELECT DATE_NUM, CITY, SUM(UNIT_SALES) AS TOT_SALES from TRANS_LINE_ITEM TR LEFT JOIN STORES S ON TR.STORE_NBR=S.STORE_NBR GROUP BY DATE_NUM,CITY HAVING TOT_SALES>0) T ) A WHERE A.RANK_CITY<=5
```

#Top 5 and Bottom 5 Items based on Sales
Creating Top_Bottom_Items:
```
CREATE TABLE TOP_BOTTOM_ITEMS (
DATE_NUM DATE,
FAMILY STRING,
TOT_SALES FLOAT,
TOP_BOTTOM STRING,
RANK_STORE INT) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' WITH SERDEPROPERTIES  
( "separatorChar" = ",", 
 "quoteChar"="\"" )
LOCATION 's3://bigdataprjct/data_today/DATE_LVL_TOP_BOTTOM_ITEMS/';

INSERT OVERWRITE TABLE TOP_BOTTOM_ITEMS 
SELECT DATE_NUM,FAMILY,TOT_SALES,A.TOP_BOTTOM, A.RANK_ITEM
FROM (SELECT DATE_NUM,FAMILY,T.TOT_SALES, 'top'  AS TOP_BOTTOM, RANK() OVER (ORDER BY T.TOT_SALES DESC) AS RANK_ITEM FROM ( SELECT DATE_NUM, FAMILY, SUM(UNIT_SALES) AS TOT_SALES from TRANS_LINE_ITEM TR LEFT JOIN ITEMS S ON TR.ITEM_NBR=S.ITEM_NBR GROUP BY DATE_NUM,FAMILY ) T 
UNION
SELECT DATE_NUM,FAMILY,T.TOT_SALES, 'bottom'  AS TOP_BOTTOM, RANK() OVER (ORDER BY T.TOT_SALES) AS RANK_ITEM FROM ( SELECT DATE_NUM, FAMILY, SUM(UNIT_SALES) AS TOT_SALES from TRANS_LINE_ITEM TR LEFT JOIN ITEMS S ON TR.ITEM_NBR=S.ITEM_NBR GROUP BY DATE_NUM,FAMILY HAVING TOT_SALES>0) T ) A WHERE A.RANK_ITEM<=5;
```


### 2. Jupyter - Pyspark

#### 2.1. Creation of the historical dataset


Importing and creating the SQL context
```
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
```
Importing the various functions for further usage.
```
from pyspark.sql.types import StructType, StringType
from pyspark import SparkConf, SparkContext
from pyspark import sql
from pyspark.sql.functions import lit
import numpy as np
import pandas as pd
```
Declaring a empty struct to later create an empty dataframe.
```
schema = StructType([])
```
Creating an empty dataframe to hold the graph numbers to include in the historical dataset to use in the dashboard.
```
empty = sqlContext.createDataFrame(sc.emptyRDD(), schema)
````
Declaring an array with the graph numbers
```
graph = np.array([1,2,3,4,5])
```
Creating the dataframe for the above created array
```
graphdf = pd.DataFrame(graph, columns = ['Graph'])
```
Registering the graph dataframe as a temp table to be used later for joining with historical datasets
```
graphDF.registerTempTable("grph_tbl")
graphDF.show()
```
##### Reading and formatting the different historical datasets to be used in the dashboard
##### Reading the historical top bottom 5 cities dataset from S3 bucket
```
city = sqlContext.read.format('com.databricks.spark.csv') \
    .option("inferSchema",True).option("header",True).load('s3://bigdataprjct/historical_data/top_bottom_5_city/top_bottom_5_city.csv')
```
Transforming the read city dataframe into RDD
```
cityDataRDD = city.rdd
```
Caching the city RDD for faster processing and avoid reading from the bucket everytime
```
cityDataRDD.cache()
```
Converting the city data RDD into a dataframe and providing the column names to the various columns a to the requirement for dashboarding
```
cityDataDF = cityDataRDD.toDF(['Date_chr1','City1','CitySales1','TopBottom1','Rank1']) 
```
Adding column for Graph# according to the Graph that will be created using this portion of the data and joining with the temp graph table created above.
```
cityDataDF = cityDataDF.withColumn('Graph',lit(1))
``` 
Registering the above created city dataframe as a temp table to join with the Graph table created earlier
```
cityDataDF.registerTempTable("city_g1")
```
Viewing the schema to check if it looks correct
```
cityDataDF.printSchema()
```
Checking the data in the city dataframe
```
cityDataDF.show()
```
Joining the above formed city table with graph table
```
g1_data = sqlContext.sql("""
    SELECT gr.*, c1.Date_chr1, c1.City1, c1.CitySales1, c1.TopBottom1, c1.Rank1 
    from grph_tbl gr
    LEFT JOIN city_g1 c1
    ON gr.Graph = c1.Graph
""")

g1_data.printSchema()
```
Looking at the above created dataset
```
g1_data.show()
```
Registering the above dataframe as a temp table to be joined later with other data parts.
```
g1_data.registerTempTable("g1_c")
```
##### Reading the historical items data for Graph 2
##### Reading the historical data for top and bottom items from S3 bucket for further processing
```
item = sqlContext.read.format('com.databricks.spark.csv') \ .option("inferSchema",True).option("header",True).load('s3://bigdataprjct/historical_data/top_bottom_5_item/top_bottom_5_item.csv')
```
Converting the iteam dataset read into RDD for further processsing
```
itemDataRDD = item.rdd
```
Caching the read item RDD for faster processing
```
itemDataRDD.cache()
```
Converting the RDD to dataframe and renaming the column as per the requirement for dashboarding
```
itemDataDF = itemDataRDD.toDF(['Date_chr2','Family2','ItemSales2','TopBottom','Rank2']) 
```
Adding the column Graph number and declaring all row values in the column to be 2 since item level graph in dashboard in number 2 and will further be joined with the above created final city dataset.
```
itemDataDF = itemDataDF.withColumn('Graph',lit(2))    
```
Registering a temp table for items data to be further joined with above creaated final city dataset
```
itemDataDF.registerTempTable("item_g2")
```
Joining the above created item table with the final city datset created earlier which also has the graph number data
```
g2_data = sqlContext.sql("""
    SELECT g1.*, ig.Date_chr2, ig.Family2, ig.ItemSales2, ig.TopBottom, ig.Rank2 
    from g1_c g1
    LEFT JOIN item_g2 ig 
    ON g1.Graph = ig.Graph
""")

g2_data.printSchema()
```
Checking if the data was populated as we wanted into the above created dataset. Now there is data from Graph 1 and 2 in the dataframe while 3, 4 and 5 are still empty.
```
g2_data.show()
```
Registering a temp table for above created dataframe so that it can be further used for processing.
```
g2_data.registerTempTable("g2_ci")
```
##### Reading the historical store level data
Reading the historical store level data from AWS S3 bucket
```
store = sqlContext.read.format('com.databricks.spark.csv') \
.option("inferSchema",True).option("header",True).load('s3://bigdataprjct/historical_data/top_bottom_5_store/top_bottom_5_store.csv')
```
Transforming the read dataset into RDD
```
storeDataRDD = store.rdd
```
Caching the RDD to avoid re-read from source and for faster processing
```
storeDataRDD.cache()
```
Converting the RDD to dataframe and renaming the column so that it can be used in the dashboard with ease
```
storeDataDF = storeDataRDD.toDF(['Date_chr3','StoreNbr3','ItemSales3','TopBottom3','Rank3'])
```
Adding the Graph number to the store dataframe and populating the column with numeric 3 since the graph number for store level data is 3.
```
storeDataDF = storeDataDF.withColumn('Graph',lit(3))
```
Registering the store dataframe as a temp table to join it with the earlier created city and item level dataset and graph number
```
storeDataDF.registerTempTable("store_g3")
```
Joining the store data to the earlier created city and item dataset based on the graph number
```
g3_data = sqlContext.sql("""
    SELECT g1.*, sg.Date_chr3, sg.StoreNbr3, sg.ItemSales3, sg.TopBottom3, sg.Rank3 
    from g2_ci g1
    LEFT JOIN store_g3 sg 
    ON g1.Graph = sg.Graph
""")

g3_data.printSchema()
```
Checking if the data is present in the format that we need for dashboarding. The graph number 1, 2 and 3 now have data corresponding to the city, item and store while 4 and 5 have blank rows
```
g3_data.show()
```
Registering the above created final datset as a temp table for further processing
```
g3_data.registerTempTable('g3_cis')
```
##### Reading the historical day level transaction data for 4th dashboard
Reading the historical day level transaction data from the S3 bucket to join with previously created dataset
```
date = sqlContext.read.format('com.databricks.spark.csv') \
.option("inferSchema",True).option("header",True).load('s3://bigdataprjct/historical_data/date_lvl/date_lvl.csv')
```    
Transforming the read day level transaction data into RDD
```
dateDataRDD = date.rdd
```
Caching the data read above for faster processing and avoiding re-read
```
dateDataRDD.cache()
```
Converting the RDD into dataframe and providing it column names according to the dashboarding requirements.
```
dateDataDF = dateDataRDD.toDF(['Date_chr4','StoreNbr4','Item4','Sales4','ItemCount4','Dcoil4','HolidayFlg4','TrnsCount4'])
```
Adding the Graph number column to the above dataframe and populating the rows with numeric value 4 since the day level transaction data will be used for creating the 4th graph in the dashbaord.
```
dateDataDF = dateDataDF.withColumn('Graph',lit(4))
```
Registering a temp table for above created dataframe to be used further
```
dateDataDF.registerTempTable("date_g4")
```
Since the dashbaord 4th graph uses only last 14 days data we process the above formed dataframe to contain only last 14 days data

Converting the date column in the above dataframe to teh date format and taking only the required sales column used for dashbaording from the data.
```
g4_data = sqlContext.sql("""
    SELECT sg.Graph,sg.Date_chr4, TO_DATE(CAST(UNIX_TIMESTAMP(sg.Date_chr4, 'MM/dd/yyyy') AS TIMESTAMP)) AS Date_dt4,sg.Sales4
    FROM date_g4 sg    
""")

g4_data.printSchema()
```
Checking if the got the data as we wanted
```
g4_data.show()
```
Registering the above transformed and filtered dataset for further transformation that is filtering the last 14 days data.
```
g4_data.registerTempTable("date_sg")
```
Filtering the above data to contain only the last 14 days data
```
final_dt_lvl = sqlContext.sql("""
     SELECT sg.Graph, sg.Date_chr4, sg.Sales4
    FROM date_sg sg 
    WHERE sg.Date_dt4 >= (SELECT date_sub(MAX(Date_dt4),13) FROM date_sg)
""")
```
Check if we got the data we needed
```
final_dt_lvl.show()
```
Registering the above created dataframe to be joined with the previously created dataset for Graphs 1, 2 and 3.
```
final_dt_lvl.registerTempTable("date_lvl_4")
```
Joining the data for Graph 4 with the previous data created for Graph 1, 2 and 3.
```
dtlvljoin = sqlContext.sql("""
        SELECT g1.*, dtl.Date_chr4, dtl.Sales4
    from g3_cis g1
    LEFT JOIN date_lvl_4 dtl 
    ON g1.Graph = dtl.Graph
        """)

dtlvljoin.printSchema()
```
Registering the data created till Graph 4 into a temp table for further processing
```
dtlvljoin.registerTempTable("g4F")
```
##### Reading the data for graph 5 - linear regression showing relation between lag days transaction amounts

Reading the liner regression graph data from S3 bucket.
```
reg = sqlContext.read.format('com.databricks.spark.csv') \
.option("inferSchema",True).option("header",True).load('s3://bigdataprjct/historical_data/linear_reg/Regression_Historical.csv')
```
Registering the above read data into a RDD
```
regRDD = reg.rdd
```
Caching the RDD for easier and faster processing further
```
regRDD.cache()
```
Transforming the above RDD into dataframe and renaming teh column as requirement for the Graph in the dashboard.
```
regDF = regRDD.toDF(['Date_chr5','Term5','Variable5','Value5']) 
```
Adding the column for graph number to the above dataframe and populating the rows with value 5 since the graph for this on the dashboard is graph 5
```
regDF = regDF.withColumn('Graph',lit(5))
```
Registering the above dataframe as temp table for further processing
```
regDF.registerTempTable("reg_g5")
```
Joining the above created graph 5 data with the previously created data for graphs 1-4.
```
g5DF = sqlContext.sql("""
    SELECT g1.*, rg.Date_chr5, rg.Term5, rg.Variable5, rg.Value5 
    from g4F g1
    LEFT JOIN reg_g5 rg 
    ON g1.Graph = rg.Graph
""")

g5DF.printSchema()
```
Registering the above created final dataset as temp table for further processing.
```
g5DF.registerTempTable("g5F")
```
Replacing and putting the columns in the place as required for the final dashbaording.
```
final5 = sqlContext.sql("""
        SELECT Graph, COALESCE(Date_chr1, Date_chr2, Date_chr3,Date_chr4,Date_chr5) AS Date, 
        COALESCE(Date_chr1, Date_chr2, Date_chr3,Date_chr4, Date_chr5) AS Date_chr,
        City1, CitySales1, TopBottom1, Rank1, Family2, ItemSales2, TopBottom, Rank2, 
        StoreNbr3, ItemSales3, TopBottom3, Rank3,
        Sales4, Term5, Variable5, Value5
        FROM g5F 
        """)

final5.printSchema()
```
Dropping the unnecessary columns from the above dataframe
```
from functools import reduce
from pyspark.sql import DataFrame

final_his = reduce(DataFrame.drop, ['Date_chr1','Date_chr2','Date_chr3', 'Date_chr4','Date_chr5'], final5)
```
##### Clearing the space on S3 where we are going to write this historical dataset
```
import os

cmd="hdfs dfs -rm -r -skipTrash s3n://bigdataprjct/Miscellaneous/historical"
os.system(cmd)
```
Writing the final dataset to a folder on S3 bucket.
```
final_his\
   .coalesce(1)\
   .write.format("com.databricks.spark.csv")\
   .option("header", "true")\
   .save("s3n://bigdataprjct/Miscellaneous/historical")
```
#### 2.2 Transforming and Appending the day level dataset to the historical dataset for the dashboard
##### The section 2.2 is sparsely commented since the code structure is similar to 2.1 but it generates the next days backend file for the daily dashboard refresh on Amazon Quick Sight
##### Reading the different daily datasets from S3

```
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
```

```
from pyspark.sql.types import StructType, StringType
from pyspark import SparkConf, SparkContext
from pyspark import sql
from pyspark.sql.functions import lit
import numpy as np
import pandas as pd
```

```
schema = StructType([])
```

```
empty = sqlContext.createDataFrame(sc.emptyRDD(), schema)
```

```
graph = np.array([1,2,3,4,5])
graphdf = pd.DataFrame(graph, columns = ['Graph'])
graphDF = sqlContext.createDataFrame(graphdf)
In [8]:
graphDF.registerTempTable("grph_tbl")
graphDF.show()
```
```
Dcity = sqlContext.read.format('com.databricks.spark.csv') \
    .option("inferSchema",True).option("header",False).load('s3://bigdataprjct/data_today/DATE_LVL_TOP_BOTTOM_CITIES')
DcityDataRDD = Dcity.rdd
DcityDataRDD.cache()
DcityDataDF = DcityDataRDD.toDF(['Date_chr1','City1','CitySales1','TopBottom1','Rank1']) 
DcityDataDF = DcityDataDF.withColumn('Graph',lit(1))
DcityDataDF.registerTempTable("Dcity_g1")
DcityDataDF.printSchema()
DcityDataDF.show()
```

```
Dg1_data = sqlContext.sql("""
    SELECT gr.*,c1.Date_chr1, c1.City1, c1.CitySales1, c1.TopBottom1, c1.Rank1 
    from grph_tbl gr
    LEFT JOIN Dcity_g1 c1
    ON gr.Graph = c1.Graph
""")

Dg1_data.printSchema()
Dg1_data.show()
Dg1_data.registerTempTable("Dg1_c")
```

##### Reading and transforming the day level items data for append and use in Graph 2

```
Ditem = sqlContext.read.format('com.databricks.spark.csv') \
    .option("inferSchema",True).option("header",False).load('s3://bigdataprjct/data_today/DATE_LVL_TOP_BOTTOM_ITEMS')

DitemDataRDD = Ditem.rdd

DitemDataRDD.cache()
```
```
DitemDataDF = DitemDataRDD.toDF(['Date_chr2','Family2','ItemSales2','TopBottom','Rank2'])

DitemDataDF = DitemDataDF.withColumn('Graph',lit(2))

DitemDataDF.registerTempTable("Ditem_g2")
```
```
Dg2_data = sqlContext.sql("""
    SELECT g1.*, ig.Date_chr2, ig.Family2, ig.ItemSales2, ig.TopBottom, ig.Rank2  
    from Dg1_c g1
    LEFT JOIN Ditem_g2 ig 
    ON g1.Graph = ig.Graph
""")

Dg2_data.printSchema()
Dg2_data.show()
Dg2_data.registerTempTable("Dg2_ci")
```

##### Reading and transforming the daily store level data from AWS S3 bucket

```
Dstore = sqlContext.read.format('com.databricks.spark.csv') \
    .option("inferSchema",True).option("header",False).load('s3://bigdataprjct/data_today/DATE_LVL_TOP_BOTTOM_STORES')
DstoreDataRDD = Dstore.rdd
DstoreDataRDD.cache()

```

```
DstoreDataDF = DstoreDataRDD.toDF(['Date_chr3','StoreNbr3','ItemSales3','TopBottom3','Rank3'])
DstoreDataDF = DstoreDataDF.withColumn('Graph',lit(3))
DstoreDataDF.registerTempTable("Dstore_g3")

```
```
Dg3_data = sqlContext.sql("""
    SELECT g1.*, sg.Date_chr3, sg.StoreNbr3, sg.ItemSales3, sg.TopBottom3, sg.Rank3
    from Dg2_ci g1
    LEFT JOIN Dstore_g3 sg 
    ON g1.Graph = sg.Graph
""")

Dg3_data.printSchema()
Dg3_data.show()
Dg3_data.registerTempTable('g3_cis')

```
##### Reading the day level transaction data for 4th dashboard
```

Ddate = sqlContext.read.format('com.databricks.spark.csv') \
    .option("inferSchema",True).option("header",False).load('s3://bigdataprjct/data_today/DATE_LVL_TODAY')
DdateDataRDD = Ddate.rdd
DdateDataRDD.cache()
```

```
DdateDataDF = DdateDataRDD.toDF(['Date_chr4','StoreNbr4','Item4','Sales4','Dcoil4','TrnsCount4'])
DdateDataDF = DdateDataDF.withColumn('Graph',lit(4))
DdateDataDF.show()
DdateDataDF.registerTempTable("date_g4")
```

```
dtlvljoin = sqlContext.sql("""
        SELECT g1.*, dtl.Date_chr4, dtl.Sales4
    from g3_cis g1
    LEFT JOIN date_g4 dtl 
    ON g1.Graph = dtl.Graph
        """)

dtlvljoin.printSchema()
dtlvljoin.registerTempTable("g4F")

```

```
Dg3F = sqlContext.sql("""
        SELECT Graph, COALESCE(Date_chr1, Date_chr2, Date_chr3, Date_chr4) AS Date, 
        COALESCE(Date_chr1, Date_chr2, Date_chr3, Date_chr4) AS Date_chr,
        City1, CitySales1, TopBottom1, Rank1, 
        Family2, ItemSales2, TopBottom, Rank2,
        StoreNbr3, ItemSales3, TopBottom3, Rank3,
        Sales4, "Term5", "Variable5", "Value5"
        FROM g4F
        """)
```
```
Dg3F.registerTempTable("gDaily")
```

```
from functools import reduce
from pyspark.sql import DataFrame

daily = reduce(DataFrame.drop, ['Date_chr1','Date_chr2','Date_chr3','Date_chr4'], Dg3F)
```

##### Reading the historical dataset to append it to the daily one

```
hist = sqlContext.read.format('com.databricks.spark.csv') \
    .option("inferSchema",True).option("header",True).\
    load('s3n://bigdataprjct/Miscellaneous/historical')
hist.printSchema()

```
##### Union the historical and day level data to create the final set

``` 
Final = hist.union(daily)
Final.printSchema()
```

##### Write Final dataset to S3 bucket
```
Final\
   .coalesce(1)\
   .write.format("com.databricks.spark.csv")\
   .option("header", "true")\
   .save("s3n://bigdataprjct/Miscellaneous/updated")
```

#### 2.3. Linear Regression Part - Used for Graph 5 on dashboard

Creating the Linear Regression dataset for finding the relation in the sales amount for last 7 days to see if there is a trend
Reading the day level transaction data present on S3

```
date_reg = sqlContext.read.format('com.databricks.spark.csv') \
    .option("inferSchema",True).option("header",True).load('s3://bigdataprjct/historical_data/date_lvl/date_lvl.csv')
```

Checking the read data
```
date_reg.limit(5).toPandas()
```

Reading the ml feature Vector Assembler for later use in regression if necessary
```
from pyspark.ml.feature import VectorAssembler
```

Adding a numeric sequence column to the dataset to later filter based on this column to create separate dataframe for capturing relation based on lag days sales.

```
from pyspark.sql.functions import monotonically_increasing_id

date_reg = date_reg.withColumn('Seq',monotonically_increasing_id())
```
Registering a temp table from above daat for further processing

```
date_reg.registerTempTable("date1")
```
Creating sepearate dataframes based on sequence number that are corresponding to different dates

```
dt1 = sqlContext.sql("""
        SELECT sales as sale1 from
        date1
        where Seq <= 742 AND Seq >= 563
        """)

dt2 = sqlContext.sql("""
        SELECT sales as sale2 from
        date1
        where Seq <= 741 AND Seq >= 562
        """)
dt3 = sqlContext.sql("""
        SELECT sales as sale3 from
        date1
        where Seq <= 740 AND Seq >= 561
        """)

dt4 = sqlContext.sql("""
        SELECT sales as sale4 from
        date1
        where Seq <= 739 AND Seq >= 560
        """)

dt5 = sqlContext.sql("""
        SELECT sales as sale5 from
        date1
        where Seq <= 738 AND Seq >= 559
        """)

dt6 = sqlContext.sql("""
        SELECT sales as sale6 from
        date1
        where Seq <= 737 AND Seq >= 558
        """)

dt7 = sqlContext.sql("""
        SELECT sales as sale7 from
        date1
        where Seq <= 736 AND Seq >= 557
        """)

dt8 = sqlContext.sql("""
        SELECT sales as sale8 from
        date1
        where Seq <= 735 AND Seq >= 556
        """)


dt1.show()
```
Adding sequence number again to these separated datasets so that they can be joined together with the lag to see if there is a corelation

```
dt1 = dt1.withColumn('Tseq',monotonically_increasing_id())
dt2 = dt2.withColumn('Tseq',monotonically_increasing_id())
dt3 = dt3.withColumn('Tseq',monotonically_increasing_id())
dt4 = dt4.withColumn('Tseq',monotonically_increasing_id())
dt5 = dt5.withColumn('Tseq',monotonically_increasing_id())
dt6 = dt6.withColumn('Tseq',monotonically_increasing_id())
dt7 = dt7.withColumn('Tseq',monotonically_increasing_id())
dt8 = dt8.withColumn('Tseq',monotonically_increasing_id())


dt1.show()
```
Registering the above dataframes as tables for joining them together
```
dt1.registerTempTable("dt1")
dt2.registerTempTable("dt2")
dt3.registerTempTable("dt3")
dt4.registerTempTable("dt4")
dt5.registerTempTable("dt5")
dt6.registerTempTable("dt6")
dt7.registerTempTable("dt7")
dt8.registerTempTable("dt8")
```
Creating the dataset by joining the above different dates dataset
```
dt11 = sqlContext.sql("""
                SELECT dt1.Tseq, dt1.sale1, dt2.sale2
                FROM dt1 INNER JOIN dt2
                ON dt1.Tseq = dt2.Tseq
                """)

dt11.registerTempTable("dt11")

dt12 = sqlContext.sql("""
                SELECT dt11.*, dt3.sale3
                FROM dt11 INNER JOIN dt3
                ON dt11.Tseq = dt3.Tseq
                """)

dt12.registerTempTable("dt12")

dt13 = sqlContext.sql("""
                SELECT dt12.*, dt4.sale4
                FROM dt12 INNER JOIN dt4
                ON dt12.Tseq = dt4.Tseq
                """)

dt13.registerTempTable("dt13")

dt14 = sqlContext.sql("""
                SELECT dt13.*, dt5.sale5
                FROM dt13 INNER JOIN dt5
                ON dt13.Tseq = dt5.Tseq
                """)

dt14.registerTempTable("dt14")

dt15 = sqlContext.sql("""
                SELECT dt14.*, dt6.sale6
                FROM dt14 INNER JOIN dt6
                ON dt14.Tseq = dt6.Tseq
                """)

dt15.registerTempTable("dt15")

dt16 = sqlContext.sql("""
                SELECT dt15.*, dt7.sale7
                FROM dt15 INNER JOIN dt7
                ON dt15.Tseq = dt7.Tseq
                """)

dt16.registerTempTable("dt16")

dt17 = sqlContext.sql("""
                SELECT dt16.*, dt8.sale8
                FROM dt16 INNER JOIN dt8
                ON dt16.Tseq = dt8.Tseq
                """)

dt17.registerTempTable("dt17")
dt17.show()
```
Importing the various libraries required for running a regression

```
import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
```
Selecting the variables needed for running the model

```
dt17 = dt17.select("sale1","sale2","sale3","sale4","sale5","sale6","sale7","sale8")
```
Converting the dataframe into the RDD
```
dt18 = dt17.rdd
```
Running the model on teh above created dataframe
```
from pyspark.mllib.regression import LinearRegressionWithSGD
dt18 = dt18.map(lambda line:LabeledPoint(line[0],[line[1:]]))
linearModel = LinearRegressionWithSGD.train(dt18,10,.2)
linearModel.weights
```

### 3. R Studio - SparklyR

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
