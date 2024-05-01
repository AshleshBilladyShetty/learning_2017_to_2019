# Hive for ETL of POS data

#### Git Bash tunneled to EMR cluster - Hive
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
