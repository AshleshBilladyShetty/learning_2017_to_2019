# On AWS SparklyR post RStudio Installation

STEP1: git bash to EMR cluster, Do port 8787 to anywhere on AWS

STEP2: update 
```
sudo yum update
sudo yum install libcurl-devel openssl-devel # used for devtool
```
STEP3:install RStudio
```
wget -P /tmp https://s3.amazonaws.com/rstudio-dailybuilds/rstudio-server-rhel-0.99.1266-x86_64.rpm
sudo yum install --nogpgcheck /tmp/rstudio-server-rhel-0.99.1266-x86_64.rpm
```
STEP4:install RStudio
## Make User
```
sudo useradd -m rstudio-user
sudo passwd rstudio-user
```

## Create new directory in hdfs
```
hadoop fs -mkdir /user/rstudio-user
hadoop fs -chmod 777 /user/rstudio-user
```
## STEP5: Swith user
### create directories on hdfs for new user
```
hadoop fs -mkdir /user/rstudio-user
hadoop fs -chmod 777 /user/rstudio-user
```
## switch user
```
su rstudio-user
```

## STEP6: launch Rstudio on URL 

Connect to R through a browser with IP:8787 (this IP will be a public IP for the EMR cluster - located in hardware tab of EMR cluster page )

http://18.217.7.238:8787/  


Important Links

https://spark.rstudio.com/examples/yarn-cluster-emr/

https://medium.com/ibm-data-science-experience/read-and-write-data-to-and-from-amazon-s3-buckets-in-rstudio-1a0f29c44fa7


