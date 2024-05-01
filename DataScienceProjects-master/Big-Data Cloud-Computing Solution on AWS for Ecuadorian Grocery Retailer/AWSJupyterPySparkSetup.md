
# Setting up Jupyter with Spark Running on AWS cluster

Most important input is my step1 everyone read it carefully, even people who are not using Jupyter on AWS.

## Step1:   
•	Make sure you have uninstalled all VPN manipulating extension on google chrome such FoxyProxy, DotVPN  
•	We all know it is not a good practice to use "Anywhere" while setting port but when you give "My IP" for your port when you move from one place to other in UMN campus your port keeps changing and you cannot access the AWS and it throws error. So just for trend market place set all port to "Anywhere". For jupyter on AWS make sure 48888 is on "Anywhere" for sure.
  
## Step2:   
You must open an EMR cluster not just EC2  
Before launching the EMR cluster make sure you are in US.East(N. Virginia)  

## Step3:   
Follow the below link step by step which was shared in the AWS lab  

### How to Configure a Spark Cluster with Jupyter Notebook
  
**This instruction is about the Create Cluster stage only.**   

In this configuration, we use an exteral bootstrap script to create a cluster with Jupyter Notebook (for spark) installed:

1. At the top of the page **Create Cluster** page, you will see the ability to switch to **Go to advanced options**
2. Choose a release between EMR 5 up until 5.8 (those have been tested to work).
3. Choose the software you need installed. The default options are sufficient unless you need additional items, such as **Spark**. The more you choose, it may take longer to start. 
4. Click **Next**
5. Under **Hardware configuratio**n:
    1. Choose **uniform instance groups**
    2. Choose the default network and default EC2 subnet.
    6. If you're working on a small project, the default EBS volume size is sufficient. However, if the project is dealing with fairly large data sets, you'll need to add additional EBS storage according to project needs. 
    7. Node types can be changed and instance counts can be changed as well. However, if you're unsure what you need, it is best to leave the instances configured as they are. 
    8. To enable auto-scaling, you will need to determine which rules will be used to scale up and scale down. On namenodes, you can add AutoScale rules, by clicking Autoscale and adding your minimum and maximum node counts. The rules can be defined as you so choose, but if you're unsure what those rules should be, use the default. 
8. Once your hardware has been configured, click **Next** to move on to the **General Cluster Settings** page. 
9. In **Cluster Name** give your cluster a name. 
10. Take note of the S3 logging folder location so that you can retrieve logs if you experience errors.
11. At the bottom, next to **Add bootstrap action,** select **Custom Action** from the drop down and click **Configure and add**
    1. A pop-up window will give you the option add a name to the action, i.e. `Jupyter Install`. 
    2. The script location is `s3://csom-emr5-packages/Jupyter/jupyter-emr.sh`
    3. In the arguments section, Type: 
```--notebook-dir, s3://<bucket>/notebooks, --copy-samples```
    12. Select `Add`
![bootstrap configuration](img/bootstrap.png)
13. Click `Next` to proceed to the last page.


## Step4:  
Without any http:// or https:// just type in your URL as mentioned below  
ec2-XX-XXX-XX-XXX.compute-1.amazonaws.com:48888  

## Step5:  
Jupyter notebook will popup on your browser and it will ask for password. The default password is  
password123  
