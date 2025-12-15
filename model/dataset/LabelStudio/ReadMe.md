# Using LabelStudio with Yaak
Looking to add your own types to the Yaak PII replacement tool? Label Studio Provides an easy way to train the model for whatever types of data you might care about. 

To learn more about Label Studio, vist our [website](labelstud.io) 

## Step 1: Install Label Studio 
< TODO > 

## Step 2: Preparing your data
Label Studio accepts a wide variety of data formats. For this project, you'll need a set of texts containing the types of PII that you're looking to train Yaak to replace. 

For this project, we'd reccomend using a CSV file to store all of the data you'd like to label, with a row title "text". You can save this file with any name you'd like -- for this demo, we'll call it `sample_pii.csv`. This sample file is available in the same folder as this ReadMe for your convenience.

## Step 3: Creating a Project
Once you're logged into Label Studio, click on the `Create Project` button, shown here on the welcome screen of the Open Source edition, or available in the Projects tab, located in the hamburger menu in the left corner next to the Label Studio logo. 

![screenshot of Label Studio welcome page, showing the create project button and the hamburger menu](/LSO_welcome.png)

On the next screen, give your new project a name that reflects the work you'll be doing. (We'd reccomend something like PII-< your label types here >). You can also provide a description, which is a great place to add extra information about your project's purpose or data set. When you're ready, click on the Data Import tab to move to the next step.

![screenshot of Label Studio create project widget, step 1: naming your project and giving project description](/create_project_1.png)

On the data import screen, there are a number of ways to upload data. For this project, we'll use the CSV file we created in step 2. All we need to do is drag it into the blue box, or click on the upload files button and select it from our file explorer. You'll be prompted to select whether you want to treat the CSV/TSV as a list of tasks or a Time Series or Whole Text File -- select "list of task". Click on the Labeling Setup button when you're ready to move to the next step. 

![screenshot of Label Studio's prompt after uploading a CSV](/csv_import.png)

Now, we're ready to set up the UI that we'll use to label our data in Label Studio. Click on the Custom template button, found at the bottom of the list of templates on the left hand side of the screen, and circled in red on the screenshot below. 

![screenshot of Label Studio's template page, with `custom template` circled in red](/custom_template.png)

You should now see the Labeling Interface's code editor on your screen. Label Studio uses a custom XML-style interface to define all the elements of the UI used for labeling. We've provided a template for you to use. Find the `LabelingConfig.txt` file in this folder, and copy-paste it into the editor on Label Studio. We've included the  < View > tags in this version, so you can delete anything in the Code editor. You can see what this looks like in the screenshot below. 

![screenshot of the updated labeling config in Label Studio](/labeling_config.png)

In lines 5-11, we define all of the PII types that we'll be looking for in this project. For our sample, we're looking for pretty standard entities, but you can change these here. All you'd need to do is swap out the `value` field for each label with the value you'd like to annotate for. You can also change the color by changing the `background` field of each label. If you'd like to add more options, simply add more `Label` tags to the list -- make sure you add them between the `Labels` tags. To get rid of extra tags you don't need, simply delete them. The right hand side of the screen will update in real time to show you what your interface will look like.

In lines 15-18 of the original file, we define the relations that we'll use in this project for coreference resolution (identifiying what pronouns refer to what nouns). In the demo, we only use one Relation, `refers-to`, to capture all of these. If you'd like to be more granual, you can add more in the same fashion that you add more labels to the Labels area of the config. When you're done, click on the blue save button. 

## Step 4: Labeling your data
Now, you're ready to label! You can label individual task by clicking on them directly, but we'll use the `Label All Tasks` button (circled in red on the screenshot below). Simply click it  to enter the labeling process.

![screenshot of Label Studio Data manager, with the Label All Tasks button circled in red](/label_all_tasks.png)

To label entities, simply click on the label you'd like to apply and then select the appropriate text. Do this for as many labels as are applicable to your sample. Make sure to use the Pronouns tag to label your pronouns! Then, to add relations, select the pronoun from the regions list on the bottom right corner of the screen, and then from the Info tab on the top right of the screen, select the "create relation between regions" button, which looks like two squares connected by an arrow. Then click on the noun that the pronoun refers to. You'll see an arrow populate on the screen to show this relation. When you're done, click `Submit` to save your annotation. If you're using the `Label All Tasks` view, this will automatically bring you to the next task. You can see this whole process in action in the gif below! 

![GIF showing the process for labeling a sample of data for PII and relations in Label Studio](/labeling_pii.gif)

## Step 5: Retraining the Yaak Model 
< To Do >
