# Data Preparation Guideline
## Prepare rawdata
To create a custom dataset, you must prepare all the following necessary raw data in the same format as the ones you can found in the `/data/steam` directory.  
The required data includes:
1. metadata.json  
This file contains the meta information of each item. Each line is a JSON object and must contain a "title" field as the natural language to represent this item.  
format:
    ```
    {"id": 1, "title": "item title 1"}
    {"id": 2, "title": "item title 2"}
    ```
1. sequential_data.txt  
This file contains the sequence of user-item interactions.  
Format:
    ```
    userid_1 itemid_11 itemid_12 ...
    userid_2 itemid_21 itemid_22 ...
    ```
2. negative_samples.txt  
For ranking tasks, this file contains negative samples for each user.  
Format:
    ```
    userid_1 neg_itemid_11 neg_itemid_12 ...
    userid_2 neg_itemid_21 neg_itemid_22 ...
    ```
## Generate prompt with template
Next, you can follow the scripts in preprocess/generate_data.py to create the prompts used for all tasks.   
Usage:
```bash
python ./preprocess/generate_data.py --tasks retrieval,ranking,explanation,conversation,chatbot --sample_num 10 --dataset steam
```