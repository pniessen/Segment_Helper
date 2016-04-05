# data set cleaner
import pandas as pd
infile = '/Users/pniessen/Rosetta_Desktop/Segmentation_2-point-0/segmentr/web/research_files/labelled_data/GP_Content_Seg_Input_File_092115_Full_Data.csv'

X = pd.DataFrame.from_csv(infile)
X = X.ix[3:,6:]

# clean up, remove bad characters
null_counter = 0
for col in X.columns:
    for index in X.index:
        val = X[col][index]
        if val == '#NULL!' or val == ' ':
            val = 'n/a'
            null_counter += 1
        a = chr(146) # single quotation mark
        X[col][index] = val.replace("'","").replace("’","").replace(a,"").replace(" ","_")
        #val.decode('CP1252', 'ignore').encode('utf-8', 'ignore')

print null_counter

# create response_dict
response_dict = {}
for col in X.columns:
    response_dict[col] = {}
    response_dict[col]['verbatim'] = sorted(list(set(X[col])))
    for item in response_dict[col]['verbatim']:
        response_dict[col][item] = response_dict[col]['verbatim'].index(item) + 1
        response_dict[col][response_dict[col][item]] = item

convert X to numeric only
for col in X.columns:
    for index in X.index:
        val = X[col][index]
        X[col][index] = response_dict[col][val]

