

```python
! pip install beautifulsoup4
```

    Requirement already satisfied: beautifulsoup4 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (4.6.0)
    [31mtensorflow 1.3.0 requires tensorflow-tensorboard<0.2.0,>=0.1.0, which is not installed.[0m



```python
! pip install requests
```

    Requirement already satisfied: requests in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (2.18.4)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from requests) (3.0.4)
    Requirement already satisfied: idna<2.7,>=2.5 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from requests) (2.6)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from requests) (1.22)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from requests) (2019.3.9)
    [31mtensorflow 1.3.0 requires tensorflow-tensorboard<0.2.0,>=0.1.0, which is not installed.[0m



```python
! pip install lxml
```

    Requirement already satisfied: lxml in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (4.1.0)
    [31mtensorflow 1.3.0 requires tensorflow-tensorboard<0.2.0,>=0.1.0, which is not installed.[0m



```python
! pip install folium
```

    Requirement already satisfied: folium in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (0.9.1)
    Requirement already satisfied: branca>=0.3.0 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from folium) (0.3.1)
    Requirement already satisfied: requests in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from folium) (2.18.4)
    Requirement already satisfied: numpy in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from folium) (1.13.3)
    Requirement already satisfied: jinja2>=2.9 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from folium) (2.9.6)
    Requirement already satisfied: six in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from branca>=0.3.0->folium) (1.11.0)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from requests->folium) (3.0.4)
    Requirement already satisfied: idna<2.7,>=2.5 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from requests->folium) (2.6)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from requests->folium) (1.22)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from requests->folium) (2019.3.9)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/DSX-Python35/lib/python3.5/site-packages (from jinja2>=2.9->folium) (1.0)
    [31mtensorflow 1.3.0 requires tensorflow-tensorboard<0.2.0,>=0.1.0, which is not installed.[0m



```python
#import liabriaries

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')
```

    Libraries imported.


# PART 1

**In the following code, we will read the html data from Wiki and clean the data.  The data can be found at https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M** 


```python
#import libraries
import pandas as pd
import numpy as np
```


```python
#use pandas to read in the html info
# create a dataframe from the first table
data_toronto_df = pd.read_html("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M", header = 0)[0]
```


```python
data_toronto_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>




```python
# preserve the order of the columns
columns = list(data_toronto_df.columns)

#reset the index to Borough
data_toronto_df = data_toronto_df.set_index('Borough')

#delete all rows whose Borough = Not Assigned
data_toronto_df = data_toronto_df.drop("Not assigned", axis = 0)

# reinstate Borough as column
data_toronto_df = data_toronto_df.reset_index()

```


```python
data_toronto_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Postcode</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>North York</td>
      <td>M3A</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North York</td>
      <td>M4A</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Downtown Toronto</td>
      <td>M5A</td>
      <td>Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Downtown Toronto</td>
      <td>M5A</td>
      <td>Regent Park</td>
    </tr>
    <tr>
      <th>4</th>
      <td>North York</td>
      <td>M6A</td>
      <td>Lawrence Heights</td>
    </tr>
  </tbody>
</table>
</div>




```python
#reinstate the orgignal order of the columns
data_toronto_df = data_toronto_df[columns]
data_toronto_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights</td>
    </tr>
  </tbody>
</table>
</div>




```python
#define a function to deal with the "Not assigned" Neighbourhoods 

def fun_replace(row):
    if row['Neighbourhood'] == 'Not assigned':
        row['Neighbourhood'] = row['Borough']
    
    return row['Neighbourhood']

#replace the values of Not Assigned in the Neighbourhoods with the values of the Boroughs    
data_toronto_df['Neighbourhood'] = data_toronto_df.apply(fun_replace, axis = 1) 
```


```python
#check there are no "Not assigned" neighborhoods
(data_toronto_df['Neighbourhood'] != 'Not assigned').all()
```




    True




```python
data_toronto_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_toronto_grouped_df = data_toronto_df.groupby('Postcode', as_index = False).agg(lambda x: ', '.join(set(x)))
```


```python
print("The number of rows in the cleaned datafrane is ", data_toronto_grouped_df.shape[0])
```

    The number of rows in the cleaned datafrane is  103


# PART 2


```python

import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_ #private data removed

# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_data_1.shape
```




    (103, 3)




```python
df_data_1 = df_data_1.rename(columns = {'Postal Code' : 'PostalCode'})
data_toronto_grouped_df = data_toronto_grouped_df.rename(columns = {'Postcode' : 'PostalCode'})
```


```python
data_toronto_grouped_df = data_toronto_grouped_df.merge(df_data_1, on = 'PostalCode')
```


```python
data_toronto_grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>West Hill, Guildwood, Morningside</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



# PART 3


```python
#create a new data frame with Bouroughs that contain Toronto string in them
neighbourhoods_of_Toronto_df = data_toronto_grouped_df[data_toronto_grouped_df['Borough'].str.contains('Toronto')]
```


```python
neighbourhoods_of_Toronto_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <th>41</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>42</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>India Bazaar, The Beaches West</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>43</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>44</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
    </tr>
  </tbody>
</table>
</div>



### Exploring venues in downtown Toronto


```python
#further restrict neighborhoods to DOwntown Toronto only. Create a new dataframe
neighbourhoods_of_Downtown_Toronto_df = neighbourhoods_of_Toronto_df [neighbourhoods_of_Toronto_df['Borough'].str.contains('Downtown Toronto')]
```


```python
neighbourhoods_of_Downtown_Toronto_df.reset_index(inplace = True, drop = True)
neighbourhoods_of_Downtown_Toronto_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>43.679563</td>
      <td>-79.377529</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>St. James Town, Cabbagetown</td>
      <td>43.667967</td>
      <td>-79.367675</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>43.665860</td>
      <td>-79.383160</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>43.657952</td>
      <td>-79.387383</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Richmond, King, Adelaide</td>
      <td>43.650571</td>
      <td>-79.384568</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Union Station, Toronto Islands, Harbourfront East</td>
      <td>43.640816</td>
      <td>-79.381752</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>43.647177</td>
      <td>-79.381576</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Victoria Hotel, Commerce Court</td>
      <td>43.648198</td>
      <td>-79.379817</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
      <td>43.662696</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Grange Park, Chinatown, Kensington Market</td>
      <td>43.653206</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront West, Bathurst Quay, Island airpo...</td>
      <td>43.628947</td>
      <td>-79.394420</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>43.646435</td>
      <td>-79.374846</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>Underground city, First Canadian Place</td>
      <td>43.648429</td>
      <td>-79.382280</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
    </tr>
  </tbody>
</table>
</div>



#### Use geopy library to get the latitude and longitude values of Downtown Toronto.

In order to define an instance of the geocoder, we need to define a user_agent. We will name our agent <em>dwntoro_explorer</em>, as shown below.


```python
address = 'Downtown Toronto, CA'

geolocator = Nominatim(user_agent="dwntoro_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Downtown Toronto are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Downtown Toronto are 43.655115, -79.380219.


#### Create a map of Downtown Toronto with neighborhoods superimposed on top.


```python
# create map of Downtown Toronto using latitude and longitude values
map_dwnttoronto = folium.Map(location=[latitude, longitude], zoom_start=12, zoom_control = False)

n_dwnt_tor = neighbourhoods_of_Downtown_Toronto_df
# add markers to map
for lat, lng, borough, neighborhood in zip(n_dwnt_tor['Latitude'], n_dwnt_tor['Longitude'], n_dwnt_tor['Borough'], n_dwnt_tor['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_dwnttoronto)  
    
map_dwnttoronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgCiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgTF9OT19UT1VDSCA9IGZhbHNlOwogICAgICAgICAgICBMX0RJU0FCTEVfM0QgPSBmYWxzZTsKICAgICAgICA8L3NjcmlwdD4KICAgIAogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjQuMC9kaXN0L2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY29kZS5qcXVlcnkuY29tL2pxdWVyeS0xLjEyLjQubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5qcyI+PC9zY3JpcHQ+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjQuMC9kaXN0L2xlYWZsZXQuY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9mb250LWF3ZXNvbWUvNC42LjMvY3NzL2ZvbnQtYXdlc29tZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC4yL2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2Nkbi5naXRoYWNrLmNvbS9weXRob24tdmlzdWFsaXphdGlvbi9mb2xpdW0vbWFzdGVyL2ZvbGl1bS90ZW1wbGF0ZXMvbGVhZmxldC5hd2Vzb21lLnJvdGF0ZS5jc3MiLz4KICAgIDxzdHlsZT5odG1sLCBib2R5IHt3aWR0aDogMTAwJTtoZWlnaHQ6IDEwMCU7bWFyZ2luOiAwO3BhZGRpbmc6IDA7fTwvc3R5bGU+CiAgICA8c3R5bGU+I21hcCB7cG9zaXRpb246YWJzb2x1dGU7dG9wOjA7Ym90dG9tOjA7cmlnaHQ6MDtsZWZ0OjA7fTwvc3R5bGU+CiAgICAKICAgICAgICAgICAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwKICAgICAgICAgICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgICAgICAgICAgPHN0eWxlPgogICAgICAgICAgICAgICAgI21hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZiB7CiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHJlbGF0aXZlOwogICAgICAgICAgICAgICAgICAgIHdpZHRoOiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgICAgICB0b3A6IDAuMCU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgCjwvaGVhZD4KPGJvZHk+ICAgIAogICAgCiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImZvbGl1bS1tYXAiIGlkPSJtYXBfMzhlYWY2NjFjMzFmNDFhNjllZWE5Yjk5MGE2MTBlMGYiID48L2Rpdj4KICAgICAgICAKPC9ib2R5Pgo8c2NyaXB0PiAgICAKICAgIAogICAgICAgICAgICB2YXIgbWFwXzM4ZWFmNjYxYzMxZjQxYTY5ZWVhOWI5OTBhNjEwZTBmID0gTC5tYXAoCiAgICAgICAgICAgICAgICAibWFwXzM4ZWFmNjYxYzMxZjQxYTY5ZWVhOWI5OTBhNjEwZTBmIiwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBjZW50ZXI6IFs0My42NTUxMTUsIC03OS4zODAyMTldLAogICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcsCiAgICAgICAgICAgICAgICAgICAgem9vbTogMTIsCiAgICAgICAgICAgICAgICAgICAgem9vbUNvbnRyb2w6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgIHByZWZlckNhbnZhczogZmFsc2UsCiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICk7CgogICAgICAgICAgICAKCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfZDNiYjYyMmM2ZmE5NGYxYjkxYzQ2OGI5Yzk5NzkzZTcgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICJodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZyIsCiAgICAgICAgICAgICAgICB7ImF0dHJpYnV0aW9uIjogIkRhdGEgYnkgXHUwMDI2Y29weTsgXHUwMDNjYSBocmVmPVwiaHR0cDovL29wZW5zdHJlZXRtYXAub3JnXCJcdTAwM2VPcGVuU3RyZWV0TWFwXHUwMDNjL2FcdTAwM2UsIHVuZGVyIFx1MDAzY2EgaHJlZj1cImh0dHA6Ly93d3cub3BlbnN0cmVldG1hcC5vcmcvY29weXJpZ2h0XCJcdTAwM2VPRGJMXHUwMDNjL2FcdTAwM2UuIiwgImRldGVjdFJldGluYSI6IGZhbHNlLCAibWF4TmF0aXZlWm9vbSI6IDE4LCAibWF4Wm9vbSI6IDE4LCAibWluWm9vbSI6IDAsICJub1dyYXAiOiBmYWxzZSwgIm9wYWNpdHkiOiAxLCAic3ViZG9tYWlucyI6ICJhYmMiLCAidG1zIjogZmFsc2V9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzM4ZWFmNjYxYzMxZjQxYTY5ZWVhOWI5OTBhNjEwZTBmKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yY2Y3MDk3MTZhM2I0OTFhYTUzYjljMzNjMDllNDJjOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsIC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiYmx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzM4ZWFmNjYxYzMxZjQxYTY5ZWVhOWI5OTBhNjEwZTBmKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9jZTg3MzlhYTE4MmM0M2FmODUxMmU2MWQxOTNiMDUwOSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNjA5N2ZlMWZhMWY4NGEwNmI4YTk2NzIwYmI5M2Y5NDEgPSAkKGA8ZGl2IGlkPSJodG1sXzYwOTdmZTFmYTFmODRhMDZiOGE5NjcyMGJiOTNmOTQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZSwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9jZTg3MzlhYTE4MmM0M2FmODUxMmU2MWQxOTNiMDUwOS5zZXRDb250ZW50KGh0bWxfNjA5N2ZlMWZhMWY4NGEwNmI4YTk2NzIwYmI5M2Y5NDEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzJjZjcwOTcxNmEzYjQ5MWFhNTNiOWMzM2MwOWU0MmM5LmJpbmRQb3B1cChwb3B1cF9jZTg3MzlhYTE4MmM0M2FmODUxMmU2MWQxOTNiMDUwOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGMyNmJhN2YyMjA5NGRlM2I0Njg3M2E2NGZhMGZhMzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsIC03OS4zNjc2NzUzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzhlYWY2NjFjMzFmNDFhNjllZWE5Yjk5MGE2MTBlMGYpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzQwYTE1ODBhNWY2OTQxNDBhMjQ4OWJkZWJlMTRkOTc1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9iZGM2ZDljZWFhMGE0ZThkYjAwMmFjMWM3ZDhlOTk5NSA9ICQoYDxkaXYgaWQ9Imh0bWxfYmRjNmQ5Y2VhYTBhNGU4ZGIwMDJhYzFjN2Q4ZTk5OTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBDYWJiYWdldG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF80MGExNTgwYTVmNjk0MTQwYTI0ODliZGViZTE0ZDk3NS5zZXRDb250ZW50KGh0bWxfYmRjNmQ5Y2VhYTBhNGU4ZGIwMDJhYzFjN2Q4ZTk5OTUpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2RjMjZiYTdmMjIwOTRkZTNiNDY4NzNhNjRmYTBmYTMzLmJpbmRQb3B1cChwb3B1cF80MGExNTgwYTVmNjk0MTQwYTI0ODliZGViZTE0ZDk3NSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGI2MjdkODljNWZmNDg2N2ExNDljZTRlZTk5ZWQwNDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LCAtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZik7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTBjZTg4MDUwNzJmNDRiY2I2ZmZkZWNjNDZiMjk2NzEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzgyZjYwNmVmNjkwYzRhNDE4YTA1NjQ2MTVmZWYxMzQ3ID0gJChgPGRpdiBpZD0iaHRtbF84MmY2MDZlZjY5MGM0YTQxOGEwNTY0NjE1ZmVmMTM0NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXksIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZTBjZTg4MDUwNzJmNDRiY2I2ZmZkZWNjNDZiMjk2NzEuc2V0Q29udGVudChodG1sXzgyZjYwNmVmNjkwYzRhNDE4YTA1NjQ2MTVmZWYxMzQ3KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl80YjYyN2Q4OWM1ZmY0ODY3YTE0OWNlNGVlOTllZDA0MC5iaW5kUG9wdXAocG9wdXBfZTBjZTg4MDUwNzJmNDRiY2I2ZmZkZWNjNDZiMjk2NzEpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZmMWYzYjA4ZjZiMTQzM2Y4YjI1MWJjZWU0ZjAxZWIzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwgLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZik7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfMzIyNjQ3NDA4N2EyNGRjNmJmMzE2NmI4NjY5MmE0ZjkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzAzNzg2YTFmNjVmNDQ4MGE4NjZmZDEzOGNmZjNkMDZiID0gJChgPGRpdiBpZD0iaHRtbF8wMzc4NmExZjY1ZjQ0ODBhODY2ZmQxMzhjZmYzZDA2YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmVnZW50IFBhcmssIEhhcmJvdXJmcm9udCwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF8zMjI2NDc0MDg3YTI0ZGM2YmYzMTY2Yjg2NjkyYTRmOS5zZXRDb250ZW50KGh0bWxfMDM3ODZhMWY2NWY0NDgwYTg2NmZkMTM4Y2ZmM2QwNmIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2ZmMWYzYjA4ZjZiMTQzM2Y4YjI1MWJjZWU0ZjAxZWIzLmJpbmRQb3B1cChwb3B1cF8zMjI2NDc0MDg3YTI0ZGM2YmYzMTY2Yjg2NjkyYTRmOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDllNzAyNTcwYzhhNDU0MDkzN2FlNWQ0Y2Q3MzZiZGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LCAtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZik7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDAwNjQ2NWRjNjgyNDdiOGEzNzZmY2E0ZGE1NTZlYjcgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Q2ZmJmYTY1YzRlNjQ1ZDA4YTJmOGI1YjIxMzJjZWJkID0gJChgPGRpdiBpZD0iaHRtbF9kNmZiZmE2NWM0ZTY0NWQwOGEyZjhiNWIyMTMyY2ViZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnllcnNvbiwgR2FyZGVuIERpc3RyaWN0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2QwMDY0NjVkYzY4MjQ3YjhhMzc2ZmNhNGRhNTU2ZWI3LnNldENvbnRlbnQoaHRtbF9kNmZiZmE2NWM0ZTY0NWQwOGEyZjhiNWIyMTMyY2ViZCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMDllNzAyNTcwYzhhNDU0MDkzN2FlNWQ0Y2Q3MzZiZGEuYmluZFBvcHVwKHBvcHVwX2QwMDY0NjVkYzY4MjQ3YjhhMzc2ZmNhNGRhNTU2ZWI3KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZjllZjliOTAzMWI0YTg3OGI1MjRlYzRjOWMzMDZiNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksIC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzhlYWY2NjFjMzFmNDFhNjllZWE5Yjk5MGE2MTBlMGYpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzQzYTQ4NWY0ZTgzMzQ5NjViOTA1MTQ4NmM1ZDZjN2M1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9hOTBkYWI0YzIyNzg0NzNmODA3NTQwNmVjNTUwNDJlNiA9ICQoYDxkaXYgaWQ9Imh0bWxfYTkwZGFiNGMyMjc4NDczZjgwNzU0MDZlYzU1MDQyZTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzQzYTQ4NWY0ZTgzMzQ5NjViOTA1MTQ4NmM1ZDZjN2M1LnNldENvbnRlbnQoaHRtbF9hOTBkYWI0YzIyNzg0NzNmODA3NTQwNmVjNTUwNDJlNik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOWY5ZWY5YjkwMzFiNGE4NzhiNTI0ZWM0YzljMzA2YjQuYmluZFBvcHVwKHBvcHVwXzQzYTQ4NWY0ZTgzMzQ5NjViOTA1MTQ4NmM1ZDZjN2M1KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMTA1MWJkMmEyNTI0NDNjYjViMTYyNzFkYzAxMzQ0YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwgLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZik7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNzUwN2I1ZWE0NGU4NGIxY2EyYjcxMDY2ZWJhYzBiNzggPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzgyMGUzMTQ4MTRkMzQ4ZDNiMmEzZTEwMmViODNlMGQ1ID0gJChgPGRpdiBpZD0iaHRtbF84MjBlMzE0ODE0ZDM0OGQzYjJhM2UxMDJlYjgzZTBkNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNzUwN2I1ZWE0NGU4NGIxY2EyYjcxMDY2ZWJhYzBiNzguc2V0Q29udGVudChodG1sXzgyMGUzMTQ4MTRkMzQ4ZDNiMmEzZTEwMmViODNlMGQ1KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9kMTA1MWJkMmEyNTI0NDNjYjViMTYyNzFkYzAxMzQ0Yi5iaW5kUG9wdXAocG9wdXBfNzUwN2I1ZWE0NGU4NGIxY2EyYjcxMDY2ZWJhYzBiNzgpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjNWYxM2FhYWFlMjQ3YzFhNWEyNDE2ZTZmZmMzZTE0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwgLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZik7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTYzOTBiNmIwY2NjNDAyMWI3MTI1ZWFlNzg2ZWY4N2EgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzQyZWE4OGUyZTU4YzRhMzBhMjI5ZmFhNTI0ZDIwYjQ5ID0gJChgPGRpdiBpZD0iaHRtbF80MmVhODhlMmU1OGM0YTMwYTIyOWZhYTUyNGQyMGI0OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBCYXkgU3RyZWV0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2U2MzkwYjZiMGNjYzQwMjFiNzEyNWVhZTc4NmVmODdhLnNldENvbnRlbnQoaHRtbF80MmVhODhlMmU1OGM0YTMwYTIyOWZhYTUyNGQyMGI0OSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNGM1ZjEzYWFhYWUyNDdjMWE1YTI0MTZlNmZmYzNlMTQuYmluZFBvcHVwKHBvcHVwX2U2MzkwYjZiMGNjYzQwMjFiNzEyNWVhZTc4NmVmODdhKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNDMzMGFlMWNjNzc0ZGI0YjYxMDBjM2Y3MDFhZjNlNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLCAtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiYmx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzM4ZWFmNjYxYzMxZjQxYTY5ZWVhOWI5OTBhNjEwZTBmKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iZWYwNzJkODVlMjM0YzM1OWNhZDE3ZDZmZjg2YjkyNiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfY2E4YWUwMTFlNGE3NDhlYmJiYjBlOGE3MjY1N2IyOTYgPSAkKGA8ZGl2IGlkPSJodG1sX2NhOGFlMDExZTRhNzQ4ZWJiYmIwZThhNzI2NTdiMjk2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SaWNobW9uZCwgS2luZywgQWRlbGFpZGUsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYmVmMDcyZDg1ZTIzNGMzNTljYWQxN2Q2ZmY4NmI5MjYuc2V0Q29udGVudChodG1sX2NhOGFlMDExZTRhNzQ4ZWJiYmIwZThhNzI2NTdiMjk2KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9iNDMzMGFlMWNjNzc0ZGI0YjYxMDBjM2Y3MDFhZjNlNC5iaW5kUG9wdXAocG9wdXBfYmVmMDcyZDg1ZTIzNGMzNTljYWQxN2Q2ZmY4NmI5MjYpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdmODhjMDY0YTVkOTQxNTZhNmM2MWNkZjE3ZjliZWU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywgLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzhlYWY2NjFjMzFmNDFhNjllZWE5Yjk5MGE2MTBlMGYpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzhhMTEyZjRmOTEwZjQ1NmVhNGE4YzJlYWE2YTQxYzgwID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8yODM0MGI2YTViNmI0MTM0OGFjYzk3Mjk2YzRlOTU4ZCA9ICQoYDxkaXYgaWQ9Imh0bWxfMjgzNDBiNmE1YjZiNDEzNDhhY2M5NzI5NmM0ZTk1OGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaW9uIFN0YXRpb24sIFRvcm9udG8gSXNsYW5kcywgSGFyYm91cmZyb250IEVhc3QsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOGExMTJmNGY5MTBmNDU2ZWE0YThjMmVhYTZhNDFjODAuc2V0Q29udGVudChodG1sXzI4MzQwYjZhNWI2YjQxMzQ4YWNjOTcyOTZjNGU5NThkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl83Zjg4YzA2NGE1ZDk0MTU2YTZjNjFjZGYxN2Y5YmVlOS5iaW5kUG9wdXAocG9wdXBfOGExMTJmNGY5MTBmNDU2ZWE0YThjMmVhYTZhNDFjODApCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIzYTJlYmRkMGYxYTRhNjFhNzgzMGQ1MDc3M2I5NzgwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3MTc2OCwgLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzhlYWY2NjFjMzFmNDFhNjllZWE5Yjk5MGE2MTBlMGYpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2NiZDZjZmQzZDIwNTRmMzFhMzZiNTdjMTU1NDZhNGE0ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF9mOTlhNGU0ZjU1NDM0MTJmYTlhMmM4OWFhZmM2ZWM5YSA9ICQoYDxkaXYgaWQ9Imh0bWxfZjk5YTRlNGY1NTQzNDEyZmE5YTJjODlhYWZjNmVjOWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRvcm9udG8gRG9taW5pb24gQ2VudHJlLCBEZXNpZ24gRXhjaGFuZ2UsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfY2JkNmNmZDNkMjA1NGYzMWEzNmI1N2MxNTU0NmE0YTQuc2V0Q29udGVudChodG1sX2Y5OWE0ZTRmNTU0MzQxMmZhOWEyYzg5YWFmYzZlYzlhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8yM2EyZWJkZDBmMWE0YTYxYTc4MzBkNTA3NzNiOTc4MC5iaW5kUG9wdXAocG9wdXBfY2JkNmNmZDNkMjA1NGYzMWEzNmI1N2MxNTU0NmE0YTQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQyMTVhYWU3MWZkNTRiMWFhZjA0ZWNjZjlhMzExMjExID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4MTk4NSwgLTc5LjM3OTgxNjkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzhlYWY2NjFjMzFmNDFhNjllZWE5Yjk5MGE2MTBlMGYpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2ZkODYyMDc1YWE0NjRkZTM5ZjA2MTZmMDg3M2VhMmJlID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF83ZjdiMTg0OGRlNDA0OGU1Yjk0YWI1Yjk4OGM2YzRlYiA9ICQoYDxkaXYgaWQ9Imh0bWxfN2Y3YjE4NDhkZTQwNDhlNWI5NGFiNWI5ODhjNmM0ZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZpY3RvcmlhIEhvdGVsLCBDb21tZXJjZSBDb3VydCwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9mZDg2MjA3NWFhNDY0ZGUzOWYwNjE2ZjA4NzNlYTJiZS5zZXRDb250ZW50KGh0bWxfN2Y3YjE4NDhkZTQwNDhlNWI5NGFiNWI5ODhjNmM0ZWIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzQyMTVhYWU3MWZkNTRiMWFhZjA0ZWNjZjlhMzExMjExLmJpbmRQb3B1cChwb3B1cF9mZDg2MjA3NWFhNDY0ZGUzOWYwNjE2ZjA4NzNlYTJiZSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGNmYjUzMTkwOTBlNDkxNjkxOTg3ZjBmYWRhMzA4YTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LCAtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiYmx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzM4ZWFmNjYxYzMxZjQxYTY5ZWVhOWI5OTBhNjEwZTBmKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF85YjFmNmI4MWZiYmI0YWZjOTRmYzcwZjBlZDEzNDA2YyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOTYyYzliNTU3MTYzNDIzNGEzMTljZWY3MTllNmZlMjIgPSAkKGA8ZGl2IGlkPSJodG1sXzk2MmM5YjU1NzE2MzQyMzRhMzE5Y2VmNzE5ZTZmZTIyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3JkLCBVbml2ZXJzaXR5IG9mIFRvcm9udG8sIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOWIxZjZiODFmYmJiNGFmYzk0ZmM3MGYwZWQxMzQwNmMuc2V0Q29udGVudChodG1sXzk2MmM5YjU1NzE2MzQyMzRhMzE5Y2VmNzE5ZTZmZTIyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8wY2ZiNTMxOTA5MGU0OTE2OTE5ODdmMGZhZGEzMDhhMS5iaW5kUG9wdXAocG9wdXBfOWIxZjZiODFmYmJiNGFmYzk0ZmM3MGYwZWQxMzQwNmMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFkYjc3OTUwOGNjNjQ1YmFiNTllYmYxZmY1ZjZiMzViID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywgLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZik7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDZjNmY2YWFiNmRjNDkyNjgyOGMxNWM1ODI3YTc3ZjQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzNmYzNjZjAxOGRhOTRjYmZiZmM4MDgwNWVkYjcyYjRiID0gJChgPGRpdiBpZD0iaHRtbF8zZmMzY2YwMThkYTk0Y2JmYmZjODA4MDVlZGI3MmI0YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R3JhbmdlIFBhcmssIENoaW5hdG93biwgS2Vuc2luZ3RvbiBNYXJrZXQsIERvd250b3duIFRvcm9udG88L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZDZjNmY2YWFiNmRjNDkyNjgyOGMxNWM1ODI3YTc3ZjQuc2V0Q29udGVudChodG1sXzNmYzNjZjAxOGRhOTRjYmZiZmM4MDgwNWVkYjcyYjRiKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8xZGI3Nzk1MDhjYzY0NWJhYjU5ZWJmMWZmNWY2YjM1Yi5iaW5kUG9wdXAocG9wdXBfZDZjNmY2YWFiNmRjNDkyNjgyOGMxNWM1ODI3YTc3ZjQpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y3MmEzMzU1ZTBhOTRjM2ZiZTM3OWRlZThiNWNjYzc5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywgLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZik7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYWRiMWQxNjM2MmM4NDJlODgzMmYxZGFlYzNkZTY5MDIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2M3ZDM4YzE3OTkwYzRkN2NiYzY4MDQwNjlkMzUzN2YwID0gJChgPGRpdiBpZD0iaHRtbF9jN2QzOGMxNzk5MGM0ZDdjYmM2ODA0MDY5ZDM1MzdmMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IFdlc3QsIEJhdGh1cnN0IFF1YXksIElzbGFuZCBhaXJwb3J0LCBSYWlsd2F5IExhbmRzLCBLaW5nIGFuZCBTcGFkaW5hLCBTb3V0aCBOaWFnYXJhLCBDTiBUb3dlciwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hZGIxZDE2MzYyYzg0MmU4ODMyZjFkYWVjM2RlNjkwMi5zZXRDb250ZW50KGh0bWxfYzdkMzhjMTc5OTBjNGQ3Y2JjNjgwNDA2OWQzNTM3ZjApOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2Y3MmEzMzU1ZTBhOTRjM2ZiZTM3OWRlZThiNWNjYzc5LmJpbmRQb3B1cChwb3B1cF9hZGIxZDE2MzYyYzg0MmU4ODMyZjFkYWVjM2RlNjkwMikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDUxYTE2NzJlMjg3NGJmNjhjYjlmYTI1NWMxNDYzY2YgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLCAtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogImJsdWUiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF8zOGVhZjY2MWMzMWY0MWE2OWVlYTliOTkwYTYxMGUwZik7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNmQyZDg0NjRiZTc0NDc0NWE1ZWIwNzBkMWE1NGU3MjkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzcwZDQ5OTljYWIwZDRkODBiZjFmOTI2MTNjYzFlYWNiID0gJChgPGRpdiBpZD0iaHRtbF83MGQ0OTk5Y2FiMGQ0ZDgwYmYxZjkyNjEzY2MxZWFjYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3RuIEEgUE8gQm94ZXMgMjUgVGhlIEVzcGxhbmFkZSwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF82ZDJkODQ2NGJlNzQ0NzQ1YTVlYjA3MGQxYTU0ZTcyOS5zZXRDb250ZW50KGh0bWxfNzBkNDk5OWNhYjBkNGQ4MGJmMWY5MjYxM2NjMWVhY2IpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzQ1MWExNjcyZTI4NzRiZjY4Y2I5ZmEyNTVjMTQ2M2NmLmJpbmRQb3B1cChwb3B1cF82ZDJkODQ2NGJlNzQ0NzQ1YTVlYjA3MGQxYTU0ZTcyOSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTE0YjUwMjEwN2ViNDk5M2FmYjBmNTU2ZGEwZTNiMjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLCAtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiYmx1ZSIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwXzM4ZWFmNjYxYzMxZjQxYTY5ZWVhOWI5OTBhNjEwZTBmKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF85NTk4ZWI4NTQ4Yzg0MTAwOTFhNjY2YTk2ZmEyODY0NiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNDczNzJkNWYzNTI1NDE0ZThlMTJkYWI3NTM5OTAzOWEgPSAkKGA8ZGl2IGlkPSJodG1sXzQ3MzcyZDVmMzUyNTQxNGU4ZTEyZGFiNzUzOTkwMzlhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5VbmRlcmdyb3VuZCBjaXR5LCBGaXJzdCBDYW5hZGlhbiBQbGFjZSwgRG93bnRvd24gVG9yb250bzwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF85NTk4ZWI4NTQ4Yzg0MTAwOTFhNjY2YTk2ZmEyODY0Ni5zZXRDb250ZW50KGh0bWxfNDczNzJkNWYzNTI1NDE0ZThlMTJkYWI3NTM5OTAzOWEpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2UxNGI1MDIxMDdlYjQ5OTNhZmIwZjU1NmRhMGUzYjIyLmJpbmRQb3B1cChwb3B1cF85NTk4ZWI4NTQ4Yzg0MTAwOTFhNjY2YTk2ZmEyODY0NikKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODc5NzhkMGI2MWQ5NDgyZTljZWQ4YWRjYTE3ZmVlZTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njk1NDIsIC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICJibHVlIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzhlYWY2NjFjMzFmNDFhNjllZWE5Yjk5MGE2MTBlMGYpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzZkZmE3N2JlZmJkNzQ0NzdiNmE1YTlkODg3YTczYWVmID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wODlmZjg2Y2Q3NWQ0NTg2OWMxYWI5N2UzZmRmYjEyYyA9ICQoYDxkaXYgaWQ9Imh0bWxfMDg5ZmY4NmNkNzVkNDU4NjljMWFiOTdlM2ZkZmIxMmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocmlzdGllLCBEb3dudG93biBUb3JvbnRvPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzZkZmE3N2JlZmJkNzQ0NzdiNmE1YTlkODg3YTczYWVmLnNldENvbnRlbnQoaHRtbF8wODlmZjg2Y2Q3NWQ0NTg2OWMxYWI5N2UzZmRmYjEyYyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfODc5NzhkMGI2MWQ5NDgyZTljZWQ4YWRjYTE3ZmVlZTAuYmluZFBvcHVwKHBvcHVwXzZkZmE3N2JlZmJkNzQ0NzdiNmE1YTlkODg3YTczYWVmKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



#### Use the Foursquare API to explore the neighborhoods and segment them.


```python
CLIENT_ID = '...' # your Foursquare ID, private data removed
CLIENT_SECRET = '...' # your Foursquare Secret, private data removed
VERSION = '20190605' # Foursquare API version
```

Create a function to extract venues for all neighbourhoods of Downtown Toronto


```python
# define LIMIT of the number of venues, set it at 50
# define radius from the center of Downtown Toronto, set it at 500 meters

LIMIT = 50
radius = 500
```


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python

dwnt_tor_data = neighbourhoods_of_Downtown_Toronto_df
dwntToronto_venues = getNearbyVenues(names=dwnt_tor_data['Neighbourhood'],
                                   latitudes=dwnt_tor_data['Latitude'],
                                   longitudes=dwnt_tor_data['Longitude']
                                  )
```

    Rosedale
    St. James Town, Cabbagetown
    Church and Wellesley
    Regent Park, Harbourfront
    Ryerson, Garden District
    St. James Town
    Berczy Park
    Central Bay Street
    Richmond, King, Adelaide
    Union Station, Toronto Islands, Harbourfront East
    Toronto Dominion Centre, Design Exchange
    Victoria Hotel, Commerce Court
    Harbord, University of Toronto
    Grange Park, Chinatown, Kensington Market
    Harbourfront West, Bathurst Quay, Island airport, Railway Lands, King and Spadina, South Niagara, CN Tower
    Stn A PO Boxes 25 The Esplanade
    Underground city, First Canadian Place
    Christie


Let's check how many venues were returned for each neighborhood


```python
dwntToronto_venues.groupby('Neighbourhood').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighbourhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Berczy Park</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Central Bay Street</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Christie</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Church and Wellesley</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Grange Park, Chinatown, Kensington Market</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Harbord, University of Toronto</th>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>Harbourfront West, Bathurst Quay, Island airport, Railway Lands, King and Spadina, South Niagara, CN Tower</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Regent Park, Harbourfront</th>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Richmond, King, Adelaide</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Rosedale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Ryerson, Garden District</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>St. James Town</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>St. James Town, Cabbagetown</th>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
    </tr>
    <tr>
      <th>Stn A PO Boxes 25 The Esplanade</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Toronto Dominion Centre, Design Exchange</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Underground city, First Canadian Place</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Union Station, Toronto Islands, Harbourfront East</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Victoria Hotel, Commerce Court</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



##  Analyze Each Neighborhood


```python
# one hot encoding
dwntToronto_venues_onehot = pd.get_dummies(dwntToronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
dwntToronto_venues_onehot['Neighbourhood'] = dwntToronto_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [dwntToronto_venues_onehot.columns[-1]] + list(dwntToronto_venues_onehot.columns[:-1])
dwntToronto_venues_onehot = dwntToronto_venues_onehot[fixed_columns]

dwntToronto_venues_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>...</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rosedale</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rosedale</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rosedale</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosedale</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>St. James Town, Cabbagetown</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 184 columns</p>
</div>



#### Grouping rows by neighbourhood  by taking the mean of the frequency of occurrence of each category


```python
dwnttoronto_grouped = dwntToronto_venues_onehot.groupby('Neighbourhood').mean().reset_index()
dwnttoronto_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>...</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berczy Park</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Central Bay Street</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Christie</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Church and Wellesley</td>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Grange Park, Chinatown, Kensington Market</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>0.000000</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Harbord, University of Toronto</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.029412</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Harbourfront West, Bathurst Quay, Island airpo...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.133333</td>
      <td>0.133333</td>
      <td>0.133333</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Regent Park, Harbourfront</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.020833</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Richmond, King, Adelaide</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.06</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rosedale</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ryerson, Garden District</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>St. James Town</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>St. James Town, Cabbagetown</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Underground city, First Canadian Place</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.06</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Union Station, Toronto Islands, Harbourfront East</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Victoria Hotel, Commerce Court</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.04</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>18 rows × 184 columns</p>
</div>



## Clustering Neighborhoods

The experiment is to cluster the neighbourhoods into few clusters.  While there are simpler ways to do it, the goal of this exercise is to use the clustering algorithm to visualize the neighborhoods of the downtown Toronto with the most venues, doesn't matter what they are.
The ultimate goal is to explore the zip codes (postal codes) belonging to the most populous cluster.
**Note:** The set up in the last part of the code is somewhat different from the set up of the Lab exercise for NY city venues


```python

# set number of clusters
kclusters = 3

dwnttoronto_grouped_clustering = dwnttoronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(dwnttoronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:20] 
```




    array([1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)




```python
dwnttoronto_grouped_clustering.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>...</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.02</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 183 columns</p>
</div>




```python
dwnttoronto_grouped.insert(0, 'Cluster Labels', kmeans.labels_)
```


```python
dwnttoronto_combined_df = neighbourhoods_of_Downtown_Toronto_df.merge(dwnttoronto_grouped, on = 'Neighbourhood')
```


```python
columns_list = ['PostalCode', 'Borough', 'Neighbourhood', 'Latitude', 'Longitude', 'Cluster Labels']
dwnttoronto_combined_df = dwnttoronto_combined_df[columns_list]
dwnttoronto_combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>43.679563</td>
      <td>-79.377529</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>St. James Town, Cabbagetown</td>
      <td>43.667967</td>
      <td>-79.367675</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>43.665860</td>
      <td>-79.383160</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
      <td>43.657162</td>
      <td>-79.378937</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Map clusters


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=10, zoom_control = False)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dwnttoronto_combined_df['Latitude'], dwnttoronto_combined_df['Longitude'], dwnttoronto_combined_df['Neighbourhood'], dwnttoronto_combined_df['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgCiAgICAgICAgPHNjcmlwdD4KICAgICAgICAgICAgTF9OT19UT1VDSCA9IGZhbHNlOwogICAgICAgICAgICBMX0RJU0FCTEVfM0QgPSBmYWxzZTsKICAgICAgICA8L3NjcmlwdD4KICAgIAogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjQuMC9kaXN0L2xlYWZsZXQuanMiPjwvc2NyaXB0PgogICAgPHNjcmlwdCBzcmM9Imh0dHBzOi8vY29kZS5qcXVlcnkuY29tL2pxdWVyeS0xLjEyLjQubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9qcy9ib290c3RyYXAubWluLmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5qcyI+PC9zY3JpcHQ+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vbGVhZmxldEAxLjQuMC9kaXN0L2xlYWZsZXQuY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vYm9vdHN0cmFwLzMuMi4wL2Nzcy9ib290c3RyYXAubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLXRoZW1lLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9mb250LWF3ZXNvbWUvNC42LjMvY3NzL2ZvbnQtYXdlc29tZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vY2RuanMuY2xvdWRmbGFyZS5jb20vYWpheC9saWJzL0xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLzIuMC4yL2xlYWZsZXQuYXdlc29tZS1tYXJrZXJzLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL3Jhd2Nkbi5naXRoYWNrLmNvbS9weXRob24tdmlzdWFsaXphdGlvbi9mb2xpdW0vbWFzdGVyL2ZvbGl1bS90ZW1wbGF0ZXMvbGVhZmxldC5hd2Vzb21lLnJvdGF0ZS5jc3MiLz4KICAgIDxzdHlsZT5odG1sLCBib2R5IHt3aWR0aDogMTAwJTtoZWlnaHQ6IDEwMCU7bWFyZ2luOiAwO3BhZGRpbmc6IDA7fTwvc3R5bGU+CiAgICA8c3R5bGU+I21hcCB7cG9zaXRpb246YWJzb2x1dGU7dG9wOjA7Ym90dG9tOjA7cmlnaHQ6MDtsZWZ0OjA7fTwvc3R5bGU+CiAgICAKICAgICAgICAgICAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwKICAgICAgICAgICAgICAgIGluaXRpYWwtc2NhbGU9MS4wLCBtYXhpbXVtLXNjYWxlPTEuMCwgdXNlci1zY2FsYWJsZT1ubyIgLz4KICAgICAgICAgICAgPHN0eWxlPgogICAgICAgICAgICAgICAgI21hcF9kNTk2NWQ2OGVhYzg0ZWVkYWEzNzc3NjMwN2EyZTUwYSB7CiAgICAgICAgICAgICAgICAgICAgcG9zaXRpb246IHJlbGF0aXZlOwogICAgICAgICAgICAgICAgICAgIHdpZHRoOiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICAgICAgbGVmdDogMC4wJTsKICAgICAgICAgICAgICAgICAgICB0b3A6IDAuMCU7CiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgIDwvc3R5bGU+CiAgICAgICAgCjwvaGVhZD4KPGJvZHk+ICAgIAogICAgCiAgICAgICAgICAgIDxkaXYgY2xhc3M9ImZvbGl1bS1tYXAiIGlkPSJtYXBfZDU5NjVkNjhlYWM4NGVlZGFhMzc3NzYzMDdhMmU1MGEiID48L2Rpdj4KICAgICAgICAKPC9ib2R5Pgo8c2NyaXB0PiAgICAKICAgIAogICAgICAgICAgICB2YXIgbWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhID0gTC5tYXAoCiAgICAgICAgICAgICAgICAibWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhIiwKICAgICAgICAgICAgICAgIHsKICAgICAgICAgICAgICAgICAgICBjZW50ZXI6IFs0My42NTUxMTUsIC03OS4zODAyMTldLAogICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcsCiAgICAgICAgICAgICAgICAgICAgem9vbTogMTAsCiAgICAgICAgICAgICAgICAgICAgem9vbUNvbnRyb2w6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgIHByZWZlckNhbnZhczogZmFsc2UsCiAgICAgICAgICAgICAgICB9CiAgICAgICAgICAgICk7CgogICAgICAgICAgICAKCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfMzRkMmViYjJiYTZmNDA3YzkxZWIxNDZjNWI2Y2I0ZjYgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICJodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZyIsCiAgICAgICAgICAgICAgICB7ImF0dHJpYnV0aW9uIjogIkRhdGEgYnkgXHUwMDI2Y29weTsgXHUwMDNjYSBocmVmPVwiaHR0cDovL29wZW5zdHJlZXRtYXAub3JnXCJcdTAwM2VPcGVuU3RyZWV0TWFwXHUwMDNjL2FcdTAwM2UsIHVuZGVyIFx1MDAzY2EgaHJlZj1cImh0dHA6Ly93d3cub3BlbnN0cmVldG1hcC5vcmcvY29weXJpZ2h0XCJcdTAwM2VPRGJMXHUwMDNjL2FcdTAwM2UuIiwgImRldGVjdFJldGluYSI6IGZhbHNlLCAibWF4TmF0aXZlWm9vbSI6IDE4LCAibWF4Wm9vbSI6IDE4LCAibWluWm9vbSI6IDAsICJub1dyYXAiOiBmYWxzZSwgIm9wYWNpdHkiOiAxLCAic3ViZG9tYWlucyI6ICJhYmMiLCAidG1zIjogZmFsc2V9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NDY5NzU5ODk3ODA0MWNiOWE0NmFmOTAyZThhZjkxNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsIC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwZmZiNCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODBmZmI0IiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9iM2I5N2EwZGQ1M2M0MDg4OGU5OTZiYTQ1MjlmOWQ4MiA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZDZiOTI1Njg1NDgyNGQwZjlmZmVjYTNlMjA4NzEyN2MgPSAkKGA8ZGl2IGlkPSJodG1sX2Q2YjkyNTY4NTQ4MjRkMGY5ZmZlY2EzZTIwODcxMjdjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZSBDbHVzdGVyIDI8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYjNiOTdhMGRkNTNjNDA4ODhlOTk2YmE0NTI5ZjlkODIuc2V0Q29udGVudChodG1sX2Q2YjkyNTY4NTQ4MjRkMGY5ZmZlY2EzZTIwODcxMjdjKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl81NDY5NzU5ODk3ODA0MWNiOWE0NmFmOTAyZThhZjkxNi5iaW5kUG9wdXAocG9wdXBfYjNiOTdhMGRkNTNjNDA4ODhlOTk2YmE0NTI5ZjlkODIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MzZmUwZTQwMDM3YjQyYWI5Nzk3MzQ0YzdiNmRlMTkxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3OTY3LCAtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwMDBmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODAwMGZmIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF82M2I4ZmFjNWJhOGQ0YTlhOTA2ZjNmNjc0OTAxZjJlMSA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZDQ4MDNjMDc4ZjE5NDQ0ZDk4OWY4YzA1YWZmZDBiMGMgPSAkKGA8ZGl2IGlkPSJodG1sX2Q0ODAzYzA3OGYxOTQ0NGQ5ODlmOGMwNWFmZmQwYjBjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gSmFtZXMgVG93biwgQ2FiYmFnZXRvd24gQ2x1c3RlciAxPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzYzYjhmYWM1YmE4ZDRhOWE5MDZmM2Y2NzQ5MDFmMmUxLnNldENvbnRlbnQoaHRtbF9kNDgwM2MwNzhmMTk0NDRkOTg5ZjhjMDVhZmZkMGIwYyk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfYzNmZTBlNDAwMzdiNDJhYjk3OTczNDRjN2I2ZGUxOTEuYmluZFBvcHVwKHBvcHVwXzYzYjhmYWM1YmE4ZDRhOWE5MDZmM2Y2NzQ5MDFmMmUxKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMDE0ZGQ3ODllNGE0ZDJjODYxMjcxMWI0YTM3YWQyNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksIC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwMDBmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODAwMGZmIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF81YmNiMWYwOWNkODI0ZDAzYTIwZTFiNWY3NGJkNGJhOCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfZGI2MmQ5NGM4M2Y4NDYzODg5MjNjOGYzZDQxNDk4NGEgPSAkKGA8ZGl2IGlkPSJodG1sX2RiNjJkOTRjODNmODQ2Mzg4OTIzYzhmM2Q0MTQ5ODRhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHVyY2ggYW5kIFdlbGxlc2xleSBDbHVzdGVyIDE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfNWJjYjFmMDljZDgyNGQwM2EyMGUxYjVmNzRiZDRiYTguc2V0Q29udGVudChodG1sX2RiNjJkOTRjODNmODQ2Mzg4OTIzYzhmM2Q0MTQ5ODRhKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9hMDE0ZGQ3ODllNGE0ZDJjODYxMjcxMWI0YTM3YWQyNC5iaW5kUG9wdXAocG9wdXBfNWJjYjFmMDljZDgyNGQwM2EyMGUxYjVmNzRiZDRiYTgpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAyMzExYjgwOWYzODQ3MjNiMmFmM2QzMGQ2NjI5MzBmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwgLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MDAwZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kNTk2NWQ2OGVhYzg0ZWVkYWEzNzc3NjMwN2EyZTUwYSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzkwYmQzYTA4OTAzNDM4YThmYmMxNzg2YzVjZTVjODAgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2E0NTg0ZWI2N2RmZDQzMWI4MmE0NDI3ZGU1ODQ1OTMxID0gJChgPGRpdiBpZD0iaHRtbF9hNDU4NGViNjdkZmQ0MzFiODJhNDQyN2RlNTg0NTkzMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmVnZW50IFBhcmssIEhhcmJvdXJmcm9udCBDbHVzdGVyIDE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYzkwYmQzYTA4OTAzNDM4YThmYmMxNzg2YzVjZTVjODAuc2V0Q29udGVudChodG1sX2E0NTg0ZWI2N2RmZDQzMWI4MmE0NDI3ZGU1ODQ1OTMxKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8wMjMxMWI4MDlmMzg0NzIzYjJhZjNkMzBkNjYyOTMwZi5iaW5kUG9wdXAocG9wdXBfYzkwYmQzYTA4OTAzNDM4YThmYmMxNzg2YzVjZTVjODApCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ViMDlmZmFhNzVhNDQ1NDdhM2FhNzUzMDRmYzk5M2NlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwgLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZDU5NjVkNjhlYWM4NGVlZGFhMzc3NzYzMDdhMmU1MGEpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2MyNmVkMzczODc1YjQ0NzRiZWI0ZjVjYjRmNmMwYzU1ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8zYmQwOWQyZDAyYmY0NzBhOGY3OTY3YTkyNjkwY2QxOSA9ICQoYDxkaXYgaWQ9Imh0bWxfM2JkMDlkMmQwMmJmNDcwYThmNzk2N2E5MjY5MGNkMTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ5ZXJzb24sIEdhcmRlbiBEaXN0cmljdCBDbHVzdGVyIDE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYzI2ZWQzNzM4NzViNDQ3NGJlYjRmNWNiNGY2YzBjNTUuc2V0Q29udGVudChodG1sXzNiZDA5ZDJkMDJiZjQ3MGE4Zjc5NjdhOTI2OTBjZDE5KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl9lYjA5ZmZhYTc1YTQ0NTQ3YTNhYTc1MzA0ZmM5OTNjZS5iaW5kUG9wdXAocG9wdXBfYzI2ZWQzNzM4NzViNDQ3NGJlYjRmNWNiNGY2YzBjNTUpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZmZjMzOGM0ZWFjMTRjOTg5ZDM3MjI0YTU1NzY2ZTE2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwgLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MDAwZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kNTk2NWQ2OGVhYzg0ZWVkYWEzNzc3NjMwN2EyZTUwYSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYTA4NWUxNjJkYmRjNGJmZTkxYTRjZmEyNzE3ZDIyMTkgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzRmM2UxYjY1NWRjNzRlZDk5MjM4YjA4NmM4MjdiYjA4ID0gJChgPGRpdiBpZD0iaHRtbF80ZjNlMWI2NTVkYzc0ZWQ5OTIzOGIwODZjODI3YmIwOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24gQ2x1c3RlciAxPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2EwODVlMTYyZGJkYzRiZmU5MWE0Y2ZhMjcxN2QyMjE5LnNldENvbnRlbnQoaHRtbF80ZjNlMWI2NTVkYzc0ZWQ5OTIzOGIwODZjODI3YmIwOCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfZmZmMzM4YzRlYWMxNGM5ODlkMzcyMjRhNTU3NjZlMTYuYmluZFBvcHVwKHBvcHVwX2EwODVlMTYyZGJkYzRiZmU5MWE0Y2ZhMjcxN2QyMjE5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMGQ2NDRjYjgyMmE0ZjhlOTQ4ZDU2OWQ3YjFhZTUzOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwgLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MDAwZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kNTk2NWQ2OGVhYzg0ZWVkYWEzNzc3NjMwN2EyZTUwYSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTM5NzA1NzlmNmU1NGQ5NzkwMDQ5YWE0YTUzNmViMDQgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2Y2YzRjYmVlNjVlYzQ3ZjU4YTY4NDRmNjU4M2U2ZDMxID0gJChgPGRpdiBpZD0iaHRtbF9mNmM0Y2JlZTY1ZWM0N2Y1OGE2ODQ0ZjY1ODNlNmQzMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcmsgQ2x1c3RlciAxPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2UzOTcwNTc5ZjZlNTRkOTc5MDA0OWFhNGE1MzZlYjA0LnNldENvbnRlbnQoaHRtbF9mNmM0Y2JlZTY1ZWM0N2Y1OGE2ODQ0ZjY1ODNlNmQzMSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMTBkNjQ0Y2I4MjJhNGY4ZTk0OGQ1NjlkN2IxYWU1MzguYmluZFBvcHVwKHBvcHVwX2UzOTcwNTc5ZjZlNTRkOTc5MDA0OWFhNGE1MzZlYjA0KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YjY1ODYzYTcyYTI0MzQwYmY0MTViNTVkYTllOGI5YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsIC03OS4zODczODI2XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZDU5NjVkNjhlYWM4NGVlZGFhMzc3NzYzMDdhMmU1MGEpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwX2I3Yjg0NzE0YjUxNDQyOWE4ODYwYTc1NTBiMDA2YmEzID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF81MTA4ZDQ3NzU1M2Q0YTFkOGEyMDExZjliOTRlMDI5MiA9ICQoYDxkaXYgaWQ9Imh0bWxfNTEwOGQ0Nzc1NTNkNGExZDhhMjAxMWY5Yjk0ZTAyOTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCBDbHVzdGVyIDE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfYjdiODQ3MTRiNTE0NDI5YTg4NjBhNzU1MGIwMDZiYTMuc2V0Q29udGVudChodG1sXzUxMDhkNDc3NTUzZDRhMWQ4YTIwMTFmOWI5NGUwMjkyKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl82YjY1ODYzYTcyYTI0MzQwYmY0MTViNTVkYTllOGI5Yi5iaW5kUG9wdXAocG9wdXBfYjdiODQ3MTRiNTE0NDI5YTg4NjBhNzU1MGIwMDZiYTMpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEyNzkyN2Y1ODY2NDQ3YjQ5MGMzY2E0MTg3N2Y5M2Y0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsIC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZDU5NjVkNjhlYWM4NGVlZGFhMzc3NzYzMDdhMmU1MGEpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzk1ZGNkMGRmY2Y2MTQ0MjhhYTRjZDNlNzRlYTM4MGE3ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF83NmMzNTViNWYzZjE0YjBkYjg5MWZhNjQ5Mjk5NDM1ZCA9ICQoYDxkaXYgaWQ9Imh0bWxfNzZjMzU1YjVmM2YxNGIwZGI4OTFmYTY0OTI5OTQzNWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJpY2htb25kLCBLaW5nLCBBZGVsYWlkZSBDbHVzdGVyIDE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfOTVkY2QwZGZjZjYxNDQyOGFhNGNkM2U3NGVhMzgwYTcuc2V0Q29udGVudChodG1sXzc2YzM1NWI1ZjNmMTRiMGRiODkxZmE2NDkyOTk0MzVkKTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8xMjc5MjdmNTg2NjQ0N2I0OTBjM2NhNDE4NzdmOTNmNC5iaW5kUG9wdXAocG9wdXBfOTVkY2QwZGZjZjYxNDQyOGFhNGNkM2U3NGVhMzgwYTcpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQwOTczMTFlNWU1YTRkOTA5NzE4NGFiOWMyMWUwZGNjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywgLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZDU5NjVkNjhlYWM4NGVlZGFhMzc3NzYzMDdhMmU1MGEpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzAxNDIxOTU4NTIyODQxNzU4ZDdhYmU0NGU0OWZlN2E5ID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF81MWI3MmI1NDBmNzk0NzQ0YTUyMWZjZjM5YTAyYWEyOCA9ICQoYDxkaXYgaWQ9Imh0bWxfNTFiNzJiNTQwZjc5NDc0NGE1MjFmY2YzOWEwMmFhMjgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaW9uIFN0YXRpb24sIFRvcm9udG8gSXNsYW5kcywgSGFyYm91cmZyb250IEVhc3QgQ2x1c3RlciAxPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzAxNDIxOTU4NTIyODQxNzU4ZDdhYmU0NGU0OWZlN2E5LnNldENvbnRlbnQoaHRtbF81MWI3MmI1NDBmNzk0NzQ0YTUyMWZjZjM5YTAyYWEyOCk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfNDA5NzMxMWU1ZTVhNGQ5MDk3MTg0YWI5YzIxZTBkY2MuYmluZFBvcHVwKHBvcHVwXzAxNDIxOTU4NTIyODQxNzU4ZDdhYmU0NGU0OWZlN2E5KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MTc2MGVjNzY3MmU0YTcyYTYyNWY4YzM2ZWQyYWRmYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsIC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwMDBmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODAwMGZmIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF9hYjg1NGYwNmMxZGI0MTBhYTlmMjJhOTdkOTMzMjM1YyA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfOWJhMTk1YzgyMjQ3NDFkZDg5OTI3MmYxZjUyYjBmZTkgPSAkKGA8ZGl2IGlkPSJodG1sXzliYTE5NWM4MjI0NzQxZGQ4OTkyNzJmMWY1MmIwZmU5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ub3JvbnRvIERvbWluaW9uIENlbnRyZSwgRGVzaWduIEV4Y2hhbmdlIENsdXN0ZXIgMTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9hYjg1NGYwNmMxZGI0MTBhYTlmMjJhOTdkOTMzMjM1Yy5zZXRDb250ZW50KGh0bWxfOWJhMTk1YzgyMjQ3NDFkZDg5OTI3MmYxZjUyYjBmZTkpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzUxNzYwZWM3NjcyZTRhNzJhNjI1ZjhjMzZlZDJhZGZhLmJpbmRQb3B1cChwb3B1cF9hYjg1NGYwNmMxZGI0MTBhYTlmMjJhOTdkOTMzMjM1YykKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWMwNWNhNTNlMDQ5NDlmYWFjNmY3YTIzMDlhN2QyZWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LCAtNzkuMzc5ODE2OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MDAwZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kNTk2NWQ2OGVhYzg0ZWVkYWEzNzc3NjMwN2EyZTUwYSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZTE2MWJiY2ZjNzJhNDNmZDhhOTU3ZjE3MGM0NzdjYmEgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sX2FjZTRkOWRkNDE4YzQyZDhiNGM5ZDc5YWRkMWQ4YzRmID0gJChgPGRpdiBpZD0iaHRtbF9hY2U0ZDlkZDQxOGM0MmQ4YjRjOWQ3OWFkZDFkOGM0ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VmljdG9yaWEgSG90ZWwsIENvbW1lcmNlIENvdXJ0IENsdXN0ZXIgMTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF9lMTYxYmJjZmM3MmE0M2ZkOGE5NTdmMTcwYzQ3N2NiYS5zZXRDb250ZW50KGh0bWxfYWNlNGQ5ZGQ0MThjNDJkOGI0YzlkNzlhZGQxZDhjNGYpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzljMDVjYTUzZTA0OTQ5ZmFhYzZmN2EyMzA5YTdkMmVhLmJpbmRQb3B1cChwb3B1cF9lMTYxYmJjZmM3MmE0M2ZkOGE5NTdmMTcwYzQ3N2NiYSkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWFhNjgzMWMzYWE1NDVkY2JkYzFmYTk0M2YyODhmNDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LCAtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiIzgwMDBmZiIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjODAwMGZmIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF85MGFmNzYyMGZmMTE0Nzc0OGUxZWI0ZTQ3MTc1NmQxOCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfMGFlMzljZjQ2ODZmNDFmNWJjZGZkOGYzZGI2NDY1MGEgPSAkKGA8ZGl2IGlkPSJodG1sXzBhZTM5Y2Y0Njg2ZjQxZjViY2RmZDhmM2RiNjQ2NTBhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3JkLCBVbml2ZXJzaXR5IG9mIFRvcm9udG8gQ2x1c3RlciAxPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzkwYWY3NjIwZmYxMTQ3NzQ4ZTFlYjRlNDcxNzU2ZDE4LnNldENvbnRlbnQoaHRtbF8wYWUzOWNmNDY4NmY0MWY1YmNkZmQ4ZjNkYjY0NjUwYSk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOWFhNjgzMWMzYWE1NDVkY2JkYzFmYTk0M2YyODhmNDguYmluZFBvcHVwKHBvcHVwXzkwYWY3NjIwZmYxMTQ3NzQ4ZTFlYjRlNDcxNzU2ZDE4KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82OGY3MjBlNjBkOGI0YjE2YmU2Zjg2YjExMzlhYzE0YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsIC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsICJjb2xvciI6ICIjODAwMGZmIiwgImRhc2hBcnJheSI6IG51bGwsICJkYXNoT2Zmc2V0IjogbnVsbCwgImZpbGwiOiB0cnVlLCAiZmlsbENvbG9yIjogIiM4MDAwZmYiLCAiZmlsbE9wYWNpdHkiOiAwLjcsICJmaWxsUnVsZSI6ICJldmVub2RkIiwgImxpbmVDYXAiOiAicm91bmQiLCAibGluZUpvaW4iOiAicm91bmQiLCAib3BhY2l0eSI6IDEuMCwgInJhZGl1cyI6IDUsICJzdHJva2UiOiB0cnVlLCAid2VpZ2h0IjogM30KICAgICAgICAgICAgKS5hZGRUbyhtYXBfZDU5NjVkNjhlYWM4NGVlZGFhMzc3NzYzMDdhMmU1MGEpOwogICAgICAgIAogICAgCiAgICAgICAgdmFyIHBvcHVwXzVhNDA3MzRhNGY2NTQyNGZiMmU5YjYyZTI1NDI3OTZkID0gTC5wb3B1cCh7Im1heFdpZHRoIjogIjEwMCUifSk7CgogICAgICAgIAogICAgICAgICAgICB2YXIgaHRtbF8wMDU5ZDc2OWViNGI0ZjNkYjFmYmY2MmM4NzA4MGRkMiA9ICQoYDxkaXYgaWQ9Imh0bWxfMDA1OWQ3NjllYjRiNGYzZGIxZmJmNjJjODcwODBkZDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdyYW5nZSBQYXJrLCBDaGluYXRvd24sIEtlbnNpbmd0b24gTWFya2V0IENsdXN0ZXIgMTwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF81YTQwNzM0YTRmNjU0MjRmYjJlOWI2MmUyNTQyNzk2ZC5zZXRDb250ZW50KGh0bWxfMDA1OWQ3NjllYjRiNGYzZGIxZmJmNjJjODcwODBkZDIpOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyXzY4ZjcyMGU2MGQ4YjRiMTZiZTZmODZiMTEzOWFjMTRhLmJpbmRQb3B1cChwb3B1cF81YTQwNzM0YTRmNjU0MjRmYjJlOWI2MmUyNTQyNzk2ZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWRmMDYwNGYwNmU1NGI3NWFmYmY1YzYxMzM0NGYyOWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LCAtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7ImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLCAiY29sb3IiOiAiI2ZmMDAwMCIsICJkYXNoQXJyYXkiOiBudWxsLCAiZGFzaE9mZnNldCI6IG51bGwsICJmaWxsIjogdHJ1ZSwgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwgImZpbGxPcGFjaXR5IjogMC43LCAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsICJsaW5lQ2FwIjogInJvdW5kIiwgImxpbmVKb2luIjogInJvdW5kIiwgIm9wYWNpdHkiOiAxLjAsICJyYWRpdXMiOiA1LCAic3Ryb2tlIjogdHJ1ZSwgIndlaWdodCI6IDN9CiAgICAgICAgICAgICkuYWRkVG8obWFwX2Q1OTY1ZDY4ZWFjODRlZWRhYTM3Nzc2MzA3YTJlNTBhKTsKICAgICAgICAKICAgIAogICAgICAgIHZhciBwb3B1cF84NjE1NzVkMDU1Y2M0NzBjYWIzZjkwOTJkMmM5MzQ4ZCA9IEwucG9wdXAoeyJtYXhXaWR0aCI6ICIxMDAlIn0pOwoKICAgICAgICAKICAgICAgICAgICAgdmFyIGh0bWxfNDI4NjFkNDYxNzFkNGFhYzk0OTRmMTc2MjgzZmE3ODAgPSAkKGA8ZGl2IGlkPSJodG1sXzQyODYxZDQ2MTcxZDRhYWM5NDk0ZjE3NjI4M2ZhNzgwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgV2VzdCwgQmF0aHVyc3QgUXVheSwgSXNsYW5kIGFpcnBvcnQsIFJhaWx3YXkgTGFuZHMsIEtpbmcgYW5kIFNwYWRpbmEsIFNvdXRoIE5pYWdhcmEsIENOIFRvd2VyIENsdXN0ZXIgMDwvZGl2PmApWzBdOwogICAgICAgICAgICBwb3B1cF84NjE1NzVkMDU1Y2M0NzBjYWIzZjkwOTJkMmM5MzQ4ZC5zZXRDb250ZW50KGh0bWxfNDI4NjFkNDYxNzFkNGFhYzk0OTRmMTc2MjgzZmE3ODApOwogICAgICAgIAoKICAgICAgICBjaXJjbGVfbWFya2VyX2FkZjA2MDRmMDZlNTRiNzVhZmJmNWM2MTMzNDRmMjljLmJpbmRQb3B1cChwb3B1cF84NjE1NzVkMDU1Y2M0NzBjYWIzZjkwOTJkMmM5MzQ4ZCkKICAgICAgICA7CgogICAgICAgIAogICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTc4MTQ0ZGQ3YzBjNGFiNzhiZGMwMTBjODM1NmI1OGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLCAtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MDAwZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kNTk2NWQ2OGVhYzg0ZWVkYWEzNzc3NjMwN2EyZTUwYSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfZDQ0ZDk5YWY2Y2I1NDFlMDhkZTYwYzg5N2FhMzI4MzIgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzEyOTgyOGIwOWExMDQ2NjRiYWM3N2FlYmZhMWMyY2M1ID0gJChgPGRpdiBpZD0iaHRtbF8xMjk4MjhiMDlhMTA0NjY0YmFjNzdhZWJmYTFjMmNjNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3RuIEEgUE8gQm94ZXMgMjUgVGhlIEVzcGxhbmFkZSBDbHVzdGVyIDE8L2Rpdj5gKVswXTsKICAgICAgICAgICAgcG9wdXBfZDQ0ZDk5YWY2Y2I1NDFlMDhkZTYwYzg5N2FhMzI4MzIuc2V0Q29udGVudChodG1sXzEyOTgyOGIwOWExMDQ2NjRiYWM3N2FlYmZhMWMyY2M1KTsKICAgICAgICAKCiAgICAgICAgY2lyY2xlX21hcmtlcl8xNzgxNDRkZDdjMGM0YWI3OGJkYzAxMGM4MzU2YjU4ZS5iaW5kUG9wdXAocG9wdXBfZDQ0ZDk5YWY2Y2I1NDFlMDhkZTYwYzg5N2FhMzI4MzIpCiAgICAgICAgOwoKICAgICAgICAKICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk3MjlmZjlkMzViMjRiNjM5OWZiNDA1MDJkMjM1NjhmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwgLTc5LjM4MjI4MDJdLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MDAwZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kNTk2NWQ2OGVhYzg0ZWVkYWEzNzc3NjMwN2EyZTUwYSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfYzk1NWMzODc4NTRjNDM5ZTg1MTExZjI1ZWU0ZmRkYzMgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzVkOThkODdjYjRmOTQ2NmNiMTZkYjUyYTI4NjJhYTQzID0gJChgPGRpdiBpZD0iaHRtbF81ZDk4ZDg3Y2I0Zjk0NjZjYjE2ZGI1MmEyODYyYWE0MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5kZXJncm91bmQgY2l0eSwgRmlyc3QgQ2FuYWRpYW4gUGxhY2UgQ2x1c3RlciAxPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwX2M5NTVjMzg3ODU0YzQzOWU4NTExMWYyNWVlNGZkZGMzLnNldENvbnRlbnQoaHRtbF81ZDk4ZDg3Y2I0Zjk0NjZjYjE2ZGI1MmEyODYyYWE0Myk7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfOTcyOWZmOWQzNWIyNGI2Mzk5ZmI0MDUwMmQyMzU2OGYuYmluZFBvcHVwKHBvcHVwX2M5NTVjMzg3ODU0YzQzOWU4NTExMWYyNWVlNGZkZGMzKQogICAgICAgIDsKCiAgICAgICAgCiAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZDc0NDk0MDNkOWY0OGU5OGZlZTRjODdlMjg4YzNkNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwgLTc5LjQyMjU2MzddLAogICAgICAgICAgICAgICAgeyJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwgImNvbG9yIjogIiM4MDAwZmYiLCAiZGFzaEFycmF5IjogbnVsbCwgImRhc2hPZmZzZXQiOiBudWxsLCAiZmlsbCI6IHRydWUsICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsICJmaWxsT3BhY2l0eSI6IDAuNywgImZpbGxSdWxlIjogImV2ZW5vZGQiLCAibGluZUNhcCI6ICJyb3VuZCIsICJsaW5lSm9pbiI6ICJyb3VuZCIsICJvcGFjaXR5IjogMS4wLCAicmFkaXVzIjogNSwgInN0cm9rZSI6IHRydWUsICJ3ZWlnaHQiOiAzfQogICAgICAgICAgICApLmFkZFRvKG1hcF9kNTk2NWQ2OGVhYzg0ZWVkYWEzNzc3NjMwN2EyZTUwYSk7CiAgICAgICAgCiAgICAKICAgICAgICB2YXIgcG9wdXBfNWEwNWRjNjNhYWQwNDUzNzkwZjliZTAyZGI4OGEzZTUgPSBMLnBvcHVwKHsibWF4V2lkdGgiOiAiMTAwJSJ9KTsKCiAgICAgICAgCiAgICAgICAgICAgIHZhciBodG1sXzFiZGIwMjY1YjcwYTQ1M2Y5NmI0YjJkYzk0ZTc2N2NiID0gJChgPGRpdiBpZD0iaHRtbF8xYmRiMDI2NWI3MGE0NTNmOTZiNGIyZGM5NGU3NjdjYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUgQ2x1c3RlciAxPC9kaXY+YClbMF07CiAgICAgICAgICAgIHBvcHVwXzVhMDVkYzYzYWFkMDQ1Mzc5MGY5YmUwMmRiODhhM2U1LnNldENvbnRlbnQoaHRtbF8xYmRiMDI2NWI3MGE0NTNmOTZiNGIyZGM5NGU3NjdjYik7CiAgICAgICAgCgogICAgICAgIGNpcmNsZV9tYXJrZXJfMGQ3NDQ5NDAzZDlmNDhlOThmZWU0Yzg3ZTI4OGMzZDUuYmluZFBvcHVwKHBvcHVwXzVhMDVkYzYzYWFkMDQ1Mzc5MGY5YmUwMmRiODhhM2U1KQogICAgICAgIDsKCiAgICAgICAgCiAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



#### Analyze the neighbourhoods to determine the most populous cluster


```python
dwnttoronto_combined_df_pop = dwnttoronto_combined_df.groupby('Cluster Labels').count()
```


```python
dwnttoronto_combined_df_pop
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
    <tr>
      <th>Cluster Labels</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**The Conclusion**: The most venues belong to the zeros cluster based on k-means algorithm
