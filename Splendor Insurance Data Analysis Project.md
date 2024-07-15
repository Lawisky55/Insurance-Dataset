```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

file_path = "C:/Users/hp/Downloads/Insurance Policies.csv"
    
df = pd.read_csv(file_path)
```


```python
#checking the data type

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37542 entries, 0 to 37541
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   ID                37542 non-null  object
     1   birthdate         37542 non-null  object
     2   marital_status    37542 non-null  object
     3   car_use           37542 non-null  object
     4   gender            37542 non-null  object
     5   kids_driving      37542 non-null  int64 
     6   parent            37542 non-null  object
     7   education         37542 non-null  object
     8   car_make          37542 non-null  object
     9   car_model         37542 non-null  object
     10  car_color         37542 non-null  object
     11  car_year          37542 non-null  int64 
     12  claim_freq        37542 non-null  int64 
     13  coverage_zone     37542 non-null  object
     14  claim_amt         37542 non-null  object
     15  household_income  37542 non-null  object
    dtypes: int64(3), object(13)
    memory usage: 4.6+ MB
    


```python
#checking the shape of the dataset

df.shape
```




    (37542, 16)




```python
#exploring the first 5 rows 
df.head()
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
      <th>ID</th>
      <th>birthdate</th>
      <th>marital_status</th>
      <th>car_use</th>
      <th>gender</th>
      <th>kids_driving</th>
      <th>parent</th>
      <th>education</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_color</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>coverage_zone</th>
      <th>claim_amt</th>
      <th>household_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62-2999778</td>
      <td>8/9/1962</td>
      <td>Single</td>
      <td>Private</td>
      <td>Male</td>
      <td>2</td>
      <td>Yes</td>
      <td>High School</td>
      <td>Acura</td>
      <td>TSX</td>
      <td>Green</td>
      <td>2010</td>
      <td>1</td>
      <td>Highly Urban</td>
      <td>$73759.88</td>
      <td>$220436.66</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70-2426103</td>
      <td>4/21/1988</td>
      <td>Married</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Corbin</td>
      <td>Sparrow</td>
      <td>Turquoise</td>
      <td>2004</td>
      <td>1</td>
      <td>Urban</td>
      <td>$78975.41</td>
      <td>$66491.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>08-3808219</td>
      <td>3/8/1999</td>
      <td>Divorced</td>
      <td>Private</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Nissan</td>
      <td>Pathfinder</td>
      <td>Orange</td>
      <td>1993</td>
      <td>0</td>
      <td>Rural</td>
      <td>$30904.01</td>
      <td>$56122.70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38-0306843</td>
      <td>5/10/1959</td>
      <td>Single</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Ford</td>
      <td>Econoline E350</td>
      <td>Pink</td>
      <td>2000</td>
      <td>1</td>
      <td>Highly Urban</td>
      <td>$30257.82</td>
      <td>$175182.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47-5163637</td>
      <td>1/15/1992</td>
      <td>Single</td>
      <td>Commercial</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Masters</td>
      <td>Nissan</td>
      <td>350Z</td>
      <td>Green</td>
      <td>2006</td>
      <td>3</td>
      <td>Rural</td>
      <td>$50434.02</td>
      <td>$137110.23</td>
    </tr>
  </tbody>
</table>
</div>




```python
#getting a statistical summary of the numerical variables of the dataset

df.describe()
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
      <th>kids_driving</th>
      <th>car_year</th>
      <th>claim_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>37542.000000</td>
      <td>37542.000000</td>
      <td>37542.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.418278</td>
      <td>2000.293005</td>
      <td>0.510308</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.736958</td>
      <td>9.045441</td>
      <td>1.015050</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1909.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1995.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>2002.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>2007.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>2013.000000</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking a statistical summary of the string or object variables 
df2 = df.dtypes[df.dtypes == "object"].index
    
df[df2].describe()

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
      <th>ID</th>
      <th>birthdate</th>
      <th>marital_status</th>
      <th>car_use</th>
      <th>gender</th>
      <th>parent</th>
      <th>education</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_color</th>
      <th>coverage_zone</th>
      <th>claim_amt</th>
      <th>household_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
      <td>37542</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>37541</td>
      <td>16525</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>78</td>
      <td>1011</td>
      <td>19</td>
      <td>5</td>
      <td>37474</td>
      <td>37502</td>
    </tr>
    <tr>
      <th>top</th>
      <td>56-5402470</td>
      <td>10/23/1981</td>
      <td>Single</td>
      <td>Private</td>
      <td>Female</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Ford</td>
      <td>Grand Prix</td>
      <td>Turquoise</td>
      <td>Urban</td>
      <td>$44939.31</td>
      <td>$109041.67</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2</td>
      <td>9</td>
      <td>15525</td>
      <td>30060</td>
      <td>18806</td>
      <td>20932</td>
      <td>18701</td>
      <td>3302</td>
      <td>250</td>
      <td>2078</td>
      <td>7588</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#removing $ char from the lables and converting the varibales to float

df["claim_amt"] = df["claim_amt"].str.lstrip("$").astype(float)
df["household_income"] = df["household_income"].str.lstrip("$").astype(float)


df
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
      <th>ID</th>
      <th>birthdate</th>
      <th>marital_status</th>
      <th>car_use</th>
      <th>gender</th>
      <th>kids_driving</th>
      <th>parent</th>
      <th>education</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_color</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>coverage_zone</th>
      <th>claim_amt</th>
      <th>household_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62-2999778</td>
      <td>8/9/1962</td>
      <td>Single</td>
      <td>Private</td>
      <td>Male</td>
      <td>2</td>
      <td>Yes</td>
      <td>High School</td>
      <td>Acura</td>
      <td>TSX</td>
      <td>Green</td>
      <td>2010</td>
      <td>1</td>
      <td>Highly Urban</td>
      <td>73759.88</td>
      <td>220436.66</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70-2426103</td>
      <td>4/21/1988</td>
      <td>Married</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Corbin</td>
      <td>Sparrow</td>
      <td>Turquoise</td>
      <td>2004</td>
      <td>1</td>
      <td>Urban</td>
      <td>78975.41</td>
      <td>66491.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>08-3808219</td>
      <td>3/8/1999</td>
      <td>Divorced</td>
      <td>Private</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Nissan</td>
      <td>Pathfinder</td>
      <td>Orange</td>
      <td>1993</td>
      <td>0</td>
      <td>Rural</td>
      <td>30904.01</td>
      <td>56122.70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38-0306843</td>
      <td>5/10/1959</td>
      <td>Single</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Ford</td>
      <td>Econoline E350</td>
      <td>Pink</td>
      <td>2000</td>
      <td>1</td>
      <td>Highly Urban</td>
      <td>30257.82</td>
      <td>175182.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47-5163637</td>
      <td>1/15/1992</td>
      <td>Single</td>
      <td>Commercial</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Masters</td>
      <td>Nissan</td>
      <td>350Z</td>
      <td>Green</td>
      <td>2006</td>
      <td>3</td>
      <td>Rural</td>
      <td>50434.02</td>
      <td>137110.23</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>37537</th>
      <td>76-9636930</td>
      <td>12/21/1963</td>
      <td>Single</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>High School</td>
      <td>Audi</td>
      <td>S4</td>
      <td>Fuscia</td>
      <td>2000</td>
      <td>2</td>
      <td>Urban</td>
      <td>36023.32</td>
      <td>126360.31</td>
    </tr>
    <tr>
      <th>37538</th>
      <td>70-8201812</td>
      <td>8/6/1957</td>
      <td>Seperated</td>
      <td>Commercial</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>GMC</td>
      <td>3500</td>
      <td>Maroon</td>
      <td>1997</td>
      <td>0</td>
      <td>Highly Rural</td>
      <td>83220.69</td>
      <td>180571.33</td>
    </tr>
    <tr>
      <th>37539</th>
      <td>14-7596380</td>
      <td>10/23/1950</td>
      <td>Divorced</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>PhD</td>
      <td>Chevrolet</td>
      <td>Camaro</td>
      <td>Turquoise</td>
      <td>1974</td>
      <td>1</td>
      <td>Highly Rural</td>
      <td>9515.35</td>
      <td>144296.53</td>
    </tr>
    <tr>
      <th>37540</th>
      <td>72-6900872</td>
      <td>4/19/1976</td>
      <td>Single</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Masters</td>
      <td>Jaguar</td>
      <td>XJ Series</td>
      <td>Pink</td>
      <td>2003</td>
      <td>0</td>
      <td>Urban</td>
      <td>56333.58</td>
      <td>117245.10</td>
    </tr>
    <tr>
      <th>37541</th>
      <td>39-6644657</td>
      <td>4/20/1965</td>
      <td>Divorced</td>
      <td>Commercial</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Kia</td>
      <td>Amanti</td>
      <td>Purple</td>
      <td>2004</td>
      <td>0</td>
      <td>Urban</td>
      <td>33764.08</td>
      <td>146342.17</td>
    </tr>
  </tbody>
</table>
<p>37542 rows × 16 columns</p>
</div>




```python
#comfirming the datatype convertion for claim_freq and claim_amt to float
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37542 entries, 0 to 37541
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   ID                37542 non-null  object 
     1   birthdate         37542 non-null  object 
     2   marital_status    37542 non-null  object 
     3   car_use           37542 non-null  object 
     4   gender            37542 non-null  object 
     5   kids_driving      37542 non-null  int64  
     6   parent            37542 non-null  object 
     7   education         37542 non-null  object 
     8   car_make          37542 non-null  object 
     9   car_model         37542 non-null  object 
     10  car_color         37542 non-null  object 
     11  car_year          37542 non-null  int64  
     12  claim_freq        37542 non-null  int64  
     13  coverage_zone     37542 non-null  object 
     14  claim_amt         37542 non-null  float64
     15  household_income  37542 non-null  float64
    dtypes: float64(2), int64(3), object(11)
    memory usage: 4.6+ MB
    


```python
#dropping duplicate

df.drop_duplicates()
df.shape
```




    (37542, 16)




```python
#ensuring the birthdate column is in datetime dtypes
df["birthdate"] = pd.to_datetime(df["birthdate"])

df["birthdate"]
```




    0       1962-08-09
    1       1988-04-21
    2       1999-03-08
    3       1959-05-10
    4       1992-01-15
               ...    
    37537   1963-12-21
    37538   1957-08-06
    37539   1950-10-23
    37540   1976-04-19
    37541   1965-04-20
    Name: birthdate, Length: 37542, dtype: datetime64[ns]




```python
#creating the age column for each policyholders

today = pd.to_datetime("today")

df["age"] = (today - df["birthdate"]).astype("<m8[Y]")

df.head()
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
      <th>ID</th>
      <th>birthdate</th>
      <th>marital_status</th>
      <th>car_use</th>
      <th>gender</th>
      <th>kids_driving</th>
      <th>parent</th>
      <th>education</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_color</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>coverage_zone</th>
      <th>claim_amt</th>
      <th>household_income</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62-2999778</td>
      <td>1962-08-09</td>
      <td>Single</td>
      <td>Private</td>
      <td>Male</td>
      <td>2</td>
      <td>Yes</td>
      <td>High School</td>
      <td>Acura</td>
      <td>TSX</td>
      <td>Green</td>
      <td>2010</td>
      <td>1</td>
      <td>Highly Urban</td>
      <td>73759.88</td>
      <td>220436.66</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70-2426103</td>
      <td>1988-04-21</td>
      <td>Married</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Corbin</td>
      <td>Sparrow</td>
      <td>Turquoise</td>
      <td>2004</td>
      <td>1</td>
      <td>Urban</td>
      <td>78975.41</td>
      <td>66491.43</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>08-3808219</td>
      <td>1999-03-08</td>
      <td>Divorced</td>
      <td>Private</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Nissan</td>
      <td>Pathfinder</td>
      <td>Orange</td>
      <td>1993</td>
      <td>0</td>
      <td>Rural</td>
      <td>30904.01</td>
      <td>56122.70</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38-0306843</td>
      <td>1959-05-10</td>
      <td>Single</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Ford</td>
      <td>Econoline E350</td>
      <td>Pink</td>
      <td>2000</td>
      <td>1</td>
      <td>Highly Urban</td>
      <td>30257.82</td>
      <td>175182.61</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47-5163637</td>
      <td>1992-01-15</td>
      <td>Single</td>
      <td>Commercial</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Masters</td>
      <td>Nissan</td>
      <td>350Z</td>
      <td>Green</td>
      <td>2006</td>
      <td>3</td>
      <td>Rural</td>
      <td>50434.02</td>
      <td>137110.23</td>
      <td>32.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#comparing the claim_freq and claim_amt by gender
df1 = df.groupby("gender").agg({"claim_freq":"mean","claim_amt":"mean"}).reset_index()

df1

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
      <th>gender</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0.514357</td>
      <td>49860.058201</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0.506245</td>
      <td>50197.599363</td>
    </tr>
  </tbody>
</table>
</div>




```python
#comparing the claim_amt and claim_amt by the marital_status

df2 = df.groupby("marital_status").agg({"claim_freq":"mean","claim_amt":"mean"}).reset_index()

df2
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
      <th>marital_status</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Divorced</td>
      <td>0.492213</td>
      <td>50089.164486</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Married</td>
      <td>0.520286</td>
      <td>50337.818821</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Seperated</td>
      <td>0.533657</td>
      <td>49259.423023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Single</td>
      <td>0.504992</td>
      <td>49906.322566</td>
    </tr>
  </tbody>
</table>
</div>




```python
#comparing the claim_freq by gender
df3 = df.groupby("gender").agg({"claim_freq":"mean","claim_amt":"mean"}).reset_index()

df3
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
      <th>gender</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>0.514357</td>
      <td>49860.058201</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>0.506245</td>
      <td>50197.599363</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sorted by freq

sorted_freq = df2.sort_values(by= "claim_amt", ascending = True)

sorted_freq
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
      <th>marital_status</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Seperated</td>
      <td>0.533657</td>
      <td>49259.423023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Single</td>
      <td>0.504992</td>
      <td>49906.322566</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Divorced</td>
      <td>0.492213</td>
      <td>50089.164486</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Married</td>
      <td>0.520286</td>
      <td>50337.818821</td>
    </tr>
  </tbody>
</table>
</div>




```python
#sorted by amt

sorted_amt = df2.sort_values(by= "claim_amt", ascending = False)

sorted_amt
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
      <th>marital_status</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Married</td>
      <td>0.520286</td>
      <td>50337.818821</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Divorced</td>
      <td>0.492213</td>
      <td>50089.164486</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Single</td>
      <td>0.504992</td>
      <td>49906.322566</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Seperated</td>
      <td>0.533657</td>
      <td>49259.423023</td>
    </tr>
  </tbody>
</table>
</div>




```python
sorted_freqt = df2.sort_values(by = "claim_freq", ascending=False)

sorted_freqt
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
      <th>marital_status</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Seperated</td>
      <td>0.533657</td>
      <td>49259.423023</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Married</td>
      <td>0.520286</td>
      <td>50337.818821</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Single</td>
      <td>0.504992</td>
      <td>49906.322566</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Divorced</td>
      <td>0.492213</td>
      <td>50089.164486</td>
    </tr>
  </tbody>
</table>
</div>




```python
#creating labels for age category
bins = [21,30,40,50,60,70,80]
labels = ["21-30","31-40","41-50","51-60", "61-70","71-80"]

df["age_group"] = pd.cut(df["age"], bins=bins,labels=labels, right= False)

df.sort_values(by = "age", ascending = True)
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
      <th>ID</th>
      <th>birthdate</th>
      <th>marital_status</th>
      <th>car_use</th>
      <th>gender</th>
      <th>kids_driving</th>
      <th>parent</th>
      <th>education</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_color</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>coverage_zone</th>
      <th>claim_amt</th>
      <th>household_income</th>
      <th>age</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28371</th>
      <td>89-6877174</td>
      <td>2002-09-22</td>
      <td>Single</td>
      <td>Private</td>
      <td>Female</td>
      <td>1</td>
      <td>Yes</td>
      <td>Masters</td>
      <td>Mitsubishi</td>
      <td>Eclipse</td>
      <td>Maroon</td>
      <td>1990</td>
      <td>1</td>
      <td>Highly Rural</td>
      <td>63868.48</td>
      <td>187029.43</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>13570</th>
      <td>63-0334033</td>
      <td>2002-10-11</td>
      <td>Single</td>
      <td>Commercial</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Chevrolet</td>
      <td>HHR</td>
      <td>Blue</td>
      <td>2010</td>
      <td>2</td>
      <td>Highly Urban</td>
      <td>65693.81</td>
      <td>235558.86</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>32729</th>
      <td>32-8766564</td>
      <td>2002-09-09</td>
      <td>Married</td>
      <td>Commercial</td>
      <td>Female</td>
      <td>1</td>
      <td>Yes</td>
      <td>Bachelors</td>
      <td>Porsche</td>
      <td>944</td>
      <td>Blue</td>
      <td>1989</td>
      <td>2</td>
      <td>Urban</td>
      <td>30889.62</td>
      <td>197889.29</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>25558</th>
      <td>61-0449493</td>
      <td>2002-08-25</td>
      <td>Single</td>
      <td>Private</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>High School</td>
      <td>Nissan</td>
      <td>Sentra</td>
      <td>Khaki</td>
      <td>2011</td>
      <td>3</td>
      <td>Suburban</td>
      <td>83595.00</td>
      <td>144404.30</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>2540</th>
      <td>98-0319293</td>
      <td>2002-07-13</td>
      <td>Married</td>
      <td>Private</td>
      <td>Male</td>
      <td>2</td>
      <td>Yes</td>
      <td>Masters</td>
      <td>Hyundai</td>
      <td>Sonata</td>
      <td>Blue</td>
      <td>1994</td>
      <td>0</td>
      <td>Suburban</td>
      <td>40585.27</td>
      <td>187781.24</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>35373</th>
      <td>06-7813206</td>
      <td>1950-06-11</td>
      <td>Seperated</td>
      <td>Commercial</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>High School</td>
      <td>Plymouth</td>
      <td>Laser</td>
      <td>Yellow</td>
      <td>1991</td>
      <td>0</td>
      <td>Highly Urban</td>
      <td>91553.02</td>
      <td>238019.00</td>
      <td>74.0</td>
      <td>71-80</td>
    </tr>
    <tr>
      <th>35371</th>
      <td>30-8154449</td>
      <td>1949-11-13</td>
      <td>Married</td>
      <td>Commercial</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Bachelors</td>
      <td>GMC</td>
      <td>Savana 1500</td>
      <td>Teal</td>
      <td>2001</td>
      <td>0</td>
      <td>Urban</td>
      <td>96972.75</td>
      <td>160679.35</td>
      <td>74.0</td>
      <td>71-80</td>
    </tr>
    <tr>
      <th>27555</th>
      <td>47-9466454</td>
      <td>1950-02-12</td>
      <td>Married</td>
      <td>Private</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Masters</td>
      <td>Dodge</td>
      <td>Durango</td>
      <td>Red</td>
      <td>1999</td>
      <td>0</td>
      <td>Suburban</td>
      <td>91790.90</td>
      <td>46151.94</td>
      <td>74.0</td>
      <td>71-80</td>
    </tr>
    <tr>
      <th>15761</th>
      <td>00-3174177</td>
      <td>1950-02-18</td>
      <td>Married</td>
      <td>Private</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>Bachelors</td>
      <td>Volvo</td>
      <td>S40</td>
      <td>Blue</td>
      <td>2004</td>
      <td>0</td>
      <td>Suburban</td>
      <td>88607.64</td>
      <td>242523.16</td>
      <td>74.0</td>
      <td>71-80</td>
    </tr>
    <tr>
      <th>19170</th>
      <td>29-0878155</td>
      <td>1950-04-03</td>
      <td>Divorced</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Bachelors</td>
      <td>Mazda</td>
      <td>Tribute</td>
      <td>Red</td>
      <td>2009</td>
      <td>4</td>
      <td>Highly Rural</td>
      <td>69701.47</td>
      <td>55560.29</td>
      <td>74.0</td>
      <td>71-80</td>
    </tr>
  </tbody>
</table>
<p>37542 rows × 18 columns</p>
</div>




```python
# comparing claim_freq and amount by age group

df4 = df.groupby("age_group").agg({"claim_freq":"count","claim_amt":"sum"}).reset_index()

df4.sort_values(by=["claim_freq","claim_amt"],ascending=False)
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
      <th>age_group</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>61-70</td>
      <td>7219</td>
      <td>3.638200e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31-40</td>
      <td>7128</td>
      <td>3.563494e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41-50</td>
      <td>7058</td>
      <td>3.543259e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>51-60</td>
      <td>7036</td>
      <td>3.512491e+08</td>
    </tr>
    <tr>
      <th>0</th>
      <td>21-30</td>
      <td>5875</td>
      <td>2.929769e+08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>71-80</td>
      <td>3226</td>
      <td>1.594491e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df["age"]==21]
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
      <th>ID</th>
      <th>birthdate</th>
      <th>marital_status</th>
      <th>car_use</th>
      <th>gender</th>
      <th>kids_driving</th>
      <th>parent</th>
      <th>education</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_color</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>coverage_zone</th>
      <th>claim_amt</th>
      <th>household_income</th>
      <th>age</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>516</th>
      <td>43-3186478</td>
      <td>2002-08-04</td>
      <td>Married</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Acura</td>
      <td>TSX</td>
      <td>Crimson</td>
      <td>2009</td>
      <td>0</td>
      <td>Rural</td>
      <td>69588.26</td>
      <td>50304.58</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>642</th>
      <td>90-5736789</td>
      <td>2002-09-24</td>
      <td>Single</td>
      <td>Private</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Masters</td>
      <td>Suzuki</td>
      <td>Swift</td>
      <td>Maroon</td>
      <td>2005</td>
      <td>0</td>
      <td>Rural</td>
      <td>78083.26</td>
      <td>160986.90</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>696</th>
      <td>38-7495650</td>
      <td>2002-07-30</td>
      <td>Married</td>
      <td>Private</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>High School</td>
      <td>Isuzu</td>
      <td>Hombre</td>
      <td>Turquoise</td>
      <td>2000</td>
      <td>0</td>
      <td>Highly Rural</td>
      <td>92614.69</td>
      <td>118645.68</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>1126</th>
      <td>36-0076526</td>
      <td>2002-08-07</td>
      <td>Single</td>
      <td>Private</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Chrysler</td>
      <td>Crossfire</td>
      <td>Puce</td>
      <td>2007</td>
      <td>3</td>
      <td>Highly Rural</td>
      <td>62661.02</td>
      <td>73840.76</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>27-2706723</td>
      <td>2002-09-28</td>
      <td>Single</td>
      <td>Private</td>
      <td>Female</td>
      <td>1</td>
      <td>Yes</td>
      <td>High School</td>
      <td>Dodge</td>
      <td>Charger</td>
      <td>Pink</td>
      <td>2011</td>
      <td>0</td>
      <td>Rural</td>
      <td>77841.28</td>
      <td>139199.08</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36890</th>
      <td>09-8241233</td>
      <td>2002-08-05</td>
      <td>Married</td>
      <td>Commercial</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>Bachelors</td>
      <td>Dodge</td>
      <td>Ram 3500</td>
      <td>Khaki</td>
      <td>2001</td>
      <td>0</td>
      <td>Suburban</td>
      <td>81920.55</td>
      <td>132085.38</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>37196</th>
      <td>58-3897681</td>
      <td>2002-08-17</td>
      <td>Married</td>
      <td>Commercial</td>
      <td>Female</td>
      <td>2</td>
      <td>Yes</td>
      <td>PhD</td>
      <td>Ford</td>
      <td>F-Series</td>
      <td>Green</td>
      <td>1989</td>
      <td>0</td>
      <td>Highly Rural</td>
      <td>91794.63</td>
      <td>107292.56</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>37279</th>
      <td>16-3922936</td>
      <td>2002-10-07</td>
      <td>Married</td>
      <td>Commercial</td>
      <td>Female</td>
      <td>1</td>
      <td>Yes</td>
      <td>High School</td>
      <td>Lamborghini</td>
      <td>Diablo</td>
      <td>Pink</td>
      <td>1999</td>
      <td>0</td>
      <td>Rural</td>
      <td>8312.19</td>
      <td>131483.51</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>37473</th>
      <td>34-4880330</td>
      <td>2002-09-24</td>
      <td>Seperated</td>
      <td>Commercial</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Nissan</td>
      <td>Rogue</td>
      <td>Crimson</td>
      <td>2008</td>
      <td>0</td>
      <td>Suburban</td>
      <td>11958.51</td>
      <td>69080.51</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
    <tr>
      <th>37495</th>
      <td>25-1339598</td>
      <td>2002-08-09</td>
      <td>Married</td>
      <td>Private</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>High School</td>
      <td>Ford</td>
      <td>Expedition EL</td>
      <td>Purple</td>
      <td>2012</td>
      <td>0</td>
      <td>Highly Rural</td>
      <td>97035.13</td>
      <td>94157.74</td>
      <td>21.0</td>
      <td>21-30</td>
    </tr>
  </tbody>
</table>
<p>203 rows × 18 columns</p>
</div>




```python
#pd.crosstab(index = df["claim_freq"], columns=df["car_make"])

df5 = df.groupby("car_make").agg({"claim_freq":"count","claim_amt":"mean"}).reset_index()

df5.sort_values(by="claim_amt", ascending= True).head(20)
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
      <th>car_make</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>Rambler</td>
      <td>5</td>
      <td>16603.768000</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Renault</td>
      <td>5</td>
      <td>19671.086000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Jensen</td>
      <td>12</td>
      <td>33874.750833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aptera</td>
      <td>10</td>
      <td>38091.509000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Fillmore</td>
      <td>5</td>
      <td>40284.594000</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Smart</td>
      <td>24</td>
      <td>41468.504583</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Panoz</td>
      <td>15</td>
      <td>42273.351333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bugatti</td>
      <td>12</td>
      <td>43246.830833</td>
    </tr>
    <tr>
      <th>48</th>
      <td>McLaren</td>
      <td>10</td>
      <td>43620.737000</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Merkur</td>
      <td>5</td>
      <td>44354.820000</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Studebaker</td>
      <td>18</td>
      <td>44504.211667</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Spyker</td>
      <td>46</td>
      <td>45419.070000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Morgan</td>
      <td>21</td>
      <td>45691.909524</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Subaru</td>
      <td>666</td>
      <td>46489.053423</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Isuzu</td>
      <td>371</td>
      <td>46998.392237</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Scion</td>
      <td>147</td>
      <td>47611.354150</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Shelby</td>
      <td>9</td>
      <td>47789.085556</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Suzuki</td>
      <td>594</td>
      <td>47818.181582</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Hummer</td>
      <td>160</td>
      <td>47990.759750</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cadillac</td>
      <td>706</td>
      <td>48091.230977</td>
    </tr>
  </tbody>
</table>
</div>




```python
#claim_freq and amount by model
df5 = df.groupby("car_model").agg({"claim_freq":"count","claim_amt":"mean"}).reset_index()

df5.sort_values(by="claim_amt", ascending= False).head(20)
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
      <th>car_model</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>277</th>
      <td>Crossfire Roadster</td>
      <td>1</td>
      <td>99403.060000</td>
    </tr>
    <tr>
      <th>904</th>
      <td>Truck Xtracab SR5</td>
      <td>3</td>
      <td>89085.600000</td>
    </tr>
    <tr>
      <th>302</th>
      <td>Datsun/Nissan Z-car</td>
      <td>4</td>
      <td>79579.117500</td>
    </tr>
    <tr>
      <th>597</th>
      <td>Minx Magnificent</td>
      <td>3</td>
      <td>79049.426667</td>
    </tr>
    <tr>
      <th>947</th>
      <td>Virage</td>
      <td>5</td>
      <td>78772.586000</td>
    </tr>
    <tr>
      <th>235</th>
      <td>Civic GX</td>
      <td>1</td>
      <td>77916.220000</td>
    </tr>
    <tr>
      <th>720</th>
      <td>Ram Wagon B350</td>
      <td>9</td>
      <td>74679.812222</td>
    </tr>
    <tr>
      <th>732</th>
      <td>Reliant</td>
      <td>7</td>
      <td>74519.141429</td>
    </tr>
    <tr>
      <th>218</th>
      <td>Carrera GT</td>
      <td>8</td>
      <td>73805.071250</td>
    </tr>
    <tr>
      <th>24</th>
      <td>300E</td>
      <td>8</td>
      <td>72414.543750</td>
    </tr>
    <tr>
      <th>601</th>
      <td>Mohave/Borrego</td>
      <td>5</td>
      <td>71998.466000</td>
    </tr>
    <tr>
      <th>157</th>
      <td>B2000</td>
      <td>9</td>
      <td>71840.060000</td>
    </tr>
    <tr>
      <th>264</th>
      <td>Corvair</td>
      <td>2</td>
      <td>70887.905000</td>
    </tr>
    <tr>
      <th>278</th>
      <td>Crosstour</td>
      <td>5</td>
      <td>70490.868000</td>
    </tr>
    <tr>
      <th>596</th>
      <td>Mini Cooper S</td>
      <td>4</td>
      <td>69996.442500</td>
    </tr>
    <tr>
      <th>603</th>
      <td>Monaro</td>
      <td>5</td>
      <td>69194.434000</td>
    </tr>
    <tr>
      <th>98</th>
      <td>924 S</td>
      <td>4</td>
      <td>69087.412500</td>
    </tr>
    <tr>
      <th>932</th>
      <td>Vega</td>
      <td>3</td>
      <td>68461.023333</td>
    </tr>
    <tr>
      <th>442</th>
      <td>GT350</td>
      <td>4</td>
      <td>68251.225000</td>
    </tr>
    <tr>
      <th>547</th>
      <td>Lumina APV</td>
      <td>4</td>
      <td>68199.450000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#claim_freq and amount by year
df5 = df.groupby("car_year").agg({"claim_freq":"count","claim_amt":"mean"}).reset_index()

df5.sort_values(by="claim_amt", ascending= False).head(20)
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
      <th>car_year</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1950</td>
      <td>3</td>
      <td>79049.426667</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1974</td>
      <td>20</td>
      <td>61584.947000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1948</td>
      <td>5</td>
      <td>58481.190000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1980</td>
      <td>35</td>
      <td>57445.832571</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1972</td>
      <td>38</td>
      <td>56149.655000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1964</td>
      <td>46</td>
      <td>54328.300435</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1978</td>
      <td>25</td>
      <td>53916.982400</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1965</td>
      <td>55</td>
      <td>53706.379273</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954</td>
      <td>17</td>
      <td>52876.774706</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1973</td>
      <td>33</td>
      <td>52634.385152</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1969</td>
      <td>45</td>
      <td>52533.357333</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1989</td>
      <td>519</td>
      <td>51829.169133</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1982</td>
      <td>15</td>
      <td>51543.304667</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1990</td>
      <td>538</td>
      <td>51506.414944</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1971</td>
      <td>20</td>
      <td>51481.488500</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2005</td>
      <td>1583</td>
      <td>51204.099046</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2013</td>
      <td>315</td>
      <td>51122.137270</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1984</td>
      <td>351</td>
      <td>50988.509772</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1995</td>
      <td>1375</td>
      <td>50980.734298</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1998</td>
      <td>1241</td>
      <td>50952.170943</td>
    </tr>
  </tbody>
</table>
</div>




```python
df6 = df.groupby("household_income").agg({"claim_freq":"sum","claim_amt":"sum"}).reset_index()

df6
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
      <th>household_income</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45004.91</td>
      <td>0</td>
      <td>64395.96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45008.78</td>
      <td>1</td>
      <td>19569.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45013.78</td>
      <td>0</td>
      <td>75276.59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45015.38</td>
      <td>0</td>
      <td>48201.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45030.64</td>
      <td>1</td>
      <td>86850.33</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>37497</th>
      <td>249961.94</td>
      <td>0</td>
      <td>1395.01</td>
    </tr>
    <tr>
      <th>37498</th>
      <td>249962.71</td>
      <td>0</td>
      <td>66491.35</td>
    </tr>
    <tr>
      <th>37499</th>
      <td>249965.70</td>
      <td>2</td>
      <td>35415.92</td>
    </tr>
    <tr>
      <th>37500</th>
      <td>249990.42</td>
      <td>0</td>
      <td>40064.00</td>
    </tr>
    <tr>
      <th>37501</th>
      <td>249991.11</td>
      <td>0</td>
      <td>89160.54</td>
    </tr>
  </tbody>
</table>
<p>37502 rows × 3 columns</p>
</div>




```python

# analyzing the relationship between household income, claim_amt and claim_freq using the mean values 

df7 = pd.DataFrame(df6)

# Calculate the correlation matrix
correlation_matrix = df7.corr()

# Display the correlation matrix
print(correlation_matrix)
```

                      household_income  claim_freq  claim_amt
    household_income          1.000000   -0.000758  -0.005118
    claim_freq               -0.000758    1.000000   0.002413
    claim_amt                -0.005118    0.002413   1.000000
    


```python
#analyzing the relationship btw education, claim_freq and amount
df8 = df.groupby("education").agg({"claim_freq":"mean","claim_amt":"mean"}).reset_index()

df8.sort_values(by="claim_freq", ascending= False)
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
      <th>education</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>High School</td>
      <td>0.514800</td>
      <td>49621.538801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Masters</td>
      <td>0.514539</td>
      <td>50193.972149</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Bachelors</td>
      <td>0.507994</td>
      <td>50275.535654</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PhD</td>
      <td>0.500362</td>
      <td>49556.277143</td>
    </tr>
  </tbody>
</table>
</div>




```python
#analyzing the relationship btw coverage zone claim_freq and amount
df9 = df.groupby("coverage_zone").agg({"claim_freq":"mean","claim_amt":"mean"}).reset_index()

df9.sort_values(by="claim_freq", ascending= False)
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
      <th>coverage_zone</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Suburban</td>
      <td>0.520091</td>
      <td>50124.843185</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Highly Urban</td>
      <td>0.516503</td>
      <td>49861.036665</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Urban</td>
      <td>0.508171</td>
      <td>50377.730389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rural</td>
      <td>0.506381</td>
      <td>49778.020247</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Highly Rural</td>
      <td>0.500403</td>
      <td>49998.132178</td>
    </tr>
  </tbody>
</table>
</div>




```python
#performing descriptive stats on numerical variables to identify frequent claim policyholders

df10 = df[df["claim_freq"] == 4]

df10.describe()
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
      <th>kids_driving</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
      <th>household_income</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1438.000000</td>
      <td>1438.000000</td>
      <td>1438.0</td>
      <td>1438.000000</td>
      <td>1438.000000</td>
      <td>1438.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.401252</td>
      <td>2000.436022</td>
      <td>4.0</td>
      <td>50170.250118</td>
      <td>146763.690841</td>
      <td>47.920723</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.718324</td>
      <td>8.589762</td>
      <td>0.0</td>
      <td>28431.191758</td>
      <td>59475.119405</td>
      <td>15.346042</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1954.000000</td>
      <td>4.0</td>
      <td>34.250000</td>
      <td>45272.890000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1995.000000</td>
      <td>4.0</td>
      <td>26302.072500</td>
      <td>95318.937500</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>2002.000000</td>
      <td>4.0</td>
      <td>48907.600000</td>
      <td>146000.525000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>2007.000000</td>
      <td>4.0</td>
      <td>74815.645000</td>
      <td>197238.022500</td>
      <td>61.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>2013.000000</td>
      <td>4.0</td>
      <td>99980.290000</td>
      <td>249864.340000</td>
      <td>74.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#performing descriptive stats on categorical variables to identify frequent claim policyholders
df11 = df10.dtypes[df10.dtypes == "object"].index

df10[df11].describe()
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
      <th>ID</th>
      <th>marital_status</th>
      <th>car_use</th>
      <th>gender</th>
      <th>parent</th>
      <th>education</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_color</th>
      <th>coverage_zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1438</td>
      <td>1438</td>
      <td>1438</td>
      <td>1438</td>
      <td>1438</td>
      <td>1438</td>
      <td>1438</td>
      <td>1438</td>
      <td>1438</td>
      <td>1438</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1438</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>58</td>
      <td>602</td>
      <td>19</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>79-8482952</td>
      <td>Single</td>
      <td>Private</td>
      <td>Male</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Ford</td>
      <td>Grand Prix</td>
      <td>Goldenrod</td>
      <td>Suburban</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>594</td>
      <td>1150</td>
      <td>721</td>
      <td>810</td>
      <td>709</td>
      <td>125</td>
      <td>15</td>
      <td>94</td>
      <td>301</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>kids_driving</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
      <th>household_income</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>37542.000000</td>
      <td>37542.000000</td>
      <td>37542.000000</td>
      <td>37542.000000</td>
      <td>37542.000000</td>
      <td>37542.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.418278</td>
      <td>2000.293005</td>
      <td>0.510308</td>
      <td>50028.514096</td>
      <td>147247.407750</td>
      <td>47.675617</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.736958</td>
      <td>9.045441</td>
      <td>1.015050</td>
      <td>28706.517988</td>
      <td>59145.588886</td>
      <td>15.296581</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1909.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>45004.910000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1995.000000</td>
      <td>0.000000</td>
      <td>25439.407500</td>
      <td>96162.182500</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>2002.000000</td>
      <td>0.000000</td>
      <td>49455.890000</td>
      <td>146674.895000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>74974.927500</td>
      <td>198277.420000</td>
      <td>61.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>2013.000000</td>
      <td>4.000000</td>
      <td>99997.700000</td>
      <td>249991.110000</td>
      <td>74.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df10 = df[df["claim_freq"] == 0]

df10.describe()
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
      <th>kids_driving</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
      <th>household_income</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27203.000000</td>
      <td>27203.000000</td>
      <td>27203.0</td>
      <td>27203.000000</td>
      <td>27203.000000</td>
      <td>27203.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.418814</td>
      <td>2000.260265</td>
      <td>0.0</td>
      <td>49991.836900</td>
      <td>147365.445102</td>
      <td>47.729148</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.735956</td>
      <td>9.054491</td>
      <td>0.0</td>
      <td>28718.808183</td>
      <td>59074.013373</td>
      <td>15.278435</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1909.000000</td>
      <td>0.0</td>
      <td>0.040000</td>
      <td>45004.910000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>1995.000000</td>
      <td>0.0</td>
      <td>25348.785000</td>
      <td>96602.170000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>2002.000000</td>
      <td>0.0</td>
      <td>49408.520000</td>
      <td>146718.720000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>2007.000000</td>
      <td>0.0</td>
      <td>74928.015000</td>
      <td>198294.555000</td>
      <td>61.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>2013.000000</td>
      <td>0.0</td>
      <td>99993.700000</td>
      <td>249991.110000</td>
      <td>74.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df11 = df10.dtypes[df10.dtypes == "object"].index

df10[df11].describe()
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
      <th>ID</th>
      <th>marital_status</th>
      <th>car_use</th>
      <th>gender</th>
      <th>parent</th>
      <th>education</th>
      <th>car_make</th>
      <th>car_model</th>
      <th>car_color</th>
      <th>coverage_zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>27203</td>
      <td>27203</td>
      <td>27203</td>
      <td>27203</td>
      <td>27203</td>
      <td>27203</td>
      <td>27203</td>
      <td>27203</td>
      <td>27203</td>
      <td>27203</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>27202</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>78</td>
      <td>1011</td>
      <td>19</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>56-5402470</td>
      <td>Single</td>
      <td>Private</td>
      <td>Male</td>
      <td>No</td>
      <td>Bachelors</td>
      <td>Ford</td>
      <td>Corvette</td>
      <td>Turquoise</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2</td>
      <td>11278</td>
      <td>21735</td>
      <td>13615</td>
      <td>15172</td>
      <td>13570</td>
      <td>2376</td>
      <td>183</td>
      <td>1516</td>
      <td>5475</td>
    </tr>
  </tbody>
</table>
</div>




```python
#demographic distribution by age_group
df12 = pd.crosstab(index = df["age_group"], columns = "count")
df12.sort_values(by="count", ascending=False)
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
      <th>col_0</th>
      <th>count</th>
    </tr>
    <tr>
      <th>age_group</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61-70</th>
      <td>7219</td>
    </tr>
    <tr>
      <th>31-40</th>
      <td>7128</td>
    </tr>
    <tr>
      <th>41-50</th>
      <td>7058</td>
    </tr>
    <tr>
      <th>51-60</th>
      <td>7036</td>
    </tr>
    <tr>
      <th>21-30</th>
      <td>5875</td>
    </tr>
    <tr>
      <th>71-80</th>
      <td>3226</td>
    </tr>
  </tbody>
</table>
</div>




```python
#demographic distribution by gender
df12 = pd.crosstab(index = df["gender"], columns = "count")
df12.sort_values(by="count", ascending=False)
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
      <th>col_0</th>
      <th>count</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>18806</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>18736</td>
    </tr>
  </tbody>
</table>
</div>




```python
#demographic distribution by marital_status
df12 = pd.crosstab(index = df["marital_status"], columns = "count")
df12.sort_values(by="count", ascending=False)
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
      <th>col_0</th>
      <th>count</th>
    </tr>
    <tr>
      <th>marital_status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Single</th>
      <td>15525</td>
    </tr>
    <tr>
      <th>Married</th>
      <td>12570</td>
    </tr>
    <tr>
      <th>Divorced</th>
      <td>6357</td>
    </tr>
    <tr>
      <th>Seperated</th>
      <td>3090</td>
    </tr>
  </tbody>
</table>
</div>




```python
#car use by age_group
df13 = pd.crosstab(index = df["age_group"], columns = df["car_use"] ,margins= True)
df13
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
      <th>car_use</th>
      <th>Commercial</th>
      <th>Private</th>
      <th>All</th>
    </tr>
    <tr>
      <th>age_group</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21-30</th>
      <td>1142</td>
      <td>4733</td>
      <td>5875</td>
    </tr>
    <tr>
      <th>31-40</th>
      <td>1453</td>
      <td>5675</td>
      <td>7128</td>
    </tr>
    <tr>
      <th>41-50</th>
      <td>1464</td>
      <td>5594</td>
      <td>7058</td>
    </tr>
    <tr>
      <th>51-60</th>
      <td>1375</td>
      <td>5661</td>
      <td>7036</td>
    </tr>
    <tr>
      <th>61-70</th>
      <td>1426</td>
      <td>5793</td>
      <td>7219</td>
    </tr>
    <tr>
      <th>71-80</th>
      <td>622</td>
      <td>2604</td>
      <td>3226</td>
    </tr>
    <tr>
      <th>All</th>
      <td>7482</td>
      <td>30060</td>
      <td>37542</td>
    </tr>
  </tbody>
</table>
</div>




```python
#claim freq and amount across coverage zone
df14 = df.groupby("coverage_zone").agg({"claim_freq":"mean","claim_amt":"mean"}).reset_index()

df14
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
      <th>coverage_zone</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Highly Rural</td>
      <td>0.500403</td>
      <td>49998.132178</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Highly Urban</td>
      <td>0.516503</td>
      <td>49861.036665</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rural</td>
      <td>0.506381</td>
      <td>49778.020247</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Suburban</td>
      <td>0.520091</td>
      <td>50124.843185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Urban</td>
      <td>0.508171</td>
      <td>50377.730389</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize the mean claim frequency and amount by coverage zone
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean Claim Frequency
sns.barplot(x='coverage_zone', y='claim_freq', data=df14, ax=axes[0])
axes[0].set_title('Mean Claim Frequency by Coverage Zone')
axes[0].set_xlabel('Coverage Zone')
axes[0].set_ylabel('Mean Claim Frequency')

# Mean Claim Amount
sns.barplot(x='coverage_zone', y='claim_amt', data=df14, ax=axes[1])
axes[1].set_title('Mean Claim Amount by Coverage Zone')
axes[1].set_xlabel('Coverage Zone')
axes[1].set_ylabel('Mean Claim Amount')

plt.tight_layout()
plt.show()
```


    
![png](output_37_0.png)
    



```python
#trends or pattern of policyholders with kids driving
df15 = df[df["kids_driving"] ==0]

df15.describe()
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
      <th>kids_driving</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
      <th>household_income</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>26685.0</td>
      <td>26685.000000</td>
      <td>26685.000000</td>
      <td>26685.000000</td>
      <td>26685.000000</td>
      <td>26685.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0</td>
      <td>2000.270452</td>
      <td>0.511973</td>
      <td>50107.541841</td>
      <td>147224.082152</td>
      <td>47.600712</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>9.027434</td>
      <td>1.016566</td>
      <td>28802.963773</td>
      <td>58970.576708</td>
      <td>15.304934</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>1909.000000</td>
      <td>0.000000</td>
      <td>19.700000</td>
      <td>45008.780000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0</td>
      <td>1995.000000</td>
      <td>0.000000</td>
      <td>25416.700000</td>
      <td>96245.460000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0</td>
      <td>2002.000000</td>
      <td>0.000000</td>
      <td>49570.410000</td>
      <td>146713.210000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>75189.730000</td>
      <td>197980.730000</td>
      <td>61.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.0</td>
      <td>2013.000000</td>
      <td>4.000000</td>
      <td>99997.700000</td>
      <td>249965.700000</td>
      <td>74.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#impact of presence of kids driving 
df15 = df[df["kids_driving"] == 0]

df15.describe()
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
      <th>kids_driving</th>
      <th>car_year</th>
      <th>claim_freq</th>
      <th>claim_amt</th>
      <th>household_income</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>26685.0</td>
      <td>26685.000000</td>
      <td>26685.000000</td>
      <td>26685.000000</td>
      <td>26685.000000</td>
      <td>26685.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0</td>
      <td>2000.270452</td>
      <td>0.511973</td>
      <td>50107.541841</td>
      <td>147224.082152</td>
      <td>47.600712</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>9.027434</td>
      <td>1.016566</td>
      <td>28802.963773</td>
      <td>58970.576708</td>
      <td>15.304934</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>1909.000000</td>
      <td>0.000000</td>
      <td>19.700000</td>
      <td>45008.780000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0</td>
      <td>1995.000000</td>
      <td>0.000000</td>
      <td>25416.700000</td>
      <td>96245.460000</td>
      <td>34.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0</td>
      <td>2002.000000</td>
      <td>0.000000</td>
      <td>49570.410000</td>
      <td>146713.210000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0</td>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>75189.730000</td>
      <td>197980.730000</td>
      <td>61.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.0</td>
      <td>2013.000000</td>
      <td>4.000000</td>
      <td>99997.700000</td>
      <td>249965.700000</td>
      <td>74.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize the mean claim frequency and amount
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean Claim Frequency
sns.barplot(x='kids_driving_', y='claim_freq', data=grouped, ax=axes[0])
axes[0].set_title('Mean Claim Frequency: Children Driving vs No Children Driving')
axes[0].set_xlabel('Children Driving')
axes[0].set_ylabel('Mean Claim Frequency')
axes[0].set_xticklabels(['No', 'Yes'])

# Mean Claim Amount
sns.barplot(x='children_driving_', y='claim_amt_mean', data=grouped, ax=axes[1])
axes[1].set_title('Mean Claim Amount: Children Driving vs No Children Driving')
axes[1].set_xlabel('Children Driving')
axes[1].set_ylabel('Mean Claim Amount')
axes[1].set_xticklabels(['No', 'Yes'])

plt.tight_layout()
plt.show()
```


```python

```


```python

```
