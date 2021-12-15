# Case-Study-UCI-Bank-Marketing-Dataset

So this is a case study using the UCI Bank Marketing Dataset. This case study will consist of several parts.

The following information is drawn for the most part from the UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/bank+marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

**Abstract:**

The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

**Source:**

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

**Data Set Information:**

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

There are four datasets:

1) `bank-additional-full.csv` with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
2) `bank-additional.csv` with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
3) `bank-full.csv` with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
4) `bank.csv` with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).

> For this case study I used the dataset 1 - `bank-additional-full.csv`

**The classification goal** is to predict if the client will subscribe (yes/no) a term deposit (variable y).

# Attribute information
## Input variables
### Bank client data

1. `age` (numeric)
2. `job` : type of job (categorical)
3. `marital` : marital status (categorical)
4. `education` (categorical)
5. `default`: has credit in default? (categorical)
6. `housing`: has housing loan? (categorical)
7. `loan`: has personal loan? (categorical)

### Related with the last contact of the current campaign

8. `contact`: contact communication type (categorical)
9. `month`: last contact month of year (categorical)
10. `day_of_week`: last contact day of the week (categorical)
11. `duration`: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

### Other attributes

12. `campaign`: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. `pdays`: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. `previous`: number of contacts performed before this campaign and for this client (numeric)
15. `poutcome`: outcome of the previous marketing campaign (categorical)

### Social and economic context attributes

16. `emp.var.rate`: employment variation rate - quarterly indicator (numeric)
17. `cons.price.idx`: consumer price index - monthly indicator (numeric)
18. `cons.conf.idx`: consumer confidence index - monthly indicator (numeric)
19. `euribor3m`: euribor 3 month rate - daily indicator (numeric)
20. `nr.employed`: number of employees - quarterly indicator (numeric)

## Output variable (desired target)

21. `y` - has the client subscribed a term deposit? (binary: 'yes','no')



