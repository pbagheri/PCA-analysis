# PCA-analysis
This code perfomrs PCA and reduces diminesions from 6 to 1.
- This code is wtitten in Python and uses the PCA package from scikit-learn.
- The input is 600 variables that fall into 100 groups of 6 variables.
- The PCA is calculated ones for each group seprately, i.e. 100 times and once with stacking those 100 groups yo create only 6 variables and then perform PCA once to go from 6 dimnesions down to 100.

- The input data contains unique ids for each row and the columns are values of 600 variables and the times associated for those values. This data is from a survey, so the times are the response times during the survey.
