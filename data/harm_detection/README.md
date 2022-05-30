This directory contains the data used for the extrinsic evaluation on Harm Detection. 

- `keyword_list.txt`: A list of the harm-related keywords we used to match queries from wikiHow. Each line is a keyword.
- `test.csv`: The harm detection test set used for extrinsic evaluation. 
Each row contains a Query (typically the form of "how to ..."), a Label (HARM means harmful; GOOD means harmless), and the keyword by which the Query is retrieved from wikiHow. 