# NSE-plotting
Code to plot key price levels for stocks and index. Uses nsepy

This code will create a csv called stock.csv/index.csv at the code location 
And then read back to produce the output
[There are some redundant bits in the code]

Sample call for Index:
  gbm(0,200,"index","nifty 50")

Sample call for stocks:
  gbm(0,200,"stock","SBIN")

Above means start at 0 day i.e. today and go back 200 calendar days
