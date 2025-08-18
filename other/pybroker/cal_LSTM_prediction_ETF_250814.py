import akshare as ak

fund_etf_hist_em_df = ak.fund_etf_hist_em(symbol="560300", period="daily", start_date="20240101", end_date="20280801", adjust="qfq")
print(fund_etf_hist_em_df)