import akshare as ak

# 请确保设置了 adjust 不为空
stock_zh_a_hist_df = ak.stock_zh_a_hist(
    symbol="000001",
    period="daily",
    start_date="20170301",
    end_date="20240528",
    adjust="qfq"
)

print(stock_zh_a_hist_df.head())
