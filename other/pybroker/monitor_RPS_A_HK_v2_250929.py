import pandas as pd
import akshare as ak
from tqdm import tqdm
import datetime
import smtplib
import email.utils
from email.mime.text import MIMEText
import time
import random
import schedule # <-- New library for scheduling
import pytz     # <-- New library for timezones

# --- Reusable Email Function ---
def send_email_notification(subject: str, html_content: str):
    """Sends an email with the given subject and HTML content."""
    SENDER_EMAIL = '2309598788@qq.com'
    SENDER_PASSWORD = 'aoyqtzjhzmxaeafg'
    RECEIVER_EMAIL = 'sunweihao97@gmail.com'
    
    message = MIMEText(html_content, 'html', 'utf-8')
    message['To'] = email.utils.formataddr(('Sun', RECEIVER_EMAIL))
    message['From'] = email.utils.formataddr(('Weihao', SENDER_EMAIL))
    message['Subject'] = subject
    
    server = None
    try:
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg=message.as_string())
        print(f"\nSuccessfully sent email notification: '{subject}'")
    except Exception as e:
        print(f"\nFailed to send email. Error: {e}")
    finally:
        if server:
            server.quit()

# --- Combined Analysis Function ---
def run_combined_screener():
    """
    Screens stocks for both MA alignment and calculates their RPS scores.
    """
    print("--- Starting Combined MA and RPS Screener ---")
    
    # This list will hold raw data for all stocks that pass the MA screen
    ma_filtered_stocks = []
    
    for market_type in ['A', 'HK']:
        print(f"\nScreening {market_type}-shares...")
        try:
            if market_type == 'A':
                stock_list_df = ak.stock_zh_a_spot_em()
            else: # HK
                stock_list_df = ak.stock_hk_spot_em()
        except Exception as e:
            print(f"Could not fetch stock list for {market_type}-shares. Error: {e}")
            continue

        for index, row in tqdm(stock_list_df.iterrows(), total=len(stock_list_df)):
            # NEW: Sleep for 1-2 seconds between each stock
            time.sleep(random.uniform(1, 2))
            
            code = row['代码']
            
            try:
                # Fetch enough data for 12M RPS and 150-day MA
                end_date = datetime.datetime.now().strftime('%Y%m%d')
                start_date = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime('%Y%m%d')
                
                if market_type == 'A':
                    hist_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                else:
                    hist_df = ak.stock_hk_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                
                if len(hist_df) < 252: # Need enough data for all calculations
                    continue

                # 1. --- MA ANALYSIS ---
                hist_df['MA5'] = hist_df['收盘'].rolling(window=5).mean()
                hist_df['MA60'] = hist_df['收盘'].rolling(window=60).mean()
                hist_df['MA150'] = hist_df['收盘'].rolling(window=150).mean()
                hist_df.dropna(inplace=True)
                if hist_df.empty: continue

                latest = hist_df.iloc[-1]
                ma_alignment_ok = (latest['MA5'] > latest['MA60']) and (latest['MA60'] > latest['MA150'])
                ma60_slope_positive = latest['MA60'] > hist_df['MA60'].iloc[-5]

                # If MA conditions are not met, skip to the next stock
                if not (ma_alignment_ok and ma60_slope_positive):
                    continue

                # Check for recent crossovers
                recent_data = hist_df.iloc[-30:]
                price_cross_ma60 = any((recent_data['收盘'] > recent_data['MA60']) & (recent_data['收盘'].shift(1) < recent_data['MA60'].shift(1)))
                ma60_cross_ma150 = any((recent_data['MA60'] > recent_data['MA150']) & (recent_data['MA60'].shift(1) < recent_data['MA150'].shift(1)))

                # 2. --- RPS RAW CALCULATION ---
                # If MA screen passed, calculate raw price changes for RPS
                periods = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}
                changes = {}
                for name, days in periods.items():
                    change_pct = (hist_df['收盘'].iloc[-1] - hist_df['收盘'].iloc[-days-1]) / hist_df['收盘'].iloc[-days-1]
                    changes[f'Change_{name}'] = change_pct

                # 3. --- STORE RESULTS ---
                # Store all data for stocks that passed the MA screen
                stock_data = row.to_dict()
                stock_data['Price_Cross_MA60_Recent'] = price_cross_ma60
                stock_data['MA60_Cross_MA150_Recent'] = ma60_cross_ma150
                stock_data.update(changes)
                ma_filtered_stocks.append(stock_data)

            except Exception as e:
                continue
    
    if not ma_filtered_stocks:
        print("\nNo stocks found meeting the MA criteria.")
        return

    # --- Post-Processing: Calculate Final RPS Scores ---
    results_df = pd.DataFrame(ma_filtered_stocks)
    
    for period in ['1M', '3M', '6M', '12M']:
        change_col = f'Change_{period}'
        rps_col = f'RPS_{period}'
        # Rank the filtered stocks against each other
        results_df[rps_col] = results_df[change_col].rank(pct=True) * 100
        
    # --- Final Reporting ---
    # Select and reorder columns for the final report
    final_cols = [
        '代码', '名称', '最新价', '涨跌幅', '总市值',
        'RPS_1M', 'RPS_3M', 'RPS_6M', 'RPS_12M',
        'Price_Cross_MA60_Recent', 'MA60_Cross_MA150_Recent'
    ]
    report_df = results_df[[col for col in final_cols if col in results_df.columns]]
    report_df = report_df.sort_values(by='RPS_6M', ascending=False) # Sort by 6-month momentum

    # Save to Excel
    output_filename = f'Combined_Screener_Report_{datetime.date.today()}.xlsx'
    report_df.to_excel(output_filename, index=False)
    print(f"\nSaved report to {output_filename}")

    # Send Email
    html_body = f"""
    <html><head><style>
        body {{ font-family: sans-serif; }} table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
        th {{ background-color: #f2f2f2; }}
    </style></head><body>
    <h1>Combined MA & RPS Screener Report</h1>
    <p>The following stocks meet the MA alignment criteria (MA5 > MA60 > MA150, positive MA60 slope).</p>
    <p>RPS scores are calculated relative to this filtered group.</p>
    {report_df.to_html(index=False, float_format='%.2f')}
    </body></html>
    """
    email_subject = f"Combined MA & RPS Report - {datetime.date.today()}"
    send_email_notification(email_subject, html_body)

# --- Main Execution Block ---
# --- Main Execution Block with Scheduler ---
if __name__ == "__main__":
    # Define the timezone for Shanghai
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    run_combined_screener()  # Run once immediately on startup

#    # Schedule the job for every workday at 21:00 Shanghai time
#    # NOTE: You can change the time here if needed
#    schedule.every().monday.at("21:00", shanghai_tz).do(run_combined_screener)
#    #schedule.every().tuesday.at("21:00", shanghai_tz).do(run_combined_screener)
#    schedule.every().wednesday.at("21:00", shanghai_tz).do(run_combined_screener)
#    #schedule.every().thursday.at("21:00", shanghai_tz).do(run_combined_screener)
#    schedule.every().friday.at("09:15", shanghai_tz).do(run_combined_screener)
#
#    print("Scheduler started. The script will run the combined analysis at the scheduled times.")
#    print("Keep this window open. Press Ctrl+C to stop.")
#
#    # --- Execution Loop ---
#    # This loop runs forever, checking every minute to see if a scheduled job is due
#    while True:
#        schedule.run_pending()
#        time.sleep(60) # Wait for 60 seconds before checking again