import pandas as pd
import akshare as ak
import datetime
import smtplib
import email.utils
from email.mime.text import MIMEText
import time
import random
import schedule
import pytz

# --- Configuration (Moved to top for easy access) ---
SENDER_EMAIL = '2309598788@qq.com'
SENDER_PASSWORD = 'aoyqtzjhzmxaeafg'
RECEIVER_EMAIL = 'sunweihao97@gmail.com'

# --- Reusable Email Function ---
def send_email_notification(subject: str, html_content: str):
    """Sends an email with the given subject and HTML content."""
    message = MIMEText(html_content, 'html', 'utf-8')
    message['To'] = email.utils.formataddr(('Sun', RECEIVER_EMAIL))
    message['From'] = email.utils.formataddr(('Weihao', SENDER_EMAIL))
    message['Subject'] = subject
    
    server = None
    try:
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg=message.as_string())
        print(f"\n✅ Successfully sent email notification: '{subject}'")
    except Exception as e:
        print(f"\n❌ Failed to send email. Error: {e}")
    finally:
        if server:
            server.quit()

# --- Combined Analysis Function ---
def run_combined_screener():
    """
    Screens stocks for both MA alignment and calculates their RPS scores.
    """
    print(f"--- Starting Combined MA and RPS Screener at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    ma_filtered_stocks = []
    
    for market_type in ['A', 'HK']:
        print(f"\nFetching stock list for {market_type}-shares...")
        try:
            if market_type == 'A':
                stock_list_df = ak.stock_zh_a_spot_em()
            else: # HK
                stock_list_df = ak.stock_hk_spot_em()
            print(f"Found {len(stock_list_df)} stocks in {market_type}-shares market.")
        except Exception as e:
            print(f"❌ Could not fetch stock list for {market_type}-shares. Error: {e}")
            continue

        # REFINED: Loop with a counter instead of tqdm
        total_stocks = len(stock_list_df)
        for index, row in stock_list_df.iterrows():
            stock_num = index + 1
            code = row['代码']
            name = row['名称']
            
            # REFINED: Informative print statement for progress tracking
            print(f"[{stock_num}/{total_stocks}] Screening {code} ({name})... ", end="")
            
            # REFINED: Reduced sleep time to speed up the process
            time.sleep(random.uniform(1, 3))
            
            try:
                # *** CRITICAL FIX ***
                # Increased timedelta to fetch more data.
                # We need at least 150 days for MA150 + 252 days for 12M RPS = 402 days.
                # Fetching 500 days provides a safe buffer.
                end_date = datetime.datetime.now().strftime('%Y%m%d')
                start_date = (datetime.datetime.now() - datetime.timedelta(days=1000)).strftime('%Y%m%d')
                
                if market_type == 'A':
                    hist_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                else:
                    hist_df = ak.stock_hk_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                
                # Check for sufficient data length BEFORE calculations
                if len(hist_df) < 402:
                    print("-> Rejected: Insufficient historical data.")
                    continue

                # 1. --- MA ANALYSIS ---
                hist_df['MA5'] = hist_df['收盘'].rolling(window=5).mean()
                hist_df['MA60'] = hist_df['收盘'].rolling(window=60).mean()
                hist_df['MA150'] = hist_df['收盘'].rolling(window=150).mean()
                
                # We need a copy of the full history for RPS calculation later
                full_history = hist_df.copy()
                
                hist_df.dropna(inplace=True)
                if hist_df.empty:
                    print("-> Rejected: Data empty after MA calculation.")
                    continue

                latest = hist_df.iloc[-1]
                
                # Check MA conditions
                ma_alignment_ok = (latest['MA5'] > latest['MA60']) and (latest['MA60'] > latest['MA150'])
                if not ma_alignment_ok:
                    print("-> Rejected: MA alignment (5 > 60 > 150) not met.")
                    continue
                
                ma60_slope_positive = latest['MA60'] > hist_df['MA60'].iloc[-5]
                if not ma60_slope_positive:
                    print("-> Rejected: MA60 slope is not positive.")
                    continue

                # If we reach here, the stock has passed the MA screen
                print("✅ Passed MA screen. Calculating RPS...")

                # Check for recent crossovers (for information, not filtering)
                recent_data = hist_df.iloc[-30:]
                price_cross_ma60 = any((recent_data['收盘'] > recent_data['MA60']) & (recent_data['收盘'].shift(1) < recent_data['MA60'].shift(1)))
                ma60_cross_ma150 = any((recent_data['MA60'] > recent_data['MA150']) & (recent_data['MA60'].shift(1) < recent_data['MA150'].shift(1)))

                # 2. --- RPS RAW CALCULATION ---
                # Use the full_history DataFrame here before rows were dropped
                periods = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}
                changes = {}
                for name, days in periods.items():
                    # Ensure we have enough data points in the full history
                    if len(full_history) > days:
                        change_pct = (full_history['收盘'].iloc[-1] - full_history['收盘'].iloc[-days-1]) / full_history['收盘'].iloc[-days-1]
                        changes[f'Change_{name}'] = change_pct
                    else:
                        changes[f'Change_{name}'] = None # Not enough data for this period

                # 3. --- STORE RESULTS ---
                stock_data = row.to_dict()
                stock_data['Price_Cross_MA60_Recent'] = price_cross_ma60
                stock_data['MA60_Cross_MA150_Recent'] = ma60_cross_ma150
                stock_data.update(changes)
                ma_filtered_stocks.append(stock_data)

            # REFINED: More informative error handling
            except Exception as e:
                print(f"-> ERROR: An exception occurred for {code}: {e}")
                continue
    
    if not ma_filtered_stocks:
        print("\n--- Screener Finished: No stocks found meeting all criteria. ---")
        return

    print(f"\n--- Found {len(ma_filtered_stocks)} stocks meeting MA criteria. Calculating final RPS scores... ---")
    
    # --- Post-Processing: Calculate Final RPS Scores ---
    results_df = pd.DataFrame(ma_filtered_stocks)
    
    for period in ['1M', '3M', '6M', '12M']:
        change_col = f'Change_{period}'
        rps_col = f'RPS_{period}'
        # Rank the filtered stocks against each other and handle potential missing data
        results_df[rps_col] = results_df[change_col].rank(pct=True, na_option='bottom') * 100
        
    # --- Final Reporting ---
    final_cols = [
        '代码', '名称', '最新价', '涨跌幅', '总市值',
        'RPS_1M', 'RPS_3M', 'RPS_6M', 'RPS_12M',
        'Price_Cross_MA60_Recent', 'MA60_Cross_MA150_Recent'
    ]
    report_df = results_df[[col for col in final_cols if col in results_df.columns]]
    report_df = report_df.sort_values(by='RPS_6M', ascending=False)

    # Save to Excel
    output_filename = f'Combined_Screener_Report_{datetime.date.today()}.xlsx'
    report_df.to_excel(output_filename, index=False)
    print(f"\n✅ Report saved to {output_filename}")

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
if __name__ == "__main__":
    shanghai_tz = pytz.timezone('Asia/Shanghai')
    run_combined_screener()  # Run once immediately on startup

    # Your scheduler code remains here if you wish to use it
    # print("\nScheduler configured. Keep this window open for scheduled runs.")
    # schedule.every().friday.at("09:15", shanghai_tz).do(run_combined_screener)
    #
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)