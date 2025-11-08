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

# --- Main Task Function ---
# All the previous logic is now wrapped in this single function
def run_rps_analysis():
    """
    This function contains the entire process of fetching data, calculating RPS,
    and sending the email report.
    """
    print(f"\n--- Starting Job at {datetime.datetime.now()} ---")
    
    # --- Inner RPS Calculation Function ---
    def calculate_market_rps(market_type='A'):
        print(f"\nFetching data for {market_type}-shares...")
        try:
            if market_type == 'A':
                stock_list_df = ak.stock_zh_a_spot_em()
            elif market_type == 'HK':
                stock_list_df = ak.stock_hk_spot_em()
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching stock list for {market_type}-shares: {e}")
            return pd.DataFrame()

        periods = {'RPS_1M': 21, 'RPS_3M': 63, 'RPS_6M': 126, 'RPS_12M': 252}
        price_changes = []
        
        for code in tqdm(stock_list_df['代码'], desc=f"Calculating {market_type}-shares RPS"):
            time.sleep(random.uniform(0.5, 2.4))
            try:
                end_date = datetime.datetime.now().strftime('%Y%m%d')
                start_date = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime('%Y%m%d')
                
                if market_type == 'A':
                    hist_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                else:
                    hist_df = ak.stock_hk_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")

                if hist_df.empty or len(hist_df) < 22:
                    continue

                changes = {'代码': code}
                for name, days in periods.items():
                    if len(hist_df) > days:
                        change_pct = (hist_df['收盘'].iloc[-1] - hist_df['收盘'].iloc[-days-1]) / hist_df['收盘'].iloc[-days-1]
                        changes[name.replace('RPS', 'Change')] = change_pct
                    else:
                        changes[name.replace('RPS', 'Change')] = None
                price_changes.append(changes)
            except Exception:
                continue
                
        if not price_changes:
            print(f"No data could be processed for {market_type}-shares.")
            return pd.DataFrame()

        changes_df = pd.DataFrame(price_changes)
        for name in periods.keys():
            change_col = name.replace('RPS', 'Change')
            changes_df[name] = changes_df[change_col].rank(pct=True, na_option='bottom') * 100

        final_df = pd.merge(stock_list_df, changes_df, on='代码', how='inner')
        base_cols = ['代码', '名称', '最新价', '涨跌幅', '总市值']
        rps_cols = ['RPS_1M', 'RPS_3M', 'RPS_6M', 'RPS_12M']
        pe_col = '市盈率-动态' if market_type == 'A' else '市盈率'
        cols_to_keep = base_cols + [pe_col] + rps_cols
        final_df = final_df[[col for col in cols_to_keep if col in final_df.columns]].dropna(subset=rps_cols)
        return final_df

    # --- Inner Email Function ---
    def send_email_notification(subject: str, html_content: str):
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
            print(f"\nSuccessfully sent summary email notification: '{subject}'")
        except Exception as e:
            print(f"\nFailed to send summary email. Error: {e}")
        finally:
            if server: server.quit()

    # --- Main Logic Execution ---
    a_share_rps_df = calculate_market_rps(market_type='A')
    hk_share_rps_df = calculate_market_rps(market_type='HK')

    output_filename = f'RPS_Market_Analysis_{datetime.date.today()}.xlsx'
    print(f"\nSaving full results to {output_filename}...")
    with pd.ExcelWriter(output_filename) as writer:
        if not a_share_rps_df.empty: a_share_rps_df.sort_values(by='RPS_12M', ascending=False).to_excel(writer, sheet_name='A-Shares_RPS', index=False)
        if not hk_share_rps_df.empty: hk_share_rps_df.sort_values(by='RPS_12M', ascending=False).to_excel(writer, sheet_name='HK-Shares_RPS', index=False)
    
    html_body = "<html><head><style>body { font-family: sans-serif; } table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; } th { background-color: #f2f2f2; } hr { border: 1px solid #f2f2f2; }</style></head><body>"
    email_has_content = False
    periods_to_report = [('1-Month', 'RPS_1M'), ('3-Month', 'RPS_3M'), ('6-Month', 'RPS_6M')]
    
    if not a_share_rps_df.empty:
        email_has_content = True
        html_body += "<h1>A-Shares RPS Leaders</h1>"
        for period_name, rps_col in periods_to_report:
            html_body += f"<h2>Top 20% by {period_name} RPS</h2>"
            df_sorted = a_share_rps_df.sort_values(by=rps_col, ascending=False)
            top_20_a = df_sorted.head(int(len(df_sorted) * 0.2))
            html_body += top_20_a.to_html(index=False, float_format='%.2f') + "<br>"
        html_body += "<hr>"

    if not hk_share_rps_df.empty:
        email_has_content = True
        html_body += "<h1>Hong Kong Shares RPS Leaders</h1>"
        for period_name, rps_col in periods_to_report:
            html_body += f"<h2>Top 20% by {period_name} RPS</h2>"
            df_sorted = hk_share_rps_df.sort_values(by=rps_col, ascending=False)
            top_20_hk = df_sorted.head(int(len(df_sorted) * 0.2))
            html_body += top_20_hk.to_html(index=False, float_format='%.2f') + "<br>"

    html_body += "</body></html>"
    
    if email_has_content:
        email_subject = f"Daily RPS Momentum Report - {datetime.date.today()}"
        send_email_notification(email_subject, html_body)
    
    print(f"\n--- Job Finished. Waiting for next scheduled run... ---")


# --- Scheduling Block ---
if __name__ == "__main__":
    # Define the timezone for Shanghai
    shanghai_tz = pytz.timezone('Asia/Shanghai')

    # Schedule the job for every workday at 21:00 Shanghai time
    schedule.every().monday.at("23:00", shanghai_tz).do(run_rps_analysis)
    schedule.every().tuesday.at("23:00", shanghai_tz).do(run_rps_analysis)
    schedule.every().wednesday.at("23:00", shanghai_tz).do(run_rps_analysis)
    schedule.every().thursday.at("23:00", shanghai_tz).do(run_rps_analysis)
    schedule.every().friday.at("23:00", shanghai_tz).do(run_rps_analysis)

    print("Scheduler started. The script will run the job at the scheduled times.")
    print("Press Ctrl+C to stop the script.")

    # --- Execution Loop ---
    # This loop runs forever, checking every minute to see if a scheduled job is due
    while True:
        schedule.run_pending()
        time.sleep(60) # Wait for 60 seconds before checking again