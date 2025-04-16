import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import time

# تكوين الصفحة
st.set_page_config(
    page_title="لوحة تحكم تحليل العملات الرقمية",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تعريف نطاق العملات المدعومة
COIN_IDS = {
    "ETH": "ethereum",
    "BTC": "bitcoin",
    "SOL": "solana",
    "CRO": "crypto-com-chain",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "XRP": "ripple"
}

# توجيهات القرارات
DECISION_GUIDES = {
    "STRONG BUY": {"color": "darkgreen", "icon": "🔥", "text": "فرصة شراء قوية: المؤشرات تشير إلى احتمالية ارتفاع كبير"},
    "BUY": {"color": "green", "icon": "✅", "text": "شراء: توجد إشارات إيجابية تدعم الشراء"},
    "HOLD": {"color": "gray", "icon": "⏹️", "text": "احتفاظ: الأفضل الانتظار وعدم اتخاذ إجراء الآن"},
    "SELL": {"color": "red", "icon": "⚠️", "text": "بيع: توجد إشارات سلبية تدعم البيع"},
    "STRONG SELL": {"color": "darkred", "icon": "🛑", "text": "بيع قوي: المؤشرات تشير إلى احتمالية هبوط كبير"}
}

# دوال المساعدة
@st.cache_data(ttl=300) # تخزين مؤقت لمدة 5 دقائق
def get_price_data(coin_id, vs_currency="usd", days=90):
    """الحصول على بيانات أسعار العملة من CoinGecko"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily" if days > 30 else "hourly"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # استخراج البيانات
        prices_data = data["prices"]
        timestamps = [datetime.fromtimestamp(price[0]/1000) for price in prices_data]
        prices = [price[1] for price in prices_data]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })
        
        return df
    except Exception as e:
        st.error(f"خطأ في الحصول على البيانات: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_current_prices(symbols):
    """الحصول على الأسعار الحالية من CoinGecko"""
    coin_ids = [COIN_IDS.get(symbol, symbol.lower()) for symbol in symbols]
    coins_str = ','.join(coin_ids)
    
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coins_str}&vs_currencies=usd&include_24hr_change=true"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        prices = {}
        for symbol in symbols:
            coin_id = COIN_IDS.get(symbol, symbol.lower())
            if coin_id in data:
                prices[symbol] = {
                    'price': data[coin_id]['usd'],
                    'change_24h': data[coin_id].get('usd_24h_change', 0)
                }
                
        return prices
    except Exception as e:
        st.error(f"خطأ في الحصول على الأسعار الحالية: {str(e)}")
        return {}

def calculate_rsi(prices, window=14):
    """حساب مؤشر القوة النسبية (RSI)"""
    if len(prices) < window+1:
        return 50
        
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1]

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """حساب مؤشر MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    if histogram.iloc[-1] > 0 and histogram.iloc[-2] < 0:
        signal = "buy"
    elif histogram.iloc[-1] < 0 and histogram.iloc[-2] > 0:
        signal = "sell"
    else:
        signal = "neutral"
        
    return {
        'macd_line': macd_line.iloc[-1],
        'signal_line': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1],
        'signal': signal
    }

def get_market_sentiment():
    """الحصول على مؤشر الخوف والطمع"""
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url)
        data = response.json()
        value = int(data["data"][0]["value"])
        
        if value <= 20:
            return "خوف شديد", value
        elif value <= 40:
            return "خوف", value
        elif value <= 60:
            return "محايد", value
        elif value <= 80:
            return "طمع", value
        else:
            return "طمع شديد", value
    except:
        return "غير متاح", 50

def analyze_crypto(symbol, prices_df):
    """تحليل العملة المشفرة"""
    if prices_df.empty:
        return None
        
    current_price = prices_df['price'].iloc[-1]
    
    # حساب المؤشرات الفنية
    rsi = calculate_rsi(prices_df['price'])
    macd = calculate_macd(prices_df['price'])
    
    # حساب المتوسطات المتحركة
    prices_df['ma50'] = prices_df['price'].rolling(window=50).mean()
    prices_df['ma200'] = prices_df['price'].rolling(window=200).mean()
    
    ma50_current = prices_df['ma50'].iloc[-1] if not pd.isna(prices_df['ma50'].iloc[-1]) else 0
    ma200_current = prices_df['ma200'].iloc[-1] if not pd.isna(prices_df['ma200'].iloc[-1]) else 0
    
    golden_cross = ma50_current > ma200_current
    
    # توليد توصية
    buy_score = 0
    sell_score = 0
    explanations = []
    
    # تحليل RSI
    if rsi <= 30:
        buy_score += 2
        explanations.append(f"RSI في منطقة التشبع البيعي ({rsi:.1f}) - إشارة شراء قوية")
    elif rsi <= 40:
        buy_score += 1
        explanations.append(f"RSI منخفض ({rsi:.1f}) - إشارة شراء متوسطة")
    elif rsi >= 70:
        sell_score += 2
        explanations.append(f"RSI في منطقة التشبع الشرائي ({rsi:.1f}) - إشارة بيع قوية")
    elif rsi >= 60:
        sell_score += 1
        explanations.append(f"RSI مرتفع ({rsi:.1f}) - إشارة بيع متوسطة")
    
    # تحليل MACD
    if macd['signal'] == "buy":
        buy_score += 1.5
        explanations.append("MACD يظهر تقاطع صعودي - إشارة شراء")
    elif macd['signal'] == "sell":
        sell_score += 1.5
        explanations.append("MACD يظهر تقاطع هبوطي - إشارة بيع")
        
    # تحليل المتوسطات المتحركة
    if golden_cross:
        buy_score += 1.5
        explanations.append("المتوسط المتحرك 50 يوم فوق المتوسط المتحرك 200 يوم - إشارة صعودية")
    else:
        sell_score += 1
        explanations.append("المتوسط المتحرك 50 يوم تحت المتوسط المتحرك 200 يوم - إشارة هبوطية")
    
    # توليد التوصية النهائية
    if buy_score >= 3 and buy_score > sell_score:
        recommendation = "STRONG BUY"
    elif buy_score >= 1.5 and buy_score > sell_score:
        recommendation = "BUY"
    elif sell_score >= 3 and sell_score > buy_score:
        recommendation = "STRONG SELL"
    elif sell_score >= 1.5 and sell_score > buy_score:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    # حساب الأهداف السعرية
    if recommendation in ["STRONG BUY", "BUY"]:
        short_term_pct = 0.10 if recommendation == "BUY" else 0.15
        long_term_pct = 0.25 if recommendation == "BUY" else 0.35
        stop_loss_pct = 0.07 if recommendation == "BUY" else 0.10
    elif recommendation in ["STRONG SELL", "SELL"]:
        short_term_pct = -0.10 if recommendation == "SELL" else -0.15
        long_term_pct = -0.20 if recommendation == "SELL" else -0.30
        stop_loss_pct = 0.05
    else:
        short_term_pct = 0.05
        long_term_pct = 0.12
        stop_loss_pct = 0.05
    
    short_term_target = current_price * (1 + short_term_pct)
    long_term_target = current_price * (1 + long_term_pct)
    stop_loss = current_price * (1 - stop_loss_pct)
    
    return {
        'symbol': symbol,
        'price': current_price,
        'rsi': rsi,
        'macd': macd,
        'golden_cross': golden_cross,
        'recommendation': recommendation,
        'buy_score': buy_score,
        'sell_score': sell_score,
        'explanation': explanations,
        'short_term_target': short_term_target,
        'long_term_target': long_term_target,
        'stop_loss': stop_loss
    }

def create_price_chart(symbol, prices_df, analysis):
    """إنشاء رسم بياني للسعر مع المؤشرات"""
    if prices_df.empty:
        return None
    
    # إضافة المتوسطات المتحركة
    prices_df['ma50'] = prices_df['price'].rolling(window=50).mean()
    prices_df['ma200'] = prices_df['price'].rolling(window=200).mean()
    
    # إنشاء الرسم البياني
    fig = go.Figure()
    
    # إضافة خط السعر
    fig.add_trace(go.Scatter(
        x=prices_df['timestamp'], 
        y=prices_df['price'],
        mode='lines',
        name='السعر',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # إضافة المتوسطات المتحركة
    if not prices_df['ma50'].isna().all():
        fig.add_trace(go.Scatter(
            x=prices_df['timestamp'],
            y=prices_df['ma50'],
            mode='lines',
            name='المتوسط المتحرك 50 يوم',
            line=dict(color='#FFA000', width=1.5, dash='dot')
        ))
        
    if not prices_df['ma200'].isna().all():
        fig.add_trace(go.Scatter(
            x=prices_df['timestamp'],
            y=prices_df['ma200'],
            mode='lines',
            name='المتوسط المتحرك 200 يوم',
            line=dict(color='#D81B60', width=1.5, dash='dot')
        ))
    
    # إضافة الأهداف السعرية إذا كانت متوفرة
    if analysis:
        current_price = analysis['price']
        short_term_target = analysis['short_term_target']
        long_term_target = analysis['long_term_target']
        stop_loss = analysis['stop_loss']
        
        # نقطة السعر الحالي
        fig.add_trace(go.Scatter(
            x=[prices_df['timestamp'].iloc[-1]],
            y=[current_price],
            mode='markers',
            name=f'السعر الحالي: ${current_price:.2f}',
            marker=dict(color='red', size=10)
        ))
        
        # خطوط الأهداف
        fig.add_shape(
            type="line",
            x0=prices_df['timestamp'].min(),
            x1=prices_df['timestamp'].max(),
            y0=short_term_target,
            y1=short_term_target,
            line=dict(color="green", width=2, dash="dash"),
            name=f'هدف قصير المدى: ${short_term_target:.2f}'
        )
        
        fig.add_shape(
            type="line",
            x0=prices_df['timestamp'].min(),
            x1=prices_df['timestamp'].max(),
            y0=long_term_target,
            y1=long_term_target,
            line=dict(color="purple", width=2, dash="dash"),
            name=f'هدف طويل المدى: ${long_term_target:.2f}'
        )
        
        fig.add_shape(
            type="line",
            x0=prices_df['timestamp'].min(),
            x1=prices_df['timestamp'].max(),
            y0=stop_loss,
            y1=stop_loss,
            line=dict(color="red", width=2, dash="dash"),
            name=f'وقف الخسارة: ${stop_loss:.2f}'
        )
        
        # إضافة تسميات للأهداف
        fig.add_annotation(
            x=prices_df['timestamp'].max(),
            y=short_term_target,
            text=f"هدف قصير: ${short_term_target:.2f} (+{((short_term_target/current_price)-1)*100:.1f}%)",
            showarrow=True,
            arrowhead=1,
            ax=80,
            ay=0,
            bgcolor="rgba(0,255,0,0.7)",
            font=dict(color="white")
        )
        
        fig.add_annotation(
            x=prices_df['timestamp'].max(),
            y=long_term_target,
            text=f"هدف طويل: ${long_term_target:.2f} (+{((long_term_target/current_price)-1)*100:.1f}%)",
            showarrow=True,
            arrowhead=1,
            ax=80,
            ay=0,
            bgcolor="rgba(128,0,128,0.7)",
            font=dict(color="white")
        )
        
        fig.add_annotation(
            x=prices_df['timestamp'].max(),
            y=stop_loss,
            text=f"وقف الخسارة: ${stop_loss:.2f} (-{((current_price-stop_loss)/current_price)*100:.1f}%)",
            showarrow=True,
            arrowhead=1,
            ax=80,
            ay=0,
            bgcolor="rgba(255,0,0,0.7)",
            font=dict(color="white")
        )
    
    # تكوين الرسم البياني
    fig.update_layout(
        title=f"سعر {symbol} (USD)",
        xaxis_title="التاريخ",
        yaxis_title="السعر (USD)",
        template="plotly_white",
        legend=dict(x=0, y=1, orientation="h"),
        height=500
    )
    
    return fig

def create_indicator_charts(symbol, prices_df):
    """إنشاء رسومات بيانية للمؤشرات الفنية"""
    if prices_df.empty:
        return None, None
        
    # حساب مؤشر RSI
    rsi_values = prices_df['price'].rolling(window=14).apply(
        lambda x: calculate_rsi(pd.Series(x))
    )
    
    # حساب مؤشر MACD
    ema12 = prices_df['price'].ewm(span=12, adjust=False).mean()
    ema26 = prices_df['price'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # إنشاء رسم بياني RSI
    rsi_fig = go.Figure()
    
    rsi_fig.add_trace(go.Scatter(
        x=prices_df['timestamp'],
        y=rsi_values,
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    # إضافة خطوط للمناطق المهمة
    rsi_fig.add_shape(
        type="line",
        x0=prices_df['timestamp'].min(),
        x1=prices_df['timestamp'].max(),
        y0=70,
        y1=70,
        line=dict(color="red", width=1),
    )
    
    rsi_fig.add_shape(
        type="line",
        x0=prices_df['timestamp'].min(),
        x1=prices_df['timestamp'].max(),
        y0=30,
        y1=30,
        line=dict(color="green", width=1),
    )
    
    # إضافة مناطق ملونة
    rsi_fig.add_trace(go.Scatter(
        x=prices_df['timestamp'],
        y=[70] * len(prices_df),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0),
        showlegend=False
    ))
    
    rsi_fig.add_trace(go.Scatter(
        x=prices_df['timestamp'],
        y=[30] * len(prices_df),
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(width=0),
        showlegend=False
    ))
    
    rsi_fig.update_layout(
        title="مؤشر القوة النسبية (RSI)",
        xaxis_title="التاريخ",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        height=250
    )
    
    # إنشاء رسم بياني MACD
    macd_fig = go.Figure()
    
    # إضافة خطوط MACD
    macd_fig.add_trace(go.Scatter(
        x=prices_df['timestamp'],
        y=macd_line,
        mode='lines',
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    macd_fig.add_trace(go.Scatter(
        x=prices_df['timestamp'],
        y=signal_line,
        mode='lines',
        name='إشارة',
        line=dict(color='red', width=1.5)
    ))
    
    # إضافة الهستوجرام
    colors = ['green' if val >= 0 else 'red' for val in histogram]
    
    macd_fig.add_trace(go.Bar(
        x=prices_df['timestamp'],
        y=histogram,
        name='الهستوجرام',
        marker_color=colors
    ))
    
    macd_fig.update_layout(
        title="مؤشر تقارب وتباعد المتوسطات المتحركة (MACD)",
        xaxis_title="التاريخ",
        yaxis_title="MACD",
        height=250
    )
    
    return rsi_fig, macd_fig

def create_portfolio_input():
    """إنشاء واجهة إدخال بيانات المحفظة"""
    st.sidebar.header("📊 المحفظة")
    
    # استرجاع المحفظة الحالية من الجلسة
    portfolio = st.session_state.get('portfolio', {})
    
    # إنشاء نموذج لإضافة عملة جديدة
    with st.sidebar.expander("➕ إضافة عملة جديدة"):
        coin_options = list(COIN_IDS.keys())
        new_coin = st.selectbox("اختر العملة", coin_options)
        amount = st.number_input("الكمية", min_value=0.0, step=0.01)
        avg_price = st.number_input("متوسط سعر الشراء (USD)", min_value=0.0, step=0.1)
        
        if st.button("إضافة إلى المحفظة"):
            if new_coin and amount > 0:
                portfolio[new_coin] = {
                    'amount': amount,
                    'avg_buy_price': avg_price
                }
                st.session_state['portfolio'] = portfolio
                st.success(f"تمت إضافة {amount} {new_coin} إلى المحفظة!")
                st.experimental_rerun()
    
    # عرض المحفظة الحالية
    st.sidebar.subheader("عملاتك")
    
    if not portfolio:
        st.sidebar.info("لم تضف أي عملات بعد. استخدم النموذج أعلاه لإضافة عملات إلى محفظتك.")
    else:
        # الحصول على الأسعار الحالية
        current_prices = get_current_prices(list(portfolio.keys()))
        
        for coin, data in portfolio.items():
            with st.sidebar.expander(f"{coin} - {data['amount']}"):
                avg_price = data['avg_buy_price']
                current_price = current_prices.get(coin, {}).get('price', 0)
                change_24h = current_prices.get(coin, {}).get('change_24h', 0)
                
                current_value = data['amount'] * current_price
                initial_value = data['amount'] * avg_price
                profit_loss = current_value - initial_value
                profit_loss_pct = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
                
                st.metric(
                    label=f"قيمة {coin}",
                    value=f"${current_value:.2f}",
                    delta=f"{profit_loss_pct:.1f}%"
                )
                
                col1, col2 = st.columns(2)
                col1.metric("السعر الحالي", f"${current_price:.2f}", f"{change_24h:.1f}%")
                col2.metric("متوسط الشراء", f"${avg_price:.2f}")
                
                st.progress(min(100, max(0, 50 + profit_loss_pct)))
                
                if st.button(f"إزالة {coin}", key=f"remove_{coin}"):
                    del portfolio[coin]
                    st.session_state['portfolio'] = portfolio
                    st.success(f"تمت إزالة {coin} من المحفظة!")
                    st.experimental_rerun()

def show_market_overview():
    """عرض نظرة عامة على السوق"""
    st.header("📊 نظرة عامة على السوق")
    
    # الحصول على بيانات السوق العامة
    market_sentiment_label, market_sentiment_value = get_market_sentiment()
    
    # عرض مؤشر الخوف والطمع
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("مؤشر الخوف والطمع")
        
        # رسم مقياس مؤشر الخوف والطمع
        color = "green" if market_sentiment_value > 50 else "red"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = market_sentiment_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': market_sentiment_label},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': "darkred"},
                    {'range': [20, 40], 'color': "red"},
                    {'range': [40, 60], 'color': "gray"},
                    {'range': [60, 80], 'color': "yellowgreen"},
                    {'range': [80, 100], 'color': "green"}
                ]
            }
        ))
        
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    # الحصول على أسعار العملات الرئيسية
    top_coins = ["BTC", "ETH", "SOL", "ADA", "XRP"]
    prices = get_current_prices(top_coins)
    
    with col2:
        st.subheader("أبرز العملات")
        for coin in top_coins[:3]:
            price_data = prices.get(coin, {})
            if price_data:
                price = price_data.get('price', 0)
                change = price_data.get('change_24h', 0)
                
                st.metric(
                    label=coin,
                    value=f"${price:.2f}",
                    delta=f"{change:.1f}%"
                )
    
    with col3:
        st.subheader("ㅤ")  # عنوان فارغ للمحاذاة
        for coin in top_coins[3:]:
            price_data = prices.get(coin, {})
            if price_data:
                price = price_data.get('price', 0)
                change = price_data.get('change_24h', 0)
                
                st.metric(
                    label=coin,
                    value=f"${price:.2f}",
                    delta=f"{change:.1f}%"
                )

def show_analysis_results(symbol, time_period):
    """عرض نتائج التحليل للعملة المختارة"""
    st.header(f"📈 تحليل {symbol}")
    
    # الحصول على بيانات السعر
    prices_df = get_price_data(
        COIN_IDS.get(symbol, symbol.lower()),
        days=time_period
    )
    
    if prices_df.empty:
        st.error(f"لم نتمكن من الحصول على بيانات لـ {symbol}. حاول مرة أخرى لاحقًا.")
        return
    
    # تحليل العملة
    analysis = analyze_crypto(symbol, prices_df)
    
    if not analysis:
        st.error(f"حدث خطأ أثناء تحليل {symbol}. حاول مرة أخرى لاحقًا.")
        return
    
    # عرض ملخص التحليل
    recommendation = analysis['recommendation']
    guide = DECISION_GUIDES[recommendation]
    
    st.subheader("ملخص التحليل")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"""
        <div style="padding: 10px; text-align: center; background-color: {guide['color']}; color: white; border-radius: 5px;">
            <h1 style="font-size: 48px; margin: 0;">{guide['icon']}</h1>
            <h3 style="margin: 5px 0;">{recommendation}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
            <p><b>السعر الحالي:</b> ${analysis['price']:.2f}</p>
            <p><b>هدف قصير المدى:</b> ${analysis['short_term_target']:.2f} ({((analysis['short_term_target']/analysis['price'])-1)*100:.1f}%)</p>
            <p><b>هدف طويل المدى:</b> ${analysis['long_term_target']:.2f} ({((analysis['long_term_target']/analysis['price'])-1)*100:.1f}%)</p>
            <p><b>وقف الخسارة:</b> ${analysis['stop_loss']:.2f} ({((analysis['price']-analysis['stop_loss'])/analysis['price'])*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="padding: 10px; background-color: {guide['color']}20; border-left: 5px solid {guide['color']}; margin: 10px 0; border-radius: 5px;">
        <p><b>{guide['text']}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض شرح التحليل
    with st.expander("شرح التحليل"):
        for explanation in analysis['explanation']:
            st.write(f"• {explanation}")
    
    # عرض الرسوم البيانية
    st.subheader("تحليل السعر")
    price_chart = create_price_chart(symbol, prices_df, analysis)
    st.plotly_chart(price_chart, use_container_width=True)
    
    rsi_chart, macd_chart = create_indicator_charts(symbol, prices_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(rsi_chart, use_container_width=True)
        
        # شرح مؤشر RSI
        rsi_value = analysis['rsi']
        rsi_text = "محايد"
        rsi_color = "gray"
        
        if rsi_value < 30:
            rsi_text = "تشبع بيعي (فرصة شراء محتملة)"
            rsi_color = "green"
        elif rsi_value < 40:
            rsi_text = "ضعف (ميل إيجابي)"
            rsi_color = "lightgreen"
        elif rsi_value > 70:
            rsi_text = "تشبع شرائي (فرصة بيع محتملة)"
            rsi_color = "red"
        elif rsi_value > 60:
            rsi_text = "قوة (ميل سلبي)"
            rsi_color = "orange"
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: -20px;">
            <h3 style="color: {rsi_color};">{rsi_value:.1f}</h3>
            <p>{rsi_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.plotly_chart(macd_chart, use_container_width=True)
        
        # شرح مؤشر MACD
        macd_signal = analysis['macd']['signal']
        macd_histogram = analysis['macd']['histogram']
        
        macd_text = "حركة جانبية"
        macd_color = "gray"
        
        if macd_signal == "buy":
            macd_text = "إشارة شراء (زخم إيجابي)"
            macd_color = "green"
        elif macd_signal == "sell":
            macd_text = "إشارة بيع (زخم سلبي)"
            macd_color = "red"
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: -20px;">
            <h3 style="color: {macd_color};">{macd_signal.upper()}</h3>
            <p>{macd_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # عرض خطة التداول
    st.subheader("خطة التداول")
    
    portfolio = st.session_state.get('portfolio', {})
    portfolio_value = sum([
        data['amount'] * get_current_prices([coin])[coin]['price']
        for coin, data in portfolio.items()
        if coin in get_current_prices([coin])
    ]) if portfolio else 10000
    
    # حساب حجم المركز المناسب
    risk_percentage = 2  # نسبة المخاطرة (2% من المحفظة)
    risk_amount = portfolio_value * (risk_percentage / 100)
    
    if recommendation in ['STRONG BUY', 'BUY']:
        risk_per_unit = analysis['price'] - analysis['stop_loss']
        position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        position_value = position_size * analysis['price']
        
        st.markdown(f"""
        <div style="padding: 15px; background-color: #e8f5e9; border-radius: 5px; margin: 10px 0;">
            <h4>خطة الشراء</h4>
            <p><b>الكمية المقترحة:</b> {position_size:.4f} {symbol} (${position_value:.2f})</p>
            <p><b>سعر الدخول:</b> ${analysis['price']:.2f}</p>
            <p><b>وقف الخسارة:</b> ${analysis['stop_loss']:.2f} (خسارة محتملة: ${risk_amount:.2f})</p>
            <p><b>هدف الربح 1:</b> ${analysis['short_term_target']:.2f} (ربح محتمل: ${position_size * (analysis['short_term_target'] - analysis['price']):.2f})</p>
            <p><b>هدف الربح 2:</b> ${analysis['long_term_target']:.2f} (ربح محتمل: ${position_size * (analysis['long_term_target'] - analysis['price']):.2f})</p>
            <p><b>نسبة المخاطرة/المكافأة:</b> {((analysis['short_term_target'] - analysis['price']) / (analysis['price'] - analysis['stop_loss'])):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif recommendation in ['STRONG SELL', 'SELL']:
        current_holding = portfolio.get(symbol, {}).get('amount', 0)
        
        if current_holding > 0:
            st.markdown(f"""
            <div style="padding: 15px; background-color: #ffebee; border-radius: 5px; margin: 10px 0;">
                <h4>خطة البيع</h4>
                <p><b>الكمية المتاحة للبيع:</b> {current_holding} {symbol}</p>
                <p><b>سعر البيع الحالي:</b> ${analysis['price']:.2f}</p>
                <p><b>إعادة الشراء عند:</b> ${analysis['short_term_target']:.2f} ({((analysis['short_term_target']/analysis['price'])-1)*100:.1f}%)</p>
                <p><b>وقف الربح:</b> ${analysis['stop_loss']:.2f} (في حال ارتفع السعر بدلاً من الهبوط)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 15px; background-color: #ffebee; border-radius: 5px; margin: 10px 0;">
                <h4>خطة البيع القصير (متقدم)</h4>
                <p><b>ملاحظة:</b> البيع القصير ينطوي على مخاطر عالية وهو مناسب للمتداولين ذوي الخبرة فقط.</p>
                <p><b>سعر البيع:</b> ${analysis['price']:.2f}</p>
                <p><b>هدف الربح:</b> ${analysis['short_term_target']:.2f} ({((analysis['short_term_target']/analysis['price'])-1)*100:.1f}%)</p>
                <p><b>وقف الخسارة:</b> ${analysis['stop_loss']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("التوصية الحالية هي الانتظار. من الأفضل البحث عن فرص أفضل أو الانتظار حتى تظهر إشارات أوضح.")
    
    with st.expander("نصائح لإدارة المخاطر"):
        st.markdown("""
        1. **لا تخاطر بأكثر من 1-2% من إجمالي محفظتك في صفقة واحدة**
        2. **استخدم دائمًا أوامر وقف الخسارة لحماية رأس المال**
        3. **فكر في تقسيم المركز إلى أجزاء للدخول تدريجيًا**
        4. **اتبع النظام بانضباط وتجنب القرارات العاطفية**
        5. **تحقق من الارتباط مع Bitcoin قبل اتخاذ القرار**
        """)

def create_watchlist_interface():
    """إنشاء واجهة قائمة المراقبة"""
    st.sidebar.header("👀 قائمة المراقبة")
    
    # استرجاع قائمة المراقبة الحالية
    watchlist = st.session_state.get('watchlist', [])
    
    # إضافة عملة جديدة
    with st.sidebar.expander("➕ إضافة عملة"):
        coin_options = [c for c in COIN_IDS.keys() if c not in watchlist]
        if coin_options:
            new_coin = st.selectbox("اختر العملة", coin_options)
            if st.button("إضافة إلى القائمة"):
                watchlist.append(new_coin)
                st.session_state['watchlist'] = watchlist
                st.success(f"تمت إضافة {new_coin} إلى قائمة المراقبة!")
                st.experimental_rerun()
        else:
            st.info("جميع العملات المدعومة موجودة بالفعل في قائمة المراقبة.")
    
    # عرض قائمة المراقبة
    if not watchlist:
        st.sidebar.info("قائمة المراقبة فارغة. أضف عملات لتتبعها.")
    else:
        # الحصول على الأسعار الحالية
        current_prices = get_current_prices(watchlist)
        
        for coin in watchlist:
            price_data = current_prices.get(coin, {})
            if price_data:
                price = price_data.get('price', 0)
                change = price_data.get('change_24h', 0)
                
                col1, col2 = st.sidebar.columns([3, 1])
                
                with col1:
                    st.metric(
                        label=coin,
                        value=f"${price:.2f}",
                        delta=f"{change:.1f}%"
                    )
                
                with col2:
                    if st.button("❌", key=f"remove_watch_{coin}"):
                        watchlist.remove(coin)
                        st.session_state['watchlist'] = watchlist
                        st.success(f"تمت إزالة {coin} من قائمة المراقبة!")
                        st.experimental_rerun()

# تهيئة التطبيق
def main():
    """الوظيفة الرئيسية للتطبيق"""
    # تهيئة متغيرات الجلسة
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = {}
    
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = ["BTC", "ETH", "SOL"]
    
    # الشعار والعنوان
    st.sidebar.title("📊 محلل العملات الرقمية")
    st.sidebar.markdown("---")
    
    # واجهة المحفظة
    create_portfolio_input()
    
    # واجهة قائمة المراقبة
    create_watchlist_interface()
    
    # الصفحة الرئيسية
    st.title("📈 محلل العملات الرقمية")
    
    # عرض نظرة عامة على السوق
    show_market_overview()
    
    # اختيار العملة وفترة التحليل
    col1, col2 = st.columns(2)
    
    with col1:
        selected_symbol = st.selectbox(
            "اختر العملة للتحليل",
            list(COIN_IDS.keys()),
            index=list(COIN_IDS.keys()).index("ETH")
        )
    
    with col2:
        time_periods = {
            "أسبوع": 7,
            "شهر": 30,
            "3 أشهر": 90,
            "6 أشهر": 180,
            "سنة": 365
        }
        
        selected_period = st.selectbox(
            "فترة التحليل",
            list(time_periods.keys()),
            index=2  # 3 أشهر افتراضيًا
        )
        days = time_periods[selected_period]
    
    # عرض تحليل العملة المختارة
    show_analysis_results(selected_symbol, days)
    
    # معلومات إضافية ونصائح
    with st.expander("📝 نصائح للمتداولين المبتدئين"):
        st.markdown("""
        ### نصائح مهمة للمبتدئين:
        
        1. **تعلم أساسيات التحليل الفني:** تعرف على المؤشرات الأساسية مثل RSI وMACD والمتوسطات المتحركة.
        
        2. **ابدأ صغيرًا:** لا تستثمر أكثر مما يمكنك تحمل خسارته، خاصة في البداية.
        
        3. **التنويع:** لا تضع كل استثماراتك في عملة واحدة.
        
        4. **الصبر:** التداول الناجح يتطلب الصبر وعدم التسرع في اتخاذ القرارات.
        
        5. **إدارة المخاطر:** حدد نسبة المخاطرة المقبولة لكل صفقة (عادة 1-2% من رأس المال).
        
        6. **الخطة:** حدد أهداف الربح ونقاط وقف الخسارة قبل الدخول في أي صفقة.
        
        7. **تجنب FOMO:** لا تتخذ قرارات بناءً على الخوف من فوات الفرصة.
        
        8. **تجاهل الضوضاء:** لا تتأثر بالآراء المتضاربة على وسائل التواصل الاجتماعي.
        
        9. **احتفظ بسجل:** دوّن صفقاتك وأسباب اتخاذها لتتعلم من أخطائك ونجاحاتك.
        
        10. **تعلم باستمرار:** سوق العملات الرقمية متغير، واصل التعلم والتكيف مع الظروف المتغيرة.
        """)
    
    # تنويه قانوني
    st.markdown("---")
    st.caption("""
    **تنويه:** المعلومات المقدمة في هذه الأداة هي لأغراض تعليمية وإعلامية فقط وليست نصيحة مالية.
    لا تتخذ قرارات استثمارية بناءً على هذه المعلومات وحدها.
    استشر مستشارًا ماليًا قبل اتخاذ أي قرارات استثمارية.
    """)
    
# تشغيل التطبيق
if __name__ == "__main__":
    main()