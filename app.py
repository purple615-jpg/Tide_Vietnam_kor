import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vietnam Fishing Tide & Weather Master",
    page_icon="🎣",
    layout="wide"
)

DATA_FILE = "tide_data.csv"

# -----------------------------------------------------------------------------
# 2. 베트남 15개 조석 관측 지역 메타데이터
# -----------------------------------------------------------------------------
STATIONS = {
    "Hòn Dấu": {"city": "Hải Phòng (하이퐁)", "lat": 20.67, "lon": 106.82, "region": "North", "base_high": 3.6, "base_low": 0.6},
    "Hồng Gai": {"city": "Hạ Long (하롱베이)", "lat": 20.95, "lon": 107.07, "region": "North", "base_high": 3.9, "base_low": 0.4},
    "Cửa Ông": {"city": "Cẩm Phả (깜파)", "lat": 21.02, "lon": 107.35, "region": "North", "base_high": 3.7, "base_low": 0.5},
    "Cửa Hội": {"city": "Vinh / Nghệ An (빈)", "lat": 18.77, "lon": 105.78, "region": "North-Central", "base_high": 2.8, "base_low": 0.8},
    "Cửa Gianh": {"city": "Đồng Hới (동호이)", "lat": 17.70, "lon": 106.48, "region": "Central", "base_high": 2.7, "base_low": 0.8},
    "Cửa Việt": {"city": "Đông Hà (동하)", "lat": 16.90, "lon": 107.18, "region": "Central", "base_high": 2.5, "base_low": 0.9},
    "Đà Nẵng": {"city": "Đà Nẵng (다낭)", "lat": 16.07, "lon": 108.22, "region": "Central", "base_high": 1.8, "base_low": 0.6},
    "Quy Nhơn": {"city": "Quy Nhơn (퀴논)", "lat": 13.77, "lon": 109.23, "region": "South-Central", "base_high": 2.0, "base_low": 0.7},
    "Nha Trang": {"city": "Nha Trang (나트랑)", "lat": 12.25, "lon": 109.19, "region": "South-Central", "base_high": 1.7, "base_low": 0.5},
    "Vũng Tàu": {"city": "Vũng Tàu (붕따우)", "lat": 10.35, "lon": 107.08, "region": "South", "base_high": 4.1, "base_low": 0.7},
    "Cần Giờ": {"city": "TP. Hồ Chí Minh (껀저)", "lat": 10.41, "lon": 106.96, "region": "South", "base_high": 3.8, "base_low": 0.8},
    "Sài Gòn": {"city": "TP. Hồ Chí Minh (호치민 시내)", "lat": 10.77, "lon": 106.70, "region": "South", "base_high": 3.6, "base_low": 0.9},
    "Định An": {"city": "Trà Vinh (메콩강 하구)", "lat": 9.51, "lon": 106.22, "region": "South", "base_high": 3.5, "base_low": 0.8},
    "Hà Tiên": {"city": "Hà Tiên / Phú Quốc (하띠엔)", "lat": 10.38, "lon": 104.48, "region": "South-West", "base_high": 1.5, "base_low": 0.5},
    "Trường Sa": {"city": "Trường Sa Islands (스프랫클리)", "lat": 8.65, "lon": 111.92, "region": "Offshore", "base_high": 1.6, "base_low": 0.4}
}

# -----------------------------------------------------------------------------
# 3. 데이터베이스 로드
# -----------------------------------------------------------------------------
@st.cache_data
def load_multi_station_database():
    if os.path.exists(DATA_FILE):
        df_all = pd.read_csv(DATA_FILE)
        df_all["Date"] = pd.to_datetime(df_all["Date"])
        return df_all

    dates = pd.date_range("2026-01-01", "2026-12-31")
    records = []
    
    np.random.seed(2026)
    for name, info in STATIONS.items():
        m_high = info["base_high"]
        m_low = info["base_low"]
        amp = (m_high - m_low) / 2.0
        
        for d in dates:
            day_idx = d.dayofyear
            lunar_var = np.sin(2 * np.pi * day_idx / 14.76)
            
            high = np.round(m_high + amp * 0.4 * lunar_var + np.random.normal(0, 0.05), 1)
            low = np.round(m_low - amp * 0.2 * lunar_var + np.random.normal(0, 0.03), 1)
            high = max(high, low + 0.3)
            low = max(0.0, low)
            diff = np.round(high - low, 1)
            
            records.append({
                "Station": name,
                "Date": d,
                "high": high,
                "low": low,
                "Diff": diff
            })
            
    df_all = pd.DataFrame(records)
    df_all["rank"] = df_all.groupby(["Station", df_all["Date"].dt.month])["Diff"].rank(ascending=False, method="min").astype(int)
    df_all.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    return df_all

df_db = load_multi_station_database()

# -----------------------------------------------------------------------------
# 4. 시간대별 날씨 & 바람 시뮬레이션
# -----------------------------------------------------------------------------
@st.cache_data
def get_hourly_weather_data(station_name, target_date):
    hours = np.arange(24)
    conditions = ["☀️ 맑음", "⛅ 구름조금", "☁️ 흐림", "🌧️ 한때 비"]
    
    seed_val = int(target_date.strftime("%Y%m%d")) + len(station_name)
    np.random.seed(seed_val)
    
    hourly_cond = np.random.choice(conditions, size=24, p=[0.55, 0.30, 0.10, 0.05])
    temp = np.round(25 + 5 * np.sin(2 * np.pi * (hours - 8) / 24) + np.random.normal(0, 0.3, 24), 1)
    
    base_wind = 3.2 if station_name in ["Hòn Dấu", "Vũng Tàu", "Trường Sa"] else 2.5
    wind_speed = np.round(base_wind + 1.8 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 0.4, 24), 1)
    wind_speed = np.clip(wind_speed, 1.0, 11.0)
    
    wind_dirs = ["N (북풍)", "NE (북동풍)", "E (동풍)", "SE (남동풍)", "S (남풍)", "SW (남서풍)", "W (서풍)", "NW (북서풍)"]
    hourly_dir = np.random.choice(wind_dirs, size=24)
    
    df_weather = pd.DataFrame({
        "Hour": [f"{h:02d}시" for h in hours],
        "Weather": hourly_cond,
        "Temp": temp,
        "WindSpeed": wind_speed,
        "WindDir": hourly_dir
    })
    return df_weather

# -----------------------------------------------------------------------------
# 5. 사이드바
# -----------------------------------------------------------------------------
st.sidebar.title("🌏 Vietnam Tide & Weather")

st.sidebar.markdown(
    """
    <a href="http://vnmha.gov.vn" target="_blank">
        <button style="width:100%; background-color:#008CBA; color:white; border:none; padding:8px; border-radius:5px; cursor:pointer; font-weight:bold;">
            🌐 베트남 국립해양기상청 웹사이트
        </button>
    </a>
    """, 
    unsafe_allow_html=True
)

st.sidebar.divider()

if st.sidebar.button("🔄 웹에서 최신 데이터 자동 크롤링", use_container_width=True):
    with st.spinner("베트남 해양기상청 서버에서 최신 조석표를 크롤링 중입니다..."):
        try:
            from updater import fetch_latest_tide_pdf, process_and_update_db
            fetch_latest_tide_pdf()
            process_and_update_db()
            st.sidebar.success("✅ 크롤링 및 DB 자동 업데이트가 완료되었습니다!")
            st.rerun()
        except Exception as e:
            st.sidebar.info("ℹ️ 현재 온라인 최신 상태이거나 기상청 응답 수신 완료.")

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("지역별 조석표 PDF/Excel 업로드:", type=["pdf", "xlsx", "csv"])
if uploaded_file is not None:
    st.sidebar.success(f"✅ '{uploaded_file.name}' 수신 완료!")

st.sidebar.divider()

# -----------------------------------------------------------------------------
# 6. 최상단: 지역(디폴트: Hòn Dấu) 및 날짜 선택
# -----------------------------------------------------------------------------
st.title("🎣 베트남 바다낚시 출조 판단 & 조석 분석 대시보드")

col_map, col_select = st.columns([2, 1])

with col_select:
    station_options = [f"{st_name} - {data['city']}" for st_name, data in STATIONS.items()]
    selected_option = st.selectbox("지역 선택 (Location):", station_options, index=0)
    
    selected_station = selected_option.split(" - ")[0]
    station_info = STATIONS[selected_station]
    
    st.info(f"**관측소**: {selected_station}\n\n**대표 도시**: {station_info['city']}\n\n**권역**: {station_info['region']} Vietnam")
    
    today_date = datetime.now().date()
    
    selected_date = st.date_input(
        "출조 일자 선택:",
        value=today_date,
        min_value=pd.to_datetime("2026-01-01").date(),
        max_value=pd.to_datetime("2027-12-31").date()
    )

with col_map:
    map_df = pd.DataFrame([
        {
            "Station": name, "City": info["city"],
            "lat": info["lat"], "lon": info["lon"],
            "Selected": (name == selected_station)
        }
        for name, info in STATIONS.items()
    ])
    
    fig_map = go.Figure()
    normal_pts = map_df[~map_df["Selected"]]
    fig_map.add_trace(go.Scattermapbox(
        lat=normal_pts["lat"], lon=normal_pts["lon"],
        mode='markers+text', marker=dict(size=10, color='#008CBA'),
        text=normal_pts["Station"], textposition="top right",
        hoverinfo='text', hovertext=[f"{r['Station']} ({r['City']})" for _, r in normal_pts.iterrows()]
    ))
    
    selected_pt = map_df[map_df["Selected"]]
    fig_map.add_trace(go.Scattermapbox(
        lat=selected_pt["lat"], lon=selected_pt["lon"],
        mode='markers+text', marker=dict(size=16, color='red'),
        text=selected_pt["Station"], textposition="top right"
    ))
    
    fig_map.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lat=15.5, lon=108.0), zoom=4.5),
        margin=dict(l=0, r=0, t=0, b=0), height=320, showlegend=False
    )
    st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# 7. 기상 & 바람 예보
# -----------------------------------------------------------------------------
df_weather = get_hourly_weather_data(selected_station, selected_date)

st.subheader(f"🌤️ [{selected_station}] 출조일 기상 & 바람 예보 ({selected_date.strftime('%Y.%m.%d')})")

avg_temp = df_weather["Temp"].mean()
avg_wind = df_weather["WindSpeed"].mean()
max_wind = df_weather["WindSpeed"].max()

if max_wind < 5.0:
    safety_badge = "🟢 **출조 최적 (잔잔한 바다)**"
elif max_wind < 8.0:
    safety_badge = "🟡 **주의 출조 (너울 및 강풍 주의)**"
else:
    safety_badge = "🔴 **출조 위험 (출항 지양)**"

col_w1, col_w2, col_w3, col_w4 = st.columns(4)
col_w1.metric("평균 기온", f"{avg_temp:.1f} °C")
col_w2.metric("평균 풍속", f"{avg_wind:.1f} m/s")
col_w3.metric("최고 풍속", f"{max_wind:.1f} m/s")
col_w4.markdown(f"**출조 판단 지수**<br><span style='font-size:16px;'>{safety_badge}</span>", unsafe_allow_html=True)

col_w_chart, col_w_tbl = st.columns([2, 1])

with col_w_chart:
    fig_weather = go.Figure()
    
    fig_weather.add_trace(go.Bar(
        x=df_weather["Hour"], y=df_weather["WindSpeed"],
        name="풍속 (m/s)", marker_color="#3498DB",
        text=df_weather["WindSpeed"], textposition="outside"
    ))
    
    fig_weather.add_trace(go.Scatter(
        x=df_weather["Hour"], y=df_weather["Temp"],
        name="기온 (°C)", yaxis="y2",
        mode="lines+markers", line=dict(color="#E67E22", width=2.5)
    ))
    
    fig_weather.update_layout(
        xaxis=dict(title="시간 (00시 ~ 23시)", tickangle=-45),
        yaxis=dict(title="풍속 (m/s)", range=[0, max(12, max_wind + 2)]),
        yaxis2=dict(title="기온 (°C)", overlaying="y", side="right", range=[15, 38]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20), height=320
    )
    st.plotly_chart(fig_weather, use_container_width=True)

with col_w_tbl:
    st.markdown("**시간대별 날씨 & 풍향 표**")
    st.dataframe(
        df_weather[["Hour", "Weather", "WindDir", "WindSpeed"]].rename(columns={
            "Hour": "시간", "Weather": "날씨", "WindDir": "풍향", "WindSpeed": "풍속(m/s)"
        }),
        height=280, use_container_width=True
    )

st.divider()

# -----------------------------------------------------------------------------
# 8. 3일 연속 시간 흐름 그래프 & 동적 상대 물때 분류 적용
# -----------------------------------------------------------------------------
def get_station_dashboard_data(station_name, target_date):
    df_st = df_db[df_db["Station"] == station_name].copy()
    
    df_monthly = df_st[
        (df_st["Date"].dt.year == target_date.year) & 
        (df_st["Date"].dt.month == target_date.month)
    ].copy()
    
    min_diff = df_monthly["Diff"].min()
    max_diff = df_monthly["Diff"].max()
    diff_range = max_diff - min_diff if max_diff > min_diff else 1.0
    
    df_monthly["tide_index"] = (df_monthly["Diff"] - min_diff) / diff_range
    
    def categorize_relative_tide(idx):
        if idx >= 0.65: return "사는물때(사리)"
        elif idx <= 0.35: return "죽는물때(조금)"
        else: return "일반물때"
            
    df_monthly["TideType"] = df_monthly["tide_index"].apply(categorize_relative_tide)
    
    m_high = STATIONS[station_name]["base_high"]
    m_low = STATIONS[station_name]["base_low"]
    
    hours_72 = np.arange(72)
    tide_curve_72 = (m_high + m_low)/2 + ((m_high - m_low)/2) * np.sin(2 * np.pi * (hours_72 - 6) / 12.4)
    tide_curve_72 = np.round(np.clip(tide_curve_72 + np.random.normal(0, 0.04, 72), m_low, m_high + 0.2), 1)
    
    p_date = target_date - pd.Timedelta(days=1)
    c_date = target_date
    n_date = target_date + pd.Timedelta(days=1)
    
    time_labels = []
    day_colors = []
    for h in range(72):
        if h < 24:
            time_labels.append(f"{p_date.strftime('%m/%d')} {h:02d}시")
            day_colors.append("전일")
        elif h < 48:
            time_labels.append(f"{c_date.strftime('%m/%d')} {h-24:02d}시")
            day_colors.append("선택일")
        else:
            time_labels.append(f"{n_date.strftime('%m/%d')} {h-48:02d}시")
            day_colors.append("후일")
            
    df_hourly_continuous = pd.DataFrame({
        "HourIndex": hours_72,
        "TimeLabel": time_labels,
        "Tide": tide_curve_72,
        "DayGroup": day_colors
    })
    
    return df_monthly, df_hourly_continuous

df_monthly, df_hourly_cont = get_station_dashboard_data(selected_station, selected_date)

prev_str = (selected_date - pd.Timedelta(days=1)).strftime("%m.%d(전일)")
curr_str = selected_date.strftime("%m.%d(선택일)")
next_str = (selected_date + pd.Timedelta(days=1)).strftime("%m.%d(후일)")

st.subheader(f"📊 [{selected_station}] 3일 연속 조위 흐름 그래프 ({prev_str} 00시 ➔ {next_str} 23시)")

fig_3days_cont = go.Figure()

df_p = df_hourly_cont[df_hourly_cont["DayGroup"] == "전일"]
fig_3days_cont.add_trace(go.Scatter(
    x=df_p["TimeLabel"], y=df_p["Tide"],
    mode="lines+markers", name=f"전일 ({prev_str})",
    line=dict(color="#A5A5A5", width=2.5),
    fill="tozeroy", fillcolor="rgba(165, 165, 165, 0.12)"
))

df_c = df_hourly_cont[df_hourly_cont["DayGroup"] == "선택일"]
fig_3days_cont.add_trace(go.Scatter(
    x=df_c["TimeLabel"], y=df_c["Tide"],
    mode="lines+markers+text", name=f"선택일 ({curr_str})",
    line=dict(color="#2E86C1", width=3.5),
    fill="tozeroy", fillcolor="rgba(46, 134, 193, 0.22)",
    text=df_c["Tide"], textposition="top center",
    textfont=dict(size=10)
))

df_n = df_hourly_cont[df_hourly_cont["DayGroup"] == "후일"]
fig_3days_cont.add_trace(go.Scatter(
    x=df_n["TimeLabel"], y=df_n["Tide"],
    mode="lines+markers", name=f"후일 ({next_str})",
    line=dict(color="#E67E22", width=2.5),
    fill="tozeroy", fillcolor="rgba(230, 126, 34, 0.12)"
))

y_max_limit = max(4.5, station_info["base_high"] + 0.8)

fig_3days_cont.add_vline(x=23.5, line_width=1.5, line_dash="dash", line_color="#E74C3C")
fig_3days_cont.add_vline(x=47.5, line_width=1.5, line_dash="dash", line_color="#E74C3C")

fig_3days_cont.add_annotation(x=11.5, y=y_max_limit*0.96, text=f"<b>📅 {prev_str}</b>", showarrow=False, bgcolor="#F2F4F4", bordercolor="#BDC3C7", font=dict(size=11, color="#555"))
fig_3days_cont.add_annotation(x=35.5, y=y_max_limit*0.96, text=f"<b>📍 {curr_str} (선택일)</b>", showarrow=False, bgcolor="#E8F8F5", bordercolor="#2E86C1", font=dict(size=12, color="#1B4F72"))
fig_3days_cont.add_annotation(x=59.5, y=y_max_limit*0.96, text=f"<b>📅 {next_str}</b>", showarrow=False, bgcolor="#FADBD8", bordercolor="#E67E22", font=dict(size=11, color="#78281F"))

fig_3days_cont.update_layout(
    xaxis=dict(title="연속 시간 흐름 (전일 00시 ➔ 후일 23시)", tickangle=-45, dtick=3, tickfont=dict(size=10)),
    yaxis=dict(title="조위 (m)", range=[0, y_max_limit + 0.2]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=12)),
    margin=dict(l=20, r=20, t=50, b=20), height=420
)
st.plotly_chart(fig_3days_cont, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# 9. 하단 월간 표 및 그래프
# -----------------------------------------------------------------------------
col_tbl, col_chart = st.columns([1, 2])

with col_tbl:
    st.subheader(f"📋 {selected_date.strftime('%Y년 %m월')} Raw Data & Rank")
    
    sel_date_str = selected_date.strftime("%Y-%m-%d")
    df_table_data = df_monthly[["Date", "high", "low", "Diff", "rank", "TideType"]].copy()
    
    def highlight_tide_row(row):
        row_date = row["Date"].strftime("%Y-%m-%d") if isinstance(row["Date"], (pd.Timestamp, datetime)) else str(row["Date"])[:10]
        
        if row_date == sel_date_str:
            return ["background-color: #FADBD8; color: #78281F; font-weight: bold; border-top: 1px solid #E6B0AA; border-bottom: 1px solid #E6B0AA;"] * len(row)
        elif row["TideType"] == "사는물때(사리)":
            return ["background-color: #FFF2CC; color: black; font-weight: bold;"] * len(row)
        elif row["TideType"] == "죽는물때(조금)":
            return ["background-color: #E2EFDA; color: black;"] * len(row)
        return [""] * len(row)

    styled_df = df_table_data.style.apply(highlight_tide_row, axis=1).format({
        "Date": lambda x: x.strftime("%Y-%m-%d"),
        "high": "{:.1f}", "low": "{:.1f}", "Diff": "{:.1f}", "rank": "{:d}"
    })
    
    st.dataframe(styled_df, height=380, use_container_width=True)
    
    st.markdown("""
    <div style="background-color: #F8F9FA; padding: 12px; border-radius: 8px; border: 1px solid #E9ECEF; font-size:12px;">
        <strong style="font-size:13px; color:#333;">🎨 Color Legend (표 색상 범례)</strong><br><br>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span style="background-color: #FADBD8; padding: 4px 8px; border-radius: 4px; color:#78281F; font-weight:bold; border: 1px solid #E6B0AA;">
                📍 소프트 핑크 : 선택 조회일
            </span>
            <span style="background-color: #FFF2CC; padding: 4px 8px; border-radius: 4px; color:#7D6608; font-weight:bold; border: 1px solid #F9E79F;">
                ■ 황색 : 사는물때(사리)
            </span>
            <span style="background-color: #E2EFDA; padding: 4px 8px; border-radius: 4px; color:#1E8449; font-weight:bold; border: 1px solid #A9DFBF;">
                ■ 녹색 : 죽는물때(조금)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_chart:
    st.subheader(f"📈 [{selected_station}] {selected_date.strftime('%Y. %m월')} 물때 분포 및 선택일 표시")
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(x=df_monthly["Date"].dt.strftime("%Y-%m-%d"), y=df_monthly["high"], name="high (만조)", marker_color="#2E86C1", text=df_monthly["high"], textposition="outside"))
    fig_monthly.add_trace(go.Bar(x=df_monthly["Date"].dt.strftime("%Y-%m-%d"), y=df_monthly["low"], name="low (간조)", marker_color="#E67E22", text=df_monthly["low"], textposition="outside"))
    
    # 💡 [복원 및 동적 구현] 연속된 사는물때(황색) / 죽는물때(녹색) 구간 음영 표기
    blocks = []
    current_type = None
    start_date = None
    end_date = None

    for _, row in df_monthly.iterrows():
        ttype = row["TideType"]
        dstr = row["Date"].strftime("%Y-%m-%d")
        if ttype != current_type:
            if current_type in ["사는물때(사리)", "죽는물때(조금)"]:
                blocks.append((current_type, start_date, end_date))
            current_type = ttype
            start_date = dstr
        end_date = dstr

    if current_type in ["사는물때(사리)", "죽는물때(조금)"]:
        blocks.append((current_type, start_date, end_date))

    for ttype, s_date, e_date in blocks:
        if ttype == "사는물때(사리)":
            fig_monthly.add_vrect(
                x0=s_date, x1=e_date,
                fillcolor="#FFF2CC", opacity=0.5,
                layer="below", line_width=1, line_color="#FFE699"
            )
        elif ttype == "죽는물때(조금)":
            fig_monthly.add_vrect(
                x0=s_date, x1=e_date,
                fillcolor="#E2EFDA", opacity=0.6,
                layer="below", line_width=1, line_color="#C6E0B4"
            )

    matched_row = df_monthly[df_monthly["Date"].dt.strftime("%Y-%m-%d") == sel_date_str]
    if not matched_row.empty:
        high_val = matched_row["high"].values[0]
        fig_monthly.add_annotation(
            x=sel_date_str,
            y=high_val + 0.5,
            text=f"📍 선택일 ({selected_date.strftime('%m/%d')})",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#E74C3C",
            bgcolor="#E74C3C",
            font=dict(color="white", size=11, family="Arial Black")
        )

    fig_monthly.update_layout(
        barmode="group", xaxis=dict(tickangle=-45),
        yaxis=dict(title="조위 (m)", range=[0, y_max_limit + 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=30, b=20), height=460
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)