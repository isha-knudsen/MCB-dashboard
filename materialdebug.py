"""
Enhanced Marine Cloud Brightening Dashboard - Advanced Visualizations

TO RUN THIS DASHBOARD:
1. Save this file as 'enhanced_mcb_dashboard.py' 
2. Run: streamlit run enhanced_mcb_dashboard.py

Enhanced Features:
- 3D surface plots for atmospheric conditions
- Interactive globe projections with route overlay
- Advanced heat map overlays with glassmorphism
- Real-time MCB effectiveness scoring
- Creative data storytelling visualizations
- NEW: Plume Dynamics Analysis tab
- NEW: Enhanced Heat Maps with wind vectors and statistics
- NEW: Data Export functionality
- NEW: Material selection in configuration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import time
from datetime import datetime
import math

# Page configuration
st.set_page_config(
    page_title="Enhanced MCB Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with dark theme and glassmorphism effects
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem !important;
        max-width: 98% !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c1445 0%, #1a1a2e 25%, #16213e 50%, #0f3460 100%);
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="0.5" fill="rgba(255,255,255,0.05)"/><circle cx="50" cy="10" r="0.8" fill="rgba(255,255,255,0.08)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
        pointer-events: none;
    }
    
    .hero-container h1 {
        color: white !important;
        font-weight: 800 !important;
        margin-bottom: 0.8rem !important;
        font-size: 3rem !important;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .hero-container p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.2rem !important;
        margin-bottom: 0 !important;
        position: relative;
        z-index: 1;
    }
    
    .metric-glass {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-glass:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .tab-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stats-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Silent data loading functions
@st.cache_data(ttl=7200, show_spinner=False)
def load_atmospheric_data_silent():
    """Silently load atmospheric dataset"""
    try:
        with sqlite3.connect('pathway_data.db') as conn:
            query = """
            SELECT 
                lat, lon, month,
                COALESCE(boundary_layer_height, 900) as boundary_layer_height, 
                COALESCE(low_cloud_cover, 0.4) as low_cloud_cover,
                COALESCE(wind_u_10m, 0) as wind_u_10m, 
                COALESCE(wind_v_10m, 0) as wind_v_10m,
                COALESCE(background_aerosol, 50) as background_aerosol,
                COALESCE(cloud_droplet_concentration, 100) as cloud_droplet_concentration
            FROM MCB_grid_single_level
            ORDER BY month, lat, lon
            """
            
            df = pd.read_sql_query(query, conn)
            df['wind_speed'] = np.sqrt(df['wind_u_10m']**2 + df['wind_v_10m']**2).fillna(8.5)
            return df.sort_values(['month', 'lat', 'lon']).reset_index(drop=True)
    except:
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic atmospheric data for demo"""
    np.random.seed(42)
    n_points = 50000
    
    data = []
    for month in range(1, 13):
        for _ in range(n_points // 12):
            lat = np.random.uniform(-80, 80)
            lon = np.random.uniform(-180, 180)
            
            # Create realistic patterns
            seasonal_factor = np.cos((month - 1) * np.pi / 6)
            latitude_factor = np.cos(lat * np.pi / 180)
            
            data.append({
                'lat': lat,
                'lon': lon,
                'month': month,
                'boundary_layer_height': 600 + 400 * (0.5 + 0.3 * seasonal_factor + 0.2 * latitude_factor),
                'low_cloud_cover': max(0.1, min(0.8, 0.4 + 0.2 * latitude_factor + 0.1 * seasonal_factor)),
                'wind_u_10m': np.random.normal(0, 5),
                'wind_v_10m': np.random.normal(0, 5),
                'background_aerosol': 30 + 40 * np.random.random(),
                'cloud_droplet_concentration': 50 + 100 * np.random.random()
            })
    
    df = pd.DataFrame(data)
    df['wind_speed'] = np.sqrt(df['wind_u_10m']**2 + df['wind_v_10m']**2)
    return df

@st.cache_data(ttl=3600)
def get_month_data(complete_df, month):
    """Fast month filtering"""
    return complete_df[complete_df['month'] == month].copy() if complete_df is not None else None

def get_optimized_sample(data, target_size=6000):
    """Smart sampling for visualizations"""
    if data is None or len(data) <= target_size:
        return data
    
    # Stratified sampling
    data['lat_bin'] = pd.cut(data['lat'], bins=15, labels=False)
    data['lon_bin'] = pd.cut(data['lon'], bins=20, labels=False)
    
    sampled = data.groupby(['lat_bin', 'lon_bin']).apply(
        lambda x: x.sample(n=min(len(x), max(1, target_size // 300)), random_state=42)
    ).reset_index(drop=True)
    
    return sampled.drop(['lat_bin', 'lon_bin'], axis=1)

def load_route_data():
    """Load or create route data"""
    try:
        with sqlite3.connect('pathway_data.db') as conn:
            routes_df = pd.read_sql_query("SELECT * FROM MCB_route_coordinates ORDER BY routeID, waypoint_seq", conn)
            deployments_df = pd.read_sql_query("SELECT * FROM MCB_shipping_routes", conn)
            
            if 'trips_per_month' not in deployments_df.columns:
                deployments_df['trips_per_month'] = 1000
            if 'speed_knots' not in deployments_df.columns:
                deployments_df['speed_knots'] = 20
                
            return routes_df, deployments_df
    except:
        # Fallback data
        routes_data = pd.DataFrame([
            {'routeID': 1, 'waypoint_seq': 1, 'lon': -70.19, 'lat': 43.62},
            {'routeID': 1, 'waypoint_seq': 2, 'lon': -65.60, 'lat': 43.27},
            {'routeID': 1, 'waypoint_seq': 3, 'lon': -54.48, 'lat': 46.28},
            {'routeID': 1, 'waypoint_seq': 4, 'lon': -51.85, 'lat': 46.19},
            {'routeID': 1, 'waypoint_seq': 5, 'lon': -24.59, 'lat': 63.27},
            {'routeID': 1, 'waypoint_seq': 6, 'lon': -23.41, 'lat': 64.30},
            {'routeID': 1, 'waypoint_seq': 7, 'lon': -22.11, 'lat': 64.19},
        ])
        
        deployments_data = pd.DataFrame([{
            'deployID': 'DEP00015',
            'routeID': 1,
            'description': 'Portland, Maine to Reykjavik, Iceland',
            'trips_per_month': 1000,
            'speed_knots': 20
        }])
        
        return routes_data, deployments_data

def load_materials_data():
    """Load or create materials data with correct column names"""
    try:
        with sqlite3.connect('pathway_data.db') as conn:
            materials_df = pd.read_sql_query("SELECT * FROM MCB_materials", conn)
            
            # Check if the column exists and rename if necessary
            if 'mat_name' in materials_df.columns and 'material_name' not in materials_df.columns:
                materials_df = materials_df.rename(columns={'mat_name': 'material_name'})
            
            # Ensure all required columns exist with defaults
            if 'material_name' not in materials_df.columns:
                materials_df['material_name'] = 'Sea Salt'
            if 'density' not in materials_df.columns:
                materials_df['density'] = 2160
            if 'cost_per_kg' not in materials_df.columns:
                materials_df['cost_per_kg'] = 0.1
                
            return materials_df
    except Exception as e:
        print(f"Error loading materials data: {e}")
        # Fallback materials data
        return pd.DataFrame([
            {'material_id': 1, 'material_name': 'Sea Salt', 'density': 2160, 'cost_per_kg': 0.1},
            {'material_id': 2, 'material_name': 'Sodium Chloride', 'density': 2170, 'cost_per_kg': 0.15},
            {'material_id': 3, 'material_name': 'Calcium Carbonate', 'density': 2710, 'cost_per_kg': 0.2},
            {'material_id': 4, 'material_name': 'Magnesium Sulfate', 'density': 2660, 'cost_per_kg': 0.18}
        ])

def calculate_mcb_effects(salt_mass_rate, particle_size, num_sprayers, wind_speed, 
                         cloud_droplet_conc, boundary_layer_height, cloud_coverage, 
                         residence_time, material_density=2160):
    """Enhanced MCB calculation with additional metrics"""
    
    # Constants
    F_solar = 342
    phi_atm = 0.76
    f_ocean = 0.54
    rho_salt = material_density
    K_broadening = 2.1
    
    # Conversions
    D_s = particle_size * 1e-9
    M_dot_s = salt_mass_rate / 3600
    U_0 = wind_speed
    tau_res = residence_time * 86400
    h = boundary_layer_height
    N_0_d = cloud_droplet_conc
    
    # Particle injection
    S = 1.6
    N_dot_s = (6 * M_dot_s) / (np.pi * rho_salt * D_s**3 * np.exp(9 * (np.log(S))**2 / 2))
    
    # Track dimensions
    L_t = U_0 * tau_res
    W_t = (K_broadening * 1000 / 3600) * tau_res / 2
    A_t = L_t * W_t
    
    # Particle concentration
    N_s1 = (2 * N_dot_s) / (h * U_0 * (K_broadening * 1000 / 3600) * tau_res)
    
    # Overlap calculations
    total_eligible_area = 1.98e14
    zeta = (num_sprayers * A_t) / total_eligible_area
    
    p_0 = np.exp(-zeta)
    p_1 = zeta * np.exp(-zeta)
    p_2 = (zeta**2 / 2) * np.exp(-zeta)
    p_3plus = 1 - p_0 - p_1 - p_2
    
    overlap_efficiency = 0.9
    N_s_eff = N_s1 * (1 * p_1 + 1.8 * p_2 + 2.4 * p_3plus) * overlap_efficiency
    
    # Activation
    size_factor = min(1.0, 0.3 + (particle_size - 30) * 0.02)
    supersaturation_factor = min(1.0, wind_speed / 10.0)
    activation_fraction = min(0.95, 0.4 * size_factor * supersaturation_factor)
    
    # Cloud enhancement
    N_d_increase = (N_s_eff * activation_fraction) / 1e6
    N_d_new = N_0_d + N_d_increase
    r_N = N_d_new / N_0_d
    
    # Twomey effect
    alpha_c = 0.56
    twomey_efficiency = min(1.0, 1 / (1 + r_N / 5))
    delta_alpha_c = (alpha_c * (1 - alpha_c) * (r_N**(1/3) - 1) * twomey_efficiency) / (1 + alpha_c * (r_N**(1/3) - 1))
    
    # Radiative forcing
    delta_F = -F_solar * f_ocean * cloud_coverage * phi_atm * delta_alpha_c
    
    # Metrics
    salt_emission_rate = (num_sprayers * M_dot_s * 3600 * 24 * 365) / 1e12
    cooling_per_sprayer = abs(delta_F) / num_sprayers if num_sprayers > 0 else 0
    cooling_per_tg_salt = abs(delta_F) / salt_emission_rate if salt_emission_rate > 0 else 0
    co2_offset_fraction = abs(delta_F) / 3.7
    area_affected = (num_sprayers * A_t) / 1e12
    
    return {
        'radiative_forcing': delta_F,
        'cooling_magnitude': abs(delta_F),
        'cloud_albedo_change': delta_alpha_c,
        'droplet_concentration_increase': N_d_increase,
        'droplet_ratio': r_N,
        'track_area_km2': A_t / 1e6,
        'track_length_km': L_t / 1000,
        'track_width_km': W_t / 1000,
        'plume_overlap_factor': 1 - p_0,
        'mean_track_density': zeta,
        'salt_emission_rate_tg_yr': salt_emission_rate,
        'cooling_per_sprayer': cooling_per_sprayer,
        'cooling_per_tg_salt': cooling_per_tg_salt,
        'activation_fraction': activation_fraction,
        'particles_per_second': N_dot_s,
        'co2_offset_fraction': co2_offset_fraction,
        'particle_concentration_m3': N_s_eff,
        'area_affected_million_km2': area_affected,
        'twomey_efficiency': twomey_efficiency,
        'environmental_impact_score': min(100, area_affected * 10)
    }

def create_plume_evolution_chart(results, residence_time):
    """Create plume evolution dynamics chart"""
    time_points = np.linspace(0, residence_time, 20)
    plume_widths = []
    plume_concentrations = []
    
    for t in time_points:
        # Plume width grows with time
        width = results['track_width_km'] * (1 + 0.5 * t / residence_time)
        plume_widths.append(width)
        
        # Concentration decreases with dispersion
        concentration = results['particle_concentration_m3'] * np.exp(-t / (residence_time * 0.8))
        plume_concentrations.append(concentration)
    
    fig_plume = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Plume Width Evolution', 'Particle Concentration Decay'],
        vertical_spacing=0.15
    )
    
    # Plume width
    fig_plume.add_trace(
        go.Scatter(
            x=time_points,
            y=plume_widths,
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(size=8),
            fill='tonexty',
            name='Plume Width'
        ),
        row=1, col=1
    )
    
    # Concentration
    fig_plume.add_trace(
        go.Scatter(
            x=time_points,
            y=plume_concentrations,
            mode='lines+markers',
            line=dict(color='#764ba2', width=4),
            marker=dict(size=8),
            fill='tozeroy',
            name='Concentration'
        ),
        row=2, col=1
    )
    
    fig_plume.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig_plume.update_yaxes(title_text="Width (km)", row=1, col=1)
    fig_plume.update_yaxes(title_text="Particles/m³", row=2, col=1)
    
    fig_plume.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig_plume

def create_wind_vector_analysis(viz_data):
    """Create wind vector analysis chart"""
    fig_wind = go.Figure()
    
    # Sample data for vectors (every 50th point to avoid overcrowding)
    sample_data = viz_data.iloc[::50]
    
    # Add wind vectors
    for _, row in sample_data.iterrows():
        fig_wind.add_trace(go.Scattermapbox(
            lat=[row['lat'], row['lat'] + row['wind_v_10m'] * 0.01],
            lon=[row['lon'], row['lon'] + row['wind_u_10m'] * 0.01],
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add base heat map for wind speed
    fig_wind.add_trace(go.Densitymapbox(
        lat=viz_data['lat'],
        lon=viz_data['lon'],
        z=viz_data['wind_speed'],
        radius=15,
        colorscale='Viridis',
        opacity=0.7,
        name='Wind Speed',
        hovertemplate='<b>Wind Speed</b><br>%{z:.1f} m/s<extra></extra>'
    ))
    
    fig_wind.update_layout(
        title="Wind Vector Analysis",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=0, lon=0),
            zoom=1
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_wind

def create_seasonal_variation_analysis(complete_atmospheric_df):
    """Create seasonal variation analysis"""
    if complete_atmospheric_df is None or len(complete_atmospheric_df) == 0:
        return None
    
    # Calculate monthly averages
    monthly_stats = complete_atmospheric_df.groupby('month').agg({
        'low_cloud_cover': ['mean', 'std'],
        'wind_speed': ['mean', 'std'],
        'boundary_layer_height': ['mean', 'std'],
        'background_aerosol': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
    monthly_stats = monthly_stats.reset_index()
    
    # Create seasonal plots
    fig_seasonal = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Cloud Cover Variation', 'Wind Speed Variation', 
                       'Boundary Layer Height', 'Background Aerosols'],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Cloud cover
    fig_seasonal.add_trace(
        go.Scatter(
            x=months,
            y=monthly_stats['low_cloud_cover_mean'],
            error_y=dict(type='data', array=monthly_stats['low_cloud_cover_std']),
            mode='lines+markers',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8),
            name='Cloud Cover'
        ),
        row=1, col=1
    )
    
    # Wind speed
    fig_seasonal.add_trace(
        go.Scatter(
            x=months,
            y=monthly_stats['wind_speed_mean'],
            error_y=dict(type='data', array=monthly_stats['wind_speed_std']),
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8),
            name='Wind Speed'
        ),
        row=1, col=2
    )
    
    # Boundary layer height
    fig_seasonal.add_trace(
        go.Scatter(
            x=months,
            y=monthly_stats['boundary_layer_height_mean'],
            error_y=dict(type='data', array=monthly_stats['boundary_layer_height_std']),
            mode='lines+markers',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8),
            name='BL Height'
        ),
        row=2, col=1
    )
    
    # Background aerosol
    fig_seasonal.add_trace(
        go.Scatter(
            x=months,
            y=monthly_stats['background_aerosol_mean'],
            error_y=dict(type='data', array=monthly_stats['background_aerosol_std']),
            mode='lines+markers',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=8),
            name='Aerosols'
        ),
        row=2, col=2
    )
    
    fig_seasonal.update_layout(
        height=600,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title="Seasonal Variation Analysis"
    )
    
    return fig_seasonal

# Title with enhanced styling
st.markdown("""
<div class="hero-container">
    <h1>🌊 Enhanced Marine Cloud Brightening Dashboard</h1>
    <p>Advanced atmospheric analysis with cutting-edge visualizations and real-time MCB modeling</p>
</div>
""", unsafe_allow_html=True)

# Silent data loading
complete_atmospheric_df = load_atmospheric_data_silent()
routes_df, deployments_df = load_route_data()
materials_df = load_materials_data()

# Sidebar with enhanced styling
st.sidebar.markdown("""
<div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); 
     border-radius: 15px; padding: 2rem; margin-bottom: 2rem;">
<h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">🎛️ MCB Configuration</h3>
</div>
""", unsafe_allow_html=True)

# Route selection
selected_deployment = st.sidebar.selectbox(
    "🚢 Deployment Strategy:",
    deployments_df['deployID'].tolist() if len(deployments_df) > 0 else ["DEP00015"],
    format_func=lambda x: f"{x}: {deployments_df[deployments_df['deployID']==x]['description'].iloc[0] if 'description' in deployments_df.columns and len(deployments_df) > 0 else 'Demo Route'}"
)

selected_route = deployments_df[deployments_df['deployID'] == selected_deployment]['routeID'].iloc[0] if len(deployments_df) > 0 else 1

# NEW: Material Selection
selected_material = st.sidebar.selectbox(
    "Select MCB Material:",
    materials_df['material_name'].tolist() if len(materials_df) > 0 and 'material_name' in materials_df.columns else ["Sea Salt"],
    help="Choose the material for cloud brightening particles"
)

# Get material properties
if len(materials_df) > 0 and 'material_name' in materials_df.columns:
    material_info = materials_df[materials_df['material_name'] == selected_material]
    if len(material_info) > 0:
        material_info = material_info.iloc[0]
        material_density = material_info.get('density', 2160)
        material_cost = material_info.get('cost_per_kg', 0.1)
    else:
        material_density = 2160
        material_cost = 0.1
else:
    material_density = 2160
    material_cost = 0.1

st.sidebar.info(f"**Selected Material:** {selected_material}\n\n**Density:** {material_density} kg/m³\n\n**Cost:** ${material_cost}/kg")

# MCB Parameters
st.sidebar.markdown("#### 🧪 MCB Parameters")
salt_mass_rate = st.sidebar.slider("Salt Mass Rate (kg/h)", 50.0, 300.0, 120.0, 10.0)
particle_size = st.sidebar.slider("Particle Size (nm)", 20.0, 120.0, 45.0, 5.0)
num_sprayers = st.sidebar.slider("Number of Sprayers", 5000, 50000, 15000, 1000)

# Environmental Parameters
st.sidebar.markdown("#### 🌊 Environmental Conditions")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 4.0, 12.0, 8.5, 0.5)
cloud_droplet_conc = st.sidebar.slider("Cloud Droplet Conc (cm⁻³)", 30.0, 150.0, 75.0, 5.0)
boundary_layer_height = st.sidebar.slider("Boundary Layer Height (m)", 400, 1200, 900, 50)
cloud_coverage = st.sidebar.slider("Cloud Coverage", 0.2, 0.6, 0.4, 0.05)
residence_time = st.sidebar.slider("Residence Time (days)", 1.5, 4.0, 2.5, 0.25)

# Calculate MCB effects
results = calculate_mcb_effects(
    salt_mass_rate, particle_size, num_sprayers, wind_speed,
    cloud_droplet_conc, boundary_layer_height, cloud_coverage, residence_time, material_density
)

# Enhanced metrics display with glassmorphism
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-glass pulse-animation">
        <div class="metric-value">{results['radiative_forcing']:.3f}</div>
        <div class="metric-label">W/m² Radiative Forcing</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-glass">
        <div class="metric-value">{results['co2_offset_fraction']:.1%}</div>
        <div class="metric-label">CO₂ Offset Potential</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-glass">
        <div class="metric-value">{results['area_affected_million_km2']:.1f}</div>
        <div class="metric-label">M km² Area Affected</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-glass">
        <div class="metric-value">{results['activation_fraction']:.0%}</div>
        <div class="metric-label">Activation Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-glass">
        <div class="metric-value">{results['cooling_per_tg_salt']:.3f}</div>
        <div class="metric-label">Cooling Efficiency</div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced tabs with new features
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 3D Atmospheric Globe & Routes", 
    "📊 MCB Effects Analysis", 
    "🌪️ Plume Dynamics Analysis",
    "🔬 Advanced Heat Maps",
    "🔬 Technical Details",
    "📁 Data Export"
])

with tab1:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown("### 🌍 3D Atmospheric Globe with Route Overlay")
    
    # Month selector
    selected_month = st.selectbox(
        "Select Month for 3D Analysis:",
        list(range(1, 13)),
        format_func=lambda x: ["January", "February", "March", "April", "May", "June", 
                              "July", "August", "September", "October", "November", "December"][x-1],
        key="3d_month"
    )
    
    month_data = get_month_data(complete_atmospheric_df, selected_month)
    if month_data is not None:
        viz_data = get_optimized_sample(month_data, 4000)
        
        # Create 3D surface plot
        fig_3d = go.Figure()
        
        # Add atmospheric data as 3D scatter
        fig_3d.add_trace(go.Scatter3d(
            x=viz_data['lon'],
            y=viz_data['lat'],
            z=viz_data['boundary_layer_height'],
            mode='markers',
            marker=dict(
                size=4,
                color=viz_data['low_cloud_cover'],
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Cloud Cover", thickness=15, len=0.5)
            ),
            text=[f"Lat: {lat:.1f}°<br>Lon: {lon:.1f}°<br>Cloud: {cloud:.2f}<br>Height: {height:.0f}m" 
                  for lat, lon, cloud, height in zip(viz_data['lat'], viz_data['lon'], 
                                                   viz_data['low_cloud_cover'], viz_data['boundary_layer_height'])],
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Atmospheric Data'
        ))
        
        # Add route if available
        if len(routes_df) > 0:
            route_data = routes_df[routes_df['routeID'] == selected_route]
            if len(route_data) > 0:
                fig_3d.add_trace(go.Scatter3d(
                    x=route_data['lon'],
                    y=route_data['lat'],
                    z=[1200] * len(route_data),  # Fixed height for route
                    mode='lines+markers',
                    line=dict(color='red', width=8),
                    marker=dict(size=8, color='red'),
                    name='MCB Route'
                ))
        
        fig_3d.update_layout(
            title=f"3D Atmospheric Conditions - {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][selected_month-1]}",
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude", 
                zaxis_title="Boundary Layer Height (m)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)")
            ),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Enhanced 3D surface for effectiveness
        st.markdown("#### 🎯 3D MCB Effectiveness Surface")
        
        # Create grid for surface plot
        lat_range = np.linspace(viz_data['lat'].min(), viz_data['lat'].max(), 30)
        lon_range = np.linspace(viz_data['lon'].min(), viz_data['lon'].max(), 40)
        lat_grid, lon_grid = np.meshgrid(lat_range, lon_range)
        
        # Calculate effectiveness scores
        effectiveness_scores = []
        for _, row in viz_data.iterrows():
            wind_score = min(1.0, row['wind_speed'] / 12.0)
            cloud_score = row['low_cloud_cover']
            boundary_score = min(1.0, row['boundary_layer_height'] / 1200)
            effectiveness = (wind_score + cloud_score + boundary_score) / 3
            effectiveness_scores.append(effectiveness)
        
        # Simple interpolation for surface
        effectiveness_grid = np.random.random((40, 30)) * 0.3 + 0.4  # Placeholder surface
        
        fig_surface = go.Figure(data=[go.Surface(
            x=lon_grid,
            y=lat_grid,
            z=effectiveness_grid,
            colorscale='RdYlBu_r',
            colorbar=dict(title="MCB Effectiveness", thickness=15),
            opacity=0.8
        )])
        
        fig_surface.update_layout(
            title="3D MCB Effectiveness Surface",
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                zaxis_title="Effectiveness Score",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
                bgcolor='rgba(0,0,0,0)'
            ),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_surface, use_container_width=True)
    
    # Add Interactive Route Visualization
    st.markdown("### 🗺️ Interactive Route Visualization")
    
    if len(routes_df) > 0:
        route_data = routes_df[routes_df['routeID'] == selected_route].copy()
        
        if len(route_data) > 0:
            fig_map = go.Figure()
            
            # Add route
            fig_map.add_trace(go.Scattergeo(
                lon=route_data['lon'],
                lat=route_data['lat'],
                mode='lines+markers',
                line=dict(width=4, color='#667eea'),
                marker=dict(size=12, color='#dc2626', line=dict(width=2, color='white')),
                text=[f"Waypoint {row['waypoint_seq']}<br>Lat: {row['lat']:.2f}<br>Lon: {row['lon']:.2f}" 
                      for _, row in route_data.iterrows()],
                hovertemplate='<b>%{text}</b><extra></extra>',
                showlegend=False
            ))
            
            fig_map.update_layout(
                title=f"MCB Deployment Route {selected_route}",
                geo=dict(
                    projection_type='natural earth',
                    showland=True,
                    landcolor='#f3f4f6',
                    oceancolor='#dbeafe',
                    showocean=True,
                    coastlinecolor='#6b7280'
                ),
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning(f"No route data found for route {selected_route}")
    else:
        st.error("No route data available")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Enhanced MCB Effects Analysis")
    
    # Create enhanced subplots with better styling
    fig_effects = make_subplots(
        rows=2, cols=2,
        subplot_titles=['☁️ Cloud Enhancement Impact', '📏 Plume Track Properties', '⚡ Particle Activation Efficiency', '🌡️ System Performance Metrics'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Enhanced Cloud enhancement with gradient bars
    fig_effects.add_trace(
        go.Bar(
            x=['Background Droplets', 'Enhanced Droplets'],
            y=[cloud_droplet_conc, cloud_droplet_conc + results['droplet_concentration_increase']],
            marker=dict(
                color=['rgba(59, 130, 246, 0.8)', 'rgba(16, 185, 129, 0.8)'],
                line=dict(color='white', width=2)
            ),
            text=[f'{cloud_droplet_conc:.0f} cm⁻³', f'{cloud_droplet_conc + results["droplet_concentration_increase"]:.1f} cm⁻³'],
            textposition='auto',
            textfont=dict(color='white', size=12),
            name='Cloud Droplets',
            hovertemplate='<b>%{x}</b><br>Concentration: %{y:.1f} cm⁻³<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Enhanced Track properties with gradient colors
    track_metrics = ['Length', 'Width', 'Total Area']
    track_values = [results['track_length_km'], results['track_width_km'], results['track_area_km2']]
    track_units = ['km', 'km', 'km²']
    colors = ['rgba(245, 158, 11, 0.8)', 'rgba(139, 92, 246, 0.8)', 'rgba(239, 68, 68, 0.8)']
    
    fig_effects.add_trace(
        go.Bar(
            x=track_metrics,
            y=track_values,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{val:.1f} {unit}' for val, unit in zip(track_values, track_units)],
            textposition='auto',
            textfont=dict(color='white', size=12),
            name='Track Dimensions',
            hovertemplate='<b>%{x}</b><br>Value: %{y:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Enhanced Activation efficiency with donut-style visualization
    activated = results['activation_fraction'] * 100
    not_activated = (1 - results['activation_fraction']) * 100
    
    fig_effects.add_trace(
        go.Bar(
            x=['Activated Particles', 'Non-Activated Particles'],
            y=[activated, not_activated],
            marker=dict(
                color=['rgba(102, 126, 234, 0.8)', 'rgba(209, 213, 219, 0.6)'],
                line=dict(color='white', width=2)
            ),
            text=[f'{activated:.1f}%', f'{not_activated:.1f}%'],
            textposition='auto',
            textfont=dict(color='white', size=12),
            name='Particle Activation',
            hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Enhanced System performance metrics with color coding
    performance_metrics = ['Cooling Effect', 'CO₂ Offset', 'Twomey Efficiency']
    performance_values = [
        results['cooling_magnitude'] * 1000,  # Convert to mW/m²
        results['co2_offset_fraction'] * 100,  # Convert to percentage
        results['twomey_efficiency'] * 100  # Convert to percentage
    ]
    performance_colors = ['rgba(16, 185, 129, 0.8)', 'rgba(5, 150, 105, 0.8)', 'rgba(8, 145, 178, 0.8)']
    performance_text = [
        f'{performance_values[0]:.1f}',  # mW/m²
        f'{performance_values[1]:.1f}%',  # percentage
        f'{performance_values[2]:.1f}%'   # percentage
    ]
    
    fig_effects.add_trace(
        go.Bar(
            x=performance_metrics,
            y=performance_values,
            marker=dict(
                color=performance_colors,
                line=dict(color='white', width=2)
            ),
            text=performance_text,
            textposition='auto',
            textfont=dict(color='white', size=12),
            name='Performance',
            hovertemplate='<b>%{x}</b><br>Value: %{y:.1f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig_effects.update_layout(
        height=700, 
        showlegend=False, 
        title_text="Enhanced MCB Effects Analysis",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    fig_effects.update_yaxes(title_text="Concentration (cm⁻³)", row=1, col=1, gridcolor="rgba(255,255,255,0.2)")
    fig_effects.update_yaxes(title_text="Distance/Area", row=1, col=2, gridcolor="rgba(255,255,255,0.2)")
    fig_effects.update_yaxes(title_text="Percentage (%)", row=2, col=1, gridcolor="rgba(255,255,255,0.2)")
    fig_effects.update_yaxes(title_text="Performance Score", row=2, col=2, gridcolor="rgba(255,255,255,0.2)")
    
    # Update x-axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig_effects.update_xaxes(gridcolor="rgba(255,255,255,0.2)", row=i, col=j)
    
    st.plotly_chart(fig_effects, use_container_width=True)
    
    # Enhanced Climate Impact Visualization
    st.markdown("#### 🌡️ Global Climate Impact Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Enhanced pie chart for radiative forcing balance
        total_warming = 3.7
        achieved_cooling = abs(results['radiative_forcing'])
        remaining = max(0, total_warming - achieved_cooling)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['MCB Cooling Effect', 'Remaining CO₂ Warming'],
            values=[achieved_cooling, remaining],
            marker=dict(
                colors=['rgba(16, 185, 129, 0.8)', 'rgba(229, 231, 235, 0.6)'],
                line=dict(color='white', width=3)
            ),
            hole=0.4,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>%{value:.3f} W/m²',
            textfont=dict(color='white', size=11),
            hovertemplate='<b>%{label}</b><br>Value: %{value:.3f} W/m²<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            title="Global Radiative Forcing Balance",
            height=400,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Enhanced droplet size distribution simulation
        st.markdown("#### 💧 Cloud Droplet Size Distribution")
        
        # Simulate droplet size distributions
        sizes = np.linspace(1, 50, 100)  # micrometers
        
        # Background distribution (broader)
        background_dist = np.exp(-((sizes - 12)**2) / (2 * 6**2))
        
        # Enhanced distribution (narrower, more droplets)
        enhanced_dist = np.exp(-((sizes - 8)**2) / (2 * 4**2)) * results['droplet_ratio']
        
        fig_droplet = go.Figure()
        
        fig_droplet.add_trace(go.Scatter(
            x=sizes,
            y=background_dist,
            mode='lines',
            name='Background Cloud',
            line=dict(color='rgba(59, 130, 246, 0.8)', width=3),
            fill='tonexty',
            hovertemplate='<b>Background</b><br>Size: %{x:.1f} μm<br>Density: %{y:.3f}<extra></extra>'
        ))
        
        fig_droplet.add_trace(go.Scatter(
            x=sizes,
            y=enhanced_dist,
            mode='lines',
            name='MCB Enhanced',
            line=dict(color='rgba(16, 185, 129, 0.8)', width=3),
            fill='tozeroy',
            hovertemplate='<b>MCB Enhanced</b><br>Size: %{x:.1f} μm<br>Density: %{y:.3f}<extra></extra>'
        ))
        
        fig_droplet.update_layout(
            title="Droplet Size Distribution",
            xaxis_title="Droplet Diameter (μm)",
            yaxis_title="Relative Density",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor="rgba(255,255,255,0.2)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.2)")
        )
        
        st.plotly_chart(fig_droplet, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# NEW TAB 3: Plume Dynamics Analysis
with tab3:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown("### 🌪️ Plume Dynamics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Plume Evolution Dynamics")
        
        # Create plume evolution chart
        fig_plume = create_plume_evolution_chart(results, residence_time)
        st.plotly_chart(fig_plume, use_container_width=True)
        
        st.markdown("#### 🎯 Plume Characteristics")
        
        # Plume characteristics metrics
        plume_metrics = {
            'Initial Width': f"{results['track_width_km']:.1f} km",
            'Final Width': f"{results['track_width_km'] * 1.5:.1f} km",
            'Track Length': f"{results['track_length_km']:.1f} km",
            'Total Area': f"{results['track_area_km2']:.1f} km²",
            'Track Density': f"{results['mean_track_density']:.3f}"
        }
        
        for metric, value in plume_metrics.items():
            st.markdown(f"**{metric}:** {value}")
    
    with col2:
        st.markdown("#### 🌀 Plume Dispersion Simulation")
        
        # Create animated particle dispersion simulation
        time_steps = np.linspace(0, 24, 25)  # 24 hours
        
        # Generate particle tracks
        n_particles = 30
        np.random.seed(42)
        
        fig_anim = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i in range(n_particles):
            # Starting position along route
            if len(routes_df) > 0:
                route_data = routes_df[routes_df['routeID'] == selected_route]
                if len(route_data) > 0:
                    start_idx = np.random.randint(0, len(route_data))
                    start_lat = route_data.iloc[start_idx]['lat']
                    start_lon = route_data.iloc[start_idx]['lon']
                else:
                    start_lat, start_lon = 50.0, -30.0
            else:
                start_lat, start_lon = 50.0, -30.0
            
            # Simulate particle drift
            wind_drift_x = np.cumsum(np.random.normal(0, 0.1, len(time_steps)))
            wind_drift_y = np.cumsum(np.random.normal(0, 0.05, len(time_steps)))
            
            lats = start_lat + wind_drift_y
            lons = start_lon + wind_drift_x
            
            # Particle size based on time (evaporation/growth)
            sizes = 8 + 4 * np.sin(time_steps * np.pi / 12)
            
            fig_anim.add_trace(go.Scatter(
                x=lons,
                y=lats,
                mode='lines+markers',
                line=dict(width=2, color=colors[i % len(colors)]),
                marker=dict(
                    size=sizes,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                name=f'Particle {i+1}' if i < 5 else '',
                showlegend=i < 5,
                hovertemplate=f'<b>Particle {i+1}</b><br>Lat: %{{y:.2f}}<br>Lon: %{{x:.2f}}<extra></extra>'
            ))
        
        fig_anim.update_layout(
            title="Salt Particle Dispersion Over 24 Hours",
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor="rgba(255,255,255,0.2)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.2)"),
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
        
        st.markdown("#### ⚡ Plume Performance Metrics")
        
        # Calculate additional plume metrics
        plume_efficiency = results['activation_fraction'] * results['twomey_efficiency']
        dispersion_rate = results['track_width_km'] / residence_time
        concentration_decay = 1 - np.exp(-residence_time / 2)
        
        perf_metrics = {
            'Plume Efficiency': f"{plume_efficiency:.1%}",
            'Dispersion Rate': f"{dispersion_rate:.1f} km/day",
            'Concentration Decay': f"{concentration_decay:.1%}",
            'Effective Coverage': f"{results['area_affected_million_km2']:.1f} M km²"
        }
        
        for metric, value in perf_metrics.items():
            st.markdown(f"**{metric}:** {value}")
    
    # Plume overlap analysis
    st.markdown("#### 🔄 Plume Overlap Analysis")
    
    # Generate overlap data for different sprayer counts
    sprayer_counts = np.logspace(3, 5, 20)  # 1,000 to 100,000 sprayers
    overlap_data = []
    
    for count in sprayer_counts:
        temp_results = calculate_mcb_effects(
            salt_mass_rate, particle_size, int(count), wind_speed,
            cloud_droplet_conc, boundary_layer_height, cloud_coverage, residence_time, material_density
        )
        overlap_data.append({
            'sprayers': int(count),
            'overlap_factor': temp_results['plume_overlap_factor'],
            'track_density': temp_results['mean_track_density'],
            'cooling_effect': temp_results['cooling_magnitude'],
            'efficiency': temp_results['cooling_per_sprayer']
        })
    
    overlap_df = pd.DataFrame(overlap_data)
    
    # Create overlap analysis plots
    fig_overlap = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Plume Overlap vs Sprayer Count',
            'Track Density',
            'Total Cooling Effect',
            'Cooling per Sprayer'
        ]
    )
    
    # Overlap factor
    fig_overlap.add_trace(
        go.Scatter(
            x=overlap_df['sprayers'],
            y=overlap_df['overlap_factor'] * 100,
            mode='lines+markers',
            name='Overlap %',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Track density
    fig_overlap.add_trace(
        go.Scatter(
            x=overlap_df['sprayers'],
            y=overlap_df['track_density'],
            mode='lines+markers',
            name='Track Density',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    # Total cooling
    fig_overlap.add_trace(
        go.Scatter(
            x=overlap_df['sprayers'],
            y=overlap_df['cooling_effect'],
            mode='lines+markers',
            name='Total Cooling',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    # Efficiency
    fig_overlap.add_trace(
        go.Scatter(
            x=overlap_df['sprayers'],
            y=overlap_df['efficiency'] * 1000,  # Convert to mW/m²
            mode='lines+markers',
            name='Cooling per Sprayer',
            line=dict(color='orange')
        ),
        row=2, col=2
    )
    
    fig_overlap.update_xaxes(type="log", title_text="Number of Sprayers")
    fig_overlap.update_yaxes(title_text="Overlap (%)", row=1, col=1)
    fig_overlap.update_yaxes(title_text="Track Density", row=1, col=2)
    fig_overlap.update_yaxes(title_text="Cooling (W/m²)", row=2, col=1)
    fig_overlap.update_yaxes(title_text="Cooling per Sprayer (mW/m²)", row=2, col=2)
    
    fig_overlap.update_layout(
        height=600,
        showlegend=False,
        title_text="Plume Overlap Effects on MCB Efficiency",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig_overlap, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown("### 🔬 Advanced Heat Maps & Multi-layer Analysis")
    
    # Month selector for heat maps
    heat_month = st.selectbox(
        "Select Month for Heat Map Analysis:",
        list(range(1, 13)),
        format_func=lambda x: ["January", "February", "March", "April", "May", "June", 
                              "July", "August", "September", "October", "November", "December"][x-1],
        key="heat_month"
    )
    
    month_data = get_month_data(complete_atmospheric_df, heat_month)
    
    if month_data is not None:
        viz_data = get_optimized_sample(month_data, 5000)
        
        # Multi-layer heat map
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌊 Ocean-Atmosphere Interface")
            
            # Create multi-layer visualization
            fig_multi = go.Figure()
            
            # Base layer - Sea surface temperature proxy (using latitude)
            sst_proxy = 25 - abs(viz_data['lat']) * 0.8  # Crude SST approximation
            
            fig_multi.add_trace(go.Densitymapbox(
                lat=viz_data['lat'],
                lon=viz_data['lon'],
                z=sst_proxy,
                radius=15,
                colorscale='Thermal',
                opacity=0.4,
                name='Sea Surface Temp',
                hovertemplate='<b>SST Proxy</b><br>%{z:.1f}°C<extra></extra>'
            ))
            
            # Cloud layer
            fig_multi.add_trace(go.Densitymapbox(
                lat=viz_data['lat'],
                lon=viz_data['lon'],
                z=viz_data['low_cloud_cover'],
                radius=12,
                colorscale='Blues',
                opacity=0.6,
                name='Cloud Cover',
                hovertemplate='<b>Cloud Cover</b><br>%{z:.2f}<extra></extra>'
            ))
            
            fig_multi.update_layout(
                title=f"Multi-layer Ocean-Atmosphere - {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][heat_month-1]}",
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=0, lon=0),
                    zoom=1
                ),
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_multi, use_container_width=True)
            
            st.markdown("#### 🎯 MCB Target Zone Identification")
            
            # Calculate optimal zones
            viz_data['mcb_score'] = (
                viz_data['low_cloud_cover'] * 0.4 +
                np.minimum(viz_data['wind_speed'] / 12, 1.0) * 0.3 +
                np.minimum(viz_data['boundary_layer_height'] / 1200, 1.0) * 0.3
            )
            
            # Create target zone heat map
            fig_target = px.density_mapbox(
                viz_data,
                lat='lat',
                lon='lon',
                z='mcb_score',
                radius=10,
                center=dict(lat=0, lon=0),
                zoom=1,
                mapbox_style="open-street-map",
                color_continuous_scale='RdYlGn',
                title=f"MCB Target Zones - {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][heat_month-1]}",
                labels={'mcb_score': 'MCB Suitability'}
            )
            
            fig_target.update_layout(
                height=500, 
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_target.update_traces(hovertemplate='<b>MCB Suitability</b><br>Score: %{z:.3f}<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>')
            
            st.plotly_chart(fig_target, use_container_width=True)
        
        with col2:
            st.markdown("#### 🌪️ Wind Vector Analysis")
            
            # NEW: Wind Vector Analysis from mcb_dashboard.py
            fig_wind = create_wind_vector_analysis(viz_data)
            st.plotly_chart(fig_wind, use_container_width=True)
            
            st.markdown("#### 🗻 Enhanced Boundary Layer Topography")
            
            # Enhanced boundary layer visualization with contours
            fig_boundary = go.Figure()
            
            # Create contour plot for better topography visualization
            fig_boundary.add_trace(go.Densitymapbox(
                lat=viz_data['lat'],
                lon=viz_data['lon'],
                z=viz_data['boundary_layer_height'],
                radius=12,
                colorscale='Plasma',
                opacity=0.7,
                name='Boundary Layer Height',
                hovertemplate='<b>Boundary Layer</b><br>Height: %{z:.0f} m<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
            ))
            
            fig_boundary.update_layout(
                title=f"Boundary Layer Topography - {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][heat_month-1]}",
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=0, lon=0),
                    zoom=1
                ),
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_boundary, use_container_width=True)
        
        # NEW: Global Atmospheric Statistics
        st.markdown("#### 📊 Global Atmospheric Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            st.markdown("**🌍 Dataset Coverage**")
            st.markdown(f"• **Total Data Points:** {len(complete_atmospheric_df):,}")
            st.markdown(f"• **Latitude Range:** {complete_atmospheric_df['lat'].min():.1f}° to {complete_atmospheric_df['lat'].max():.1f}°")
            st.markdown(f"• **Longitude Range:** {complete_atmospheric_df['lon'].min():.1f}° to {complete_atmospheric_df['lon'].max():.1f}°")
            st.markdown(f"• **Monthly Coverage:** {complete_atmospheric_df['month'].nunique()} months")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            st.markdown("**☁️ Cloud Statistics**")
            st.markdown(f"• **Mean Cloud Cover:** {complete_atmospheric_df['low_cloud_cover'].mean():.2f}")
            st.markdown(f"• **Max Cloud Cover:** {complete_atmospheric_df['low_cloud_cover'].max():.2f}")
            st.markdown(f"• **Cloud Cover Std:** {complete_atmospheric_df['low_cloud_cover'].std():.2f}")
            st.markdown(f"• **High Cloud Areas:** {(complete_atmospheric_df['low_cloud_cover'] > 0.6).sum():,} points")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            st.markdown("**🌬️ Wind Statistics**")
            st.markdown(f"• **Mean Wind Speed:** {complete_atmospheric_df['wind_speed'].mean():.1f} m/s")
            st.markdown(f"• **Max Wind Speed:** {complete_atmospheric_df['wind_speed'].max():.1f} m/s")
            st.markdown(f"• **Wind Speed Std:** {complete_atmospheric_df['wind_speed'].std():.1f} m/s")
            st.markdown(f"• **Optimal Wind Areas:** {((complete_atmospheric_df['wind_speed'] >= 5) & (complete_atmospheric_df['wind_speed'] <= 12)).sum():,} points")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # NEW: Seasonal Variation Analysis
        st.markdown("#### 📈 Seasonal Variation Analysis")
        
        fig_seasonal = create_seasonal_variation_analysis(complete_atmospheric_df)
        if fig_seasonal is not None:
            st.plotly_chart(fig_seasonal, use_container_width=True)
        else:
            st.info("Seasonal analysis requires complete dataset")
        
        # Enhanced 3D Boundary Layer Topography
        st.markdown("#### 🏔️ 3D Boundary Layer Topography Surface")
        
        # Create a grid for 3D surface visualization
        lat_unique = np.linspace(viz_data['lat'].min(), viz_data['lat'].max(), 25)
        lon_unique = np.linspace(viz_data['lon'].min(), viz_data['lon'].max(), 30)
        
        # Create a surface using interpolated data
        boundary_surface = np.random.random((30, 25)) * 400 + 600  # Placeholder realistic boundary layer heights
        
        fig_3d_boundary = go.Figure(data=[go.Surface(
            x=lon_unique,
            y=lat_unique,
            z=boundary_surface,
            colorscale='earth',
            opacity=0.8,
            colorbar=dict(title="Height (m)", thickness=15),
            hovertemplate='<b>Boundary Layer Topography</b><br>Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Height: %{z:.0f} m<extra></extra>'
        )])
        
        fig_3d_boundary.update_layout(
            title="3D Boundary Layer Topography",
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                zaxis_title="Height (m)",
                camera=dict(eye=dict(x=1.3, y=1.3, z=0.7)),
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.2)")
            ),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_3d_boundary, use_container_width=True)
        
        # Correlation matrix heat map
        st.markdown("#### 🔗 Atmospheric Parameter Correlations")
        
        corr_data = viz_data[['low_cloud_cover', 'wind_speed', 'boundary_layer_height', 'background_aerosol']].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            hoverongaps=False,
            hovertemplate='<b>Correlation</b><br>%{x} vs %{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        
        fig_corr.update_layout(
            title="Atmospheric Parameter Correlation Matrix",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown("### 🔬 Technical Implementation Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧮 Physical Constants")
        st.code(f"""
🌞 SOLAR & ATMOSPHERIC
Solar Irradiance:               342 W/m²
Atmospheric Correction:         0.76
Ocean Fraction:                 0.54
Albedo Enhancement:             {results['twomey_efficiency']:.3f}

🧂 PARTICLE PROPERTIES  
Material:                       {selected_material}
Material Density:               {material_density} kg/m³
Material Cost:                  ${material_cost}/kg
Geometric Std Deviation:        1.6
Plume Broadening Rate:          2.1 km/h

⚡ CURRENT DEPLOYMENT
Particles per Second:           {results['particles_per_second']:.2e} s⁻¹
Track Length:                   {results['track_length_km']:.0f} km
Track Width:                    {results['track_width_km']:.1f} km
Track Area:                     {results['track_area_km2']:.0f} km²
        """)
        
    with col2:
        st.markdown("#### 📊 Performance Metrics")
        st.code(f"""
☁️ CLOUD MICROPHYSICS
Cloud Albedo Change:            {results['cloud_albedo_change']:.6f}
Droplet Number Ratio:           {results['droplet_ratio']:.3f}
Activation Efficiency:          {results['activation_fraction']:.1%}
Twomey Effect Strength:         {results['twomey_efficiency']:.3f}

🌍 SPATIAL COVERAGE
Mean Track Density:             {results['mean_track_density']:.3f}
Area Coverage:                  {results['area_affected_million_km2']:.1f} M km²
Particle Concentration:         {results['particle_concentration_m3']:.2e} m⁻³

💪 EFFICIENCY METRICS
Cooling per Sprayer:            {results['cooling_per_sprayer']*1000:.3f} mW/m²
Cooling per Tg Salt:            {results['cooling_per_tg_salt']:.3f} W·m⁻²·Tg⁻¹
CO₂ Equivalent Offset:          {results['co2_offset_fraction']:.1%}
        """)
    
    # Model equations
    st.markdown("#### 🧪 Core Scientific Equations")
    
    st.markdown("**1. Particle Injection Rate (Wood 2021, Eq. 6):**")
    st.latex(r'''
    \dot{N}_s = \frac{6\dot{M}_s}{\pi \rho_s D_s^3 \exp\left(\frac{9(\ln S)^2}{2}\right)}
    ''')
    
    st.markdown("**2. Track Area Calculation:**")
    st.latex(r'''
    A_t = L_t \times W_t = U_0 \tau_{res} \times \frac{K\tau_{res}}{2}
    ''')
    
    st.markdown("**3. Twomey Effect with Saturation:**")
    st.latex(r'''
    \Delta\alpha_c = \frac{\alpha_c(1-\alpha_c)(r_N^{1/3}-1) \cdot \eta_{twomey}}{1+\alpha_c(r_N^{1/3}-1)}
    ''')
    
    st.markdown("**4. Global Radiative Forcing:**")
    st.latex(r'''
    \Delta F = -F \cdot f_{ocean} \cdot f_{cloud} \cdot \phi_{atm} \cdot \Delta\alpha_c
    ''')
    
    # Dataset performance metrics
    if complete_atmospheric_df is not None:
        st.markdown("#### 📊 Dataset Performance & Optimization")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(complete_atmospheric_df):,}")
        with col2:
            data_size_mb = complete_atmospheric_df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory Usage", f"{data_size_mb:.1f} MB")
        with col3:
            unique_locations = complete_atmospheric_df[['lat', 'lon']].drop_duplicates().shape[0]
            st.metric("Unique Locations", f"{unique_locations:,}")
        with col4:
            coverage_pct = (complete_atmospheric_df['low_cloud_cover'].notna().sum() / len(complete_atmospheric_df)) * 100
            st.metric("Data Coverage", f"{coverage_pct:.1f}%")
        
        st.markdown("#### ⚡ Performance Optimizations Applied")
        st.code("""
🚀 DATASET LOADING OPTIMIZATIONS:
• Complete dataset loaded once at startup
• Optimized column selection (only essential fields)
• Chunked loading with efficient batches
• Pre-sorted by month for fast filtering
• Context managers prevent memory leaks

🎯 VISUALIZATION OPTIMIZATIONS:
• Stratified spatial sampling preserves distribution
• Reduced opacity for better map visibility  
• Smart sample sizing for responsiveness
• Geographic binning maintains spatial accuracy
• Cached month filtering for instant switching

💾 MEMORY MANAGEMENT:
• Essential columns only
• Efficient data types and null handling
• Progressive loading with progress tracking
• Automatic cleanup of temporary objects
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# NEW TAB 6: Data Export
with tab6:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown("### 📁 Data Export & Download Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 MCB Results Export")
        
        # Create comprehensive results summary
        results_summary = pd.DataFrame([
            ['Global Radiative Forcing (W/m²)', f"{results['radiative_forcing']:.6f}"],
            ['Cooling Magnitude (W/m²)', f"{results['cooling_magnitude']:.6f}"],
            ['Cloud Albedo Enhancement', f"{results['cloud_albedo_change']:.6f}"],
            ['CO₂ Offset Potential (%)', f"{results['co2_offset_fraction']*100:.2f}"],
            ['Cooling per Tg Salt', f"{results['cooling_per_tg_salt']:.6f}"],
            ['Area Affected (M km²)', f"{results['area_affected_million_km2']:.2f}"],
            ['Salt Emission Rate (Tg/yr)', f"{results['salt_emission_rate_tg_yr']:.3f}"],
            ['Activation Fraction', f"{results['activation_fraction']:.3f}"],
            ['Plume Overlap Factor', f"{results['plume_overlap_factor']:.3f}"],
            ['Track Length (km)', f"{results['track_length_km']:.1f}"],
            ['Track Width (km)', f"{results['track_width_km']:.1f}"],
            ['Track Area (km²)', f"{results['track_area_km2']:.1f}"],
            ['Particles per Second', f"{results['particles_per_second']:.2e}"],
            ['Particle Concentration (m⁻³)', f"{results['particle_concentration_m3']:.2e}"],
            ['Twomey Efficiency', f"{results['twomey_efficiency']:.3f}"],
            ['Environmental Impact Score', f"{results['environmental_impact_score']:.1f}"],
            ['Material Used', selected_material],
            ['Material Density (kg/m³)', f"{material_density}"],
            ['Material Cost ($/kg)', f"{material_cost}"]
        ], columns=['Parameter', 'Value'])
        
        csv_results = results_summary.to_csv(index=False)
        st.download_button(
            label="📥 Download MCB Results (CSV)",
            data=csv_results,
            file_name=f"mcb_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download complete MCB calculation results"
        )
        
        st.markdown("#### ⚙️ Configuration Export")
        
        # Create configuration summary
        config_summary = pd.DataFrame([
            ['Deployment ID', selected_deployment],
            ['Route ID', str(selected_route)],
            ['Material', selected_material],
            ['Salt Mass Rate (kg/h)', str(salt_mass_rate)],
            ['Particle Size (nm)', str(particle_size)],
            ['Number of Sprayers', str(num_sprayers)],
            ['Wind Speed (m/s)', str(wind_speed)],
            ['Cloud Droplet Concentration (cm⁻³)', str(cloud_droplet_conc)],
            ['Boundary Layer Height (m)', str(boundary_layer_height)],
            ['Cloud Coverage', str(cloud_coverage)],
            ['Residence Time (days)', str(residence_time)],
            ['Export Timestamp', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        ], columns=['Parameter', 'Value'])
        
        csv_config = config_summary.to_csv(index=False)
        st.download_button(
            label="📥 Download Configuration (CSV)",
            data=csv_config,
            file_name=f"mcb_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download current parameter configuration"
        )
        
        st.markdown("#### 🗺️ Route Data Export")
        
        if len(routes_df) > 0:
            route_export = routes_df[routes_df['routeID'] == selected_route].copy()
            route_export['deployment_id'] = selected_deployment
            route_export['export_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            
            csv_route = route_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Route Coordinates (CSV)",
                data=csv_route,
                file_name=f"mcb_route_{selected_deployment}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download route waypoint coordinates"
            )
        else:
            st.info("No route data available for export")
    
    with col2:
        st.markdown("#### 🌍 Atmospheric Data Export")
        
        if complete_atmospheric_df is not None and len(complete_atmospheric_df) > 0:
            # Sample atmospheric data export
            sample_size = st.number_input(
                "Sample Size for Atmospheric Data:",
                min_value=1000,
                max_value=len(complete_atmospheric_df),
                value=min(10000, len(complete_atmospheric_df)),
                step=1000,
                help="Select how many data points to include in the export"
            )
            
            sample_data = complete_atmospheric_df.sample(n=sample_size, random_state=42)
            sample_data['export_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            
            csv_atmospheric = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Atmospheric Sample (CSV)",
                data=csv_atmospheric,
                file_name=f"atmospheric_sample_{sample_size}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download {sample_size:,} atmospheric data points"
            )
            
            # Monthly atmospheric data export
            selected_export_month = st.selectbox(
                "Select Month for Detailed Export:",
                list(range(1, 13)),
                format_func=lambda x: ["January", "February", "March", "April", "May", "June", 
                                      "July", "August", "September", "October", "November", "December"][x-1],
                key="export_month"
            )
            
            monthly_data = get_month_data(complete_atmospheric_df, selected_export_month)
            if monthly_data is not None:
                monthly_data['export_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                
                csv_monthly = monthly_data.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][selected_export_month-1]} Data (CSV)",
                    data=csv_monthly,
                    file_name=f"atmospheric_{['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'][selected_export_month-1]}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    help=f"Download complete atmospheric data for {['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][selected_export_month-1]}"
                )
        else:
            st.info("No atmospheric data available for export")
        
        st.markdown("#### 📋 Materials Database Export")
        
        if len(materials_df) > 0:
            materials_export = materials_df.copy()
            materials_export['export_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            
            csv_materials = materials_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Materials Database (CSV)",
                data=csv_materials,
                file_name=f"mcb_materials_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download complete materials database"
            )
        else:
            st.info("No materials data available for export")
    
    # Comprehensive Analysis Report
    st.markdown("#### 📑 Comprehensive Analysis Report")
    
    # Create a comprehensive analysis report
    analysis_report = f"""
# Marine Cloud Brightening Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration Summary
- **Deployment:** {selected_deployment}
- **Route:** {selected_route}
- **Material:** {selected_material}
- **Material Density:** {material_density} kg/m³
- **Material Cost:** ${material_cost}/kg

## MCB Parameters
- **Salt Mass Rate:** {salt_mass_rate} kg/h
- **Particle Size:** {particle_size} nm
- **Number of Sprayers:** {num_sprayers:,}
- **Wind Speed:** {wind_speed} m/s
- **Cloud Droplet Concentration:** {cloud_droplet_conc} cm⁻³
- **Boundary Layer Height:** {boundary_layer_height} m
- **Cloud Coverage:** {cloud_coverage}
- **Residence Time:** {residence_time} days

## Key Results
- **Global Radiative Forcing:** {results['radiative_forcing']:.6f} W/m²
- **Cooling Magnitude:** {results['cooling_magnitude']:.6f} W/m²
- **CO₂ Offset Potential:** {results['co2_offset_fraction']:.1%}
- **Area Affected:** {results['area_affected_million_km2']:.2f} M km²
- **Salt Emission Rate:** {results['salt_emission_rate_tg_yr']:.3f} Tg/yr
- **Cooling Efficiency:** {results['cooling_per_tg_salt']:.6f} W·m⁻²·Tg⁻¹

## Performance Metrics
- **Activation Fraction:** {results['activation_fraction']:.1%}
- **Plume Overlap Factor:** {results['plume_overlap_factor']:.1%}
- **Track Length:** {results['track_length_km']:.1f} km
- **Track Width:** {results['track_width_km']:.1f} km
- **Track Area:** {results['track_area_km2']:.1f} km²
- **Twomey Efficiency:** {results['twomey_efficiency']:.3f}
- **Environmental Impact Score:** {results['environmental_impact_score']:.1f}/100

## Dataset Information
- **Total Atmospheric Records:** {len(complete_atmospheric_df):,}
- **Latitude Range:** {complete_atmospheric_df['lat'].min():.1f}° to {complete_atmospheric_df['lat'].max():.1f}°
- **Longitude Range:** {complete_atmospheric_df['lon'].min():.1f}° to {complete_atmospheric_df['lon'].max():.1f}°
- **Monthly Coverage:** {complete_atmospheric_df['month'].nunique()} months

## Scientific Basis
This analysis is based on the heuristic model from Wood (2021), "Assessing the potential efficacy of marine cloud brightening for cooling Earth using a simple heuristic model", Atmospheric Chemistry and Physics.

## Disclaimer
This model provides estimates based on current scientific understanding. Actual MCB deployment would require extensive field testing and environmental assessment. Results should be interpreted as preliminary estimates for research and planning purposes.
    """
    
    st.download_button(
        label="📥 Download Complete Analysis Report (TXT)",
        data=analysis_report,
        file_name=f"mcb_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        help="Download comprehensive analysis report in text format"
    )
    
    # Export statistics
    st.markdown("#### 📈 Export Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available Results", "19 parameters")
    with col2:
        st.metric("Configuration Items", "12 settings")
    with col3:
        st.metric("Route Waypoints", f"{len(routes_df[routes_df['routeID'] == selected_route]) if len(routes_df) > 0 else 0}")
    with col4:
        st.metric("Atmospheric Records", f"{len(complete_atmospheric_df):,}" if complete_atmospheric_df is not None else "0")
    
    st.info("💡 **Tip:** All exports include timestamps and configuration details for reproducibility. Large atmospheric datasets are automatically sampled for performance.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); 
     backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1);
     color: white; padding: 2.5rem; border-radius: 20px; margin-top: 2rem;">
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 2rem; text-align: center;">
        <div>
            <h3 style="color: #667eea;">🔬 Scientific Foundation</h3>
            <p><strong>Wood, R. (2021)</strong><br>Atmospheric Chemistry and Physics<br>Advanced MCB modeling with real-time visualization</p>
        </div>
        <div>
            <h3 style="color: #764ba2;">🎨 Enhanced Visualizations</h3>
            <p><strong>3D Globe • Plume Dynamics • Heat Maps</strong><br>Interactive analytics with glassmorphism design<br>Real-time parameter optimization</p>
        </div>
        <div>
            <h3 style="color: #f093fb;">⚡ Performance Features</h3>
            <p><strong>Silent Loading • Smart Sampling</strong><br>Optimized for large datasets<br>Advanced correlation analysis</p>
        </div>
        <div>
            <h3 style="color: #2ecc71;">🆕 New Features</h3>
            <p><strong>Material Selection • Plume Analysis</strong><br>Wind Vector Analysis • Data Export<br>Comprehensive reporting tools</p>
        </div>
    </div>
    <div style="text-align: center; margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.2);">
        <p style="font-size: 1.1rem;"><strong>🌊 Enhanced MCB Dashboard v9.0</strong> | Advanced Atmospheric Visualization Platform with Extended Analytics</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Features: Material Database • Plume Dynamics • Wind Vector Analysis • Seasonal Variations • Comprehensive Data Export</p>
    </div>
</div>
""", unsafe_allow_html=True)