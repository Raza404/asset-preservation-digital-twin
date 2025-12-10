"""
Flight Trajectory Visualization using OpenCV
Renders 2D/3D flight paths with component health color-coding.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Tuple, List


def map_coordinates_to_pixels(lat: float, lon: float, 
                              bounds: Tuple[float, float, float, float],
                              img_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Map GPS coordinates to pixel coordinates.
    
    Args:
        lat, lon: GPS coordinates
        bounds: (min_lat, max_lat, min_lon, max_lon)
        img_size: (width, height)
    
    Returns:
        (x, y) pixel coordinates
    """
    min_lat, max_lat, min_lon, max_lon = bounds
    width, height = img_size
    
    # Normalize to 0-1
    x_norm = (lon - min_lon) / (max_lon - min_lon) if max_lon != min_lon else 0.5
    y_norm = (lat - min_lat) / (max_lat - min_lat) if max_lat != min_lat else 0.5
    
    # Add margins
    margin = 50
    x = int(margin + x_norm * (width - 2 * margin))
    y = int(height - (margin + y_norm * (height - 2 * margin)))  # Flip Y
    
    return (x, y)


def get_risk_color(risk_level: float) -> Tuple[int, int, int]:
    """
    Get BGR color based on risk level.
    
    Args:
        risk_level: 0.0 (safe) to 1.0 (critical)
    
    Returns:
        (B, G, R) tuple
    """
    if risk_level < 0.3:
        return (0, 255, 0)  # Green - healthy
    elif risk_level < 0.7:
        return (0, 165, 255)  # Orange - warning
    else:
        return (0, 0, 255)  # Red - critical


def render_flight_trajectory(
    telemetry_df: pd.DataFrame,
    output_path: str = "flight_trajectory.png",
    img_size: Tuple[int, int] = (1200, 900),
    show_altitude: bool = True
) -> np.ndarray:
    """
    Render flight trajectory with component health color-coding.
    
    Args:
        telemetry_df: DataFrame with lat, lon, altitude, risk_level columns
        output_path: Path to save image
        img_size: (width, height)
        show_altitude: Show altitude as line thickness
    
    Returns:
        BGR image array
    """
    print(f"\n📊 Rendering flight trajectory...")
    print(f"   Samples: {len(telemetry_df)}")
    
    # Create white background
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
    
    # Get coordinate bounds
    if 'latitude' in telemetry_df.columns and 'longitude' in telemetry_df.columns:
        min_lat = telemetry_df['latitude'].min()
        max_lat = telemetry_df['latitude'].max()
        min_lon = telemetry_df['longitude'].min()
        max_lon = telemetry_df['longitude'].max()
    else:
        # Use dummy bounds if GPS not available
        min_lat, max_lat = 0, 1
        min_lon, max_lon = 0, 1
    
    bounds = (min_lat, max_lat, min_lon, max_lon)
    
    # Add title
    cv2.putText(img, "Flight Trajectory Analysis", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Draw trajectory
    points = []
    for idx, row in telemetry_df.iterrows():
        if 'latitude' in row and 'longitude' in row:
            lat, lon = row['latitude'], row['longitude']
        else:
            # Use index-based positioning if GPS unavailable
            lat = idx / len(telemetry_df)
            lon = row.get('altitude', 50) / 200.0
        
        x, y = map_coordinates_to_pixels(lat, lon, bounds, img_size)
        
        # Calculate risk level
        if 'risk_level' in row:
            risk = row['risk_level']
        else:
            # Default low risk
            risk = 0.2
        
        color = get_risk_color(risk)
        
        # Draw point
        cv2.circle(img, (x, y), 3, color, -1)
        points.append((x, y, color, risk))
        
        # Draw line to previous point
        if len(points) > 1:
            prev_x, prev_y, prev_color, _ = points[-2]
            thickness = 2
            if show_altitude and 'altitude' in row:
                # Thicker line for higher altitude
                thickness = max(1, min(8, int(row['altitude'] / 20)))
            
            cv2.line(img, (prev_x, prev_y), (x, y), color, thickness)
    
    # Add start/end markers
    if points:
        start_x, start_y = points[0][:2]
        end_x, end_y = points[-1][:2]
        
        cv2.circle(img, (start_x, start_y), 15, (255, 0, 0), 3)  # Blue start
        cv2.putText(img, "START", (start_x - 30, start_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.circle(img, (end_x, end_y), 15, (0, 0, 0), 3)  # Black end
        cv2.putText(img, "END", (end_x - 20, end_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add legend
    legend_x = img_size[0] - 250
    legend_y = 100
    
    cv2.putText(img, "Component Health:", (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Green - Healthy
    cv2.rectangle(img, (legend_x, legend_y + 20), 
                 (legend_x + 40, legend_y + 40), (0, 255, 0), -1)
    cv2.putText(img, "Healthy", (legend_x + 50, legend_y + 37),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Orange - Warning
    cv2.rectangle(img, (legend_x, legend_y + 50), 
                 (legend_x + 40, legend_y + 70), (0, 165, 255), -1)
    cv2.putText(img, "Warning", (legend_x + 50, legend_y + 67),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Red - Critical
    cv2.rectangle(img, (legend_x, legend_y + 80), 
                 (legend_x + 40, legend_y + 100), (0, 0, 255), -1)
    cv2.putText(img, "Critical", (legend_x + 50, legend_y + 97),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add statistics
    stats_y = legend_y + 140
    cv2.putText(img, "Flight Statistics:", (legend_x, stats_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    stats = [
        f"Duration: {len(telemetry_df) * 0.1:.1f}s",
        f"Samples: {len(telemetry_df)}",
    ]
    
    if 'altitude' in telemetry_df.columns:
        stats.append(f"Max Alt: {telemetry_df['altitude'].max():.1f}m")
    
    if 'ground_speed' in telemetry_df.columns:
        stats.append(f"Max Speed: {telemetry_df['ground_speed'].max():.1f}m/s")
    
    for i, stat in enumerate(stats):
        cv2.putText(img, stat, (legend_x, stats_y + 30 + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add warning count
    if 'risk_level' in telemetry_df.columns:
        warnings = (telemetry_df['risk_level'] >= 0.4).sum()
        critical = (telemetry_df['risk_level'] >= 0.7).sum()
        
        alert_y = stats_y + 30 + len(stats) * 25 + 20
        cv2.putText(img, "Alerts:", (legend_x, alert_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"⚠ Warnings: {warnings}", (legend_x, alert_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(img, f"🚨 Critical: {critical}", (legend_x, alert_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Save image
    cv2.imwrite(output_path, img)
    print(f"✓ Trajectory saved to {output_path}")
    
    return img


def demo_trajectory_visualization():
    """Demo: Visualize sample flight trajectory."""
    print("="*60)
    print("OpenCV Flight Trajectory Visualization Demo")
    print("="*60)
    
    # Load sample flight data
    try:
        df = pd.read_csv("data/raw/sample_flight_log.csv")
        print(f"\n✓ Loaded {len(df)} telemetry samples")
        
        # Subsample for clearer visualization
        df_sample = df[::20]  # Every 20th point
        
        # Add mock risk levels based on altitude and speed
        df_sample['risk_level'] = 0.2  # Default healthy
        
        # Simulate risk increases
        if 'altitude' in df_sample.columns:
            high_alt = df_sample['altitude'] > df_sample['altitude'].quantile(0.8)
            df_sample.loc[high_alt, 'risk_level'] = 0.5
        
        if 'ground_speed' in df_sample.columns:
            high_speed = df_sample['ground_speed'] > df_sample['ground_speed'].quantile(0.9)
            df_sample.loc[high_speed, 'risk_level'] = 0.8
        
        # Render trajectory
        img = render_flight_trajectory(df_sample, "flight_trajectory.png")
        
        # Display
        print("\nDisplaying visualization (press any key to close)...")
        cv2.imshow("Flight Trajectory", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\n✓ Demo complete!")
        
    except FileNotFoundError:
        print("❌ Sample flight log not found at data/raw/sample_flight_log.csv")
        print("   Creating synthetic trajectory instead...")
        
        # Create synthetic data
        n_points = 100
        df_synthetic = pd.DataFrame({
            'latitude': 17.4 + np.random.randn(n_points) * 0.001,
            'longitude': 78.4 + np.random.randn(n_points) * 0.001,
            'altitude': 50 + np.sin(np.linspace(0, 4*np.pi, n_points)) * 20,
            'risk_level': np.clip(np.random.randn(n_points) * 0.2 + 0.3, 0, 1)
        })
        
        img = render_flight_trajectory(df_synthetic, "flight_trajectory_synthetic.png")
        
        print("\nDisplaying visualization (press any key to close)...")
        cv2.imshow("Flight Trajectory (Synthetic)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\n✓ Demo complete!")


if __name__ == "__main__":
    try:
        import cv2
        demo_trajectory_visualization()
    except ImportError:
        print("❌ OpenCV not installed. Install with:")
        print("   pip install opencv-python")
