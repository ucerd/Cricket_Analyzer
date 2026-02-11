import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*Trying to examine the path of torch.classes raised.*')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
from typing import List, Tuple, Dict
import numpy as np
import cv2
from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from filterpy.kalman import KalmanFilter
import logging
import yaml
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="Enhanced Cricket Analysis with Crease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_cricket_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def load_yolo_model(model_path):
    """Load YOLO model while isolating torch import"""
    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    return model, device

def plot_results(frame_numbers, speeds_kmh, positions):
    """Function to plot various analysis results"""
    fig_speed = px.line(x=frame_numbers, y=speeds_kmh, title="Speed Trend Over Time", labels={"x": "Frame Number", "y": "Speed (km/h)"})
    fig_histogram = px.histogram(speeds_kmh, nbins=20, title="Speed Distribution", labels={"value": "Speed (km/h)", "count": "Frequency"})
    fig_scatter = px.scatter(x=[p[0] for p in positions], y=[p[1] for p in positions], title="Ball Position Scatter Plot", labels={"x": "X Position", "y": "Y Position"})
    fig_box = px.box(y=speeds_kmh, title="Speed Box Plot", labels={"y": "Speed (km/h)"})
    return fig_speed, fig_histogram, fig_scatter, fig_box

def create_export_file(frame_numbers, timestamps, smoothed_positions, speeds, fps, meters_per_pixel, pitch_length_pixels):
    """
    Create an Excel file containing ball tracking analysis with dynamic pitch length and improved formatting
    Modified to exclude speeds above threshold from the Excel file
    """
    try:
        decimal_places = 3
        min_frame_diff = 1  
        frame_duration = 1.0 / fps  
        pitch_length_meters = st.session_state.pitch_length_meters
        max_threshold = 125.0  
        
        data = {
            'Frame Number': [],
            'X Position': [],
            'Y Position': [],
            'Distance (pixels)': [],
            'Distance (meters)': [],
            'Frame Difference': [],
            'Time Interval (seconds)': [],
            'Speed (m/s)': [],
            'Speed (km/h)': [],
            'In Crease Region': []
        }
        
        excluded_count = 0  
        
        for i in range(1, len(smoothed_positions)):
            if any(pd.isna(val) for val in smoothed_positions[i-1] + smoothed_positions[i]):
                logger.warning(f"NaN values detected in positions at index {i}, skipping")
                continue
                
            x1, y1 = map(int, map(round, smoothed_positions[i-1]))
            x2, y2 = map(int, map(round, smoothed_positions[i]))
            
            frame_diff = frame_numbers[i] - frame_numbers[i-1]
            
            
            dx = x2 - x1
            dy = y2 - y1
            distance_pixels = np.sqrt(dx**2 + dy**2)   
            
            time_interval = frame_diff * frame_duration
            distance_meters = distance_pixels * meters_per_pixel
            
            
            if time_interval > 0 and frame_diff >= min_frame_diff:
                speed_ms = distance_meters / time_interval
                speed_kmh = speed_ms * 3.6
                
               
                if speed_kmh > max_threshold:
                    
                    excluded_count += 1
                    continue  
                elif speed_kmh < 0:
                    speed_kmh = 0
                    speed_ms = 0
            else:
                speed_ms = 0
                speed_kmh = 0
            
            
            in_crease_region = False
            if hasattr(st.session_state, 'roi_coordinates') and st.session_state.roi_coordinates:
                roi = st.session_state.roi_coordinates
                if hasattr(roi, 'is_point_inside_crease_region'):
                    in_crease_region = roi.is_point_inside_crease_region(x2, y2)
            
            
            data['Frame Number'].append(frame_numbers[i])
            data['X Position'].append(x2)
            data['Y Position'].append(y2)
            data['Distance (pixels)'].append(round(distance_pixels, decimal_places))
            data['Distance (meters)'].append(round(distance_meters, decimal_places))
            data['Frame Difference'].append(frame_diff)
            data['Time Interval (seconds)'].append(round(time_interval, decimal_places))
            data['Speed (m/s)'].append(round(speed_ms, decimal_places))
            data['Speed (km/h)'].append(round(speed_kmh, decimal_places))
            data['In Crease Region'].append(in_crease_region)
        
        analysis_df = pd.DataFrame(data)
        
      
        logger.info(f"Excel Export: Included {len(data['Speed (km/h)'])} rows, excluded {excluded_count} rows (above {max_threshold} km/h)")
        
        
        valid_speeds = [s for s in data['Speed (km/h)'] if s > 0]  
        crease_speeds = [data['Speed (km/h)'][i] for i in range(len(data['Speed (km/h)'])) 
                        if data['In Crease Region'][i] and data['Speed (km/h)'][i] > 0]
        
        if valid_speeds:
            
            filtered_speeds = SpeedDataFilter.filter_speed_outliers(valid_speeds)
            avg_speed = np.mean(filtered_speeds) if filtered_speeds else 0
            max_speed = np.max(filtered_speeds) if filtered_speeds else 0
            sustained_speed = SpeedDataFilter.get_max_sustained_speed(filtered_speeds)
        else:
            avg_speed = max_speed = sustained_speed = 0
            
        if crease_speeds:
            filtered_crease_speeds = SpeedDataFilter.filter_speed_outliers(crease_speeds)
            avg_crease_speed = np.mean(filtered_crease_speeds) if filtered_crease_speeds else 0
            max_crease_speed = np.max(filtered_crease_speeds) if filtered_crease_speeds else 0
        else:
            avg_crease_speed = max_crease_speed = 0
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            analysis_df.to_excel(writer, sheet_name='Analysis', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Analysis']
            
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'vcenter',
                'align': 'center',
                'fg_color': '#C6EFCE',  
                'border': 1,
                'border_color': '#000000',
                'font_size': 11,
                'font_name': 'Calibri'
            })
            
            data_format = workbook.add_format({
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'border_color': '#000000',
                'text_wrap': True,
                'font_size': 11,
                'font_name': 'Calibri'
            })
            
            number_format = workbook.add_format({
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'border_color': '#000000',
                'num_format': '0.000',
                'font_size': 11,
                'font_name': 'Calibri'
            })
            
            for col_num, value in enumerate(analysis_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
                max_length = max(
                    len(str(value)),
                    analysis_df[value].astype(str).str.len().max()
                )
                worksheet.set_column(col_num, col_num, max_length + 2)
            
            for row_num in range(len(analysis_df)):
                for col_num, value in enumerate(analysis_df.iloc[row_num]):
                    if isinstance(value, (int, float)):
                        worksheet.write(row_num + 1, col_num, value, number_format)
                    else:
                        worksheet.write(row_num + 1, col_num, value, data_format)
            
            summary_stats = [
                ("Average Speed (km/h)", avg_speed),
                ("Maximum Speed (km/h)", max_speed),
                ("Sustained Speed (km/h)", sustained_speed),
                ("Average Crease Speed (km/h)", avg_crease_speed),
                ("Maximum Crease Speed (km/h)", max_crease_speed),
                ("Pitch Length (pixels)", pitch_length_pixels),
                ("Pitch Length (meters)", pitch_length_meters),
                ("Meters per Pixel", meters_per_pixel),
                ("FPS", fps),
                ("Frame Duration (seconds)", frame_duration),
                ("Crease Detections", len(crease_speeds)),
                ("Total Detections", len(valid_speeds)),
                ("Speed Threshold Applied (km/h)", max_threshold),  
                ("Excluded High Speed Rows", excluded_count)  
            ]
            
            summary_row = len(analysis_df) + 2
            worksheet.write(summary_row, 0, "Summary Statistics", header_format)
            
            stats_format = workbook.add_format({
                'align': 'left',
                'valign': 'vcenter',
                'border': 1,
                'border_color': '#000000',
                'font_size': 11,
                'font_name': 'Calibri'
            })
            
            stats_value_format = workbook.add_format({
                'align': 'center',
                'valign': 'vcenter',
                'border': 1,
                'border_color': '#000000',
                'num_format': '0.000',
                'font_size': 11,
                'font_name': 'Calibri'
            })
            
            for i, (label, value) in enumerate(summary_stats):
                worksheet.write(summary_row + 1 + i, 0, label, stats_format)
                worksheet.write(summary_row + 1 + i, 1, value, stats_value_format)
                worksheet.set_row(summary_row + 1 + i, 20)  
            
            formula_row = summary_row + len(summary_stats) + 2
            worksheet.write(formula_row, 0, "Speed Calculation Formula", header_format)
            worksheet.write(formula_row + 1, 0, "Distance (meters) = Distance (pixels) × Meters per Pixel", stats_format)
            worksheet.write(formula_row + 2, 0, "Time Interval (seconds) = Frame Difference × Frame Duration", stats_format)
            worksheet.write(formula_row + 3, 0, "Speed (m/s) = Distance (meters) / Time Interval (seconds)", stats_format)
            worksheet.write(formula_row + 4, 0, "Speed (km/h) = Speed (m/s) × 3.6", stats_format)
            worksheet.write(formula_row + 5, 0, "Crease Region: 4 feet (1.22m) from wickets", stats_format)
            worksheet.write(formula_row + 6, 0, "Outlier filtering: IQR method applied", stats_format)
            worksheet.write(formula_row + 7, 0, f"Excluded {excluded_count} measurements above {max_threshold} km/h", stats_format)  # NEW
            
            worksheet.freeze_panes(1, 0)
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating export file: {str(e)}")
        raise



@dataclass
class SpeedCalculator:
    """Speed calculator using dynamic cricket pitch length with corrected FPS processing"""
    
    def __init__(self, fps: float, pitch_length_pixels: int):
        """
        Initialize speed calculator with video and pitch parameters
        
        Args:
            fps (float): Video frames per second
            pitch_length_pixels (int): Detected pitch length in pixels
        """
        if fps <= 0:
            raise ValueError("FPS must be greater than 0")
        if pitch_length_pixels <= 0:
            raise ValueError("pitch_length_pixels must be greater than 0")
            
        self.fps = float(fps)
        self.decimal_places = 3
        
        
        self.min_frame_diff = 1  
            
        self.pitch_length_pixels = pitch_length_pixels
        self.pitch_length_meters = st.session_state.pitch_length_meters
        self.meters_per_pixel = self.pitch_length_meters / self.pitch_length_pixels
        
        if self.meters_per_pixel <= 0:
            raise ValueError("Calculated meters_per_pixel must be greater than 0")
            
        self.frame_duration = 1.0 / fps  
        
        logger.info(f"SpeedCalculator initialized with pitch length {pitch_length_pixels}px, "
                   f"{self.meters_per_pixel:.6f} m/px, {fps} FPS")

    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate distance between two positions with improved precision
        
        Args:
            pos1: Starting position (x, y)
            pos2: Ending position (x, y)
            
        Returns:
            float: Distance in pixels (keeping decimal precision)
        """
        try:
            if any(pd.isna(val) for val in pos1) or any(pd.isna(val) for val in pos2):
                logger.warning(f"NaN values detected in position data: {pos1}, {pos2}")
                return 0.0
    
            x1, y1 = float(pos1[0]), float(pos1[1])
            x2, y2 = float(pos2[0]), float(pos2[1])
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx**2 + dy**2)
            
            logger.debug(f"Distance calculation: ({x1:.2f},{y1:.2f}) to ({x2:.2f},{y2:.2f}) = {distance:.3f} pixels")
            
            return distance
            
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
            return 0.0

    def calculate_speed(self, pos1: Tuple[float, float], pos2: Tuple[float, float], frames_between: int) -> float:
        """
        Calculate speed between two positions with improved accuracy
        
        Args:
            pos1: Starting position (x, y)
            pos2: Ending position (x, y)
            frames_between: Number of frames between positions
            
        Returns:
            float: Speed in km/h, rounded to set decimal places
        """
        try:
            if frames_between < self.min_frame_diff:
                logger.debug(f"Frame difference {frames_between} below minimum threshold {self.min_frame_diff}")
                return round(0.0, self.decimal_places)
            
            if any(pd.isna(val) for val in pos1) or any(pd.isna(val) for val in pos2):
                logger.warning(f"NaN values detected in position data: {pos1}, {pos2}")
                return round(0.0, self.decimal_places)
                
            distance_pixels = self.calculate_distance(pos1, pos2)
            distance_meters = distance_pixels * self.meters_per_pixel
            time_interval = frames_between * self.frame_duration
            
            if time_interval <= 0:
                return round(0.0, self.decimal_places)
            
            speed_ms = distance_meters / time_interval
            speed_kmh = speed_ms * 3.6
            
            # Add speed validation to prevent unrealistic values with fixed threshold
            max_threshold = 125.0  # Fixed threshold
            if speed_kmh > max_threshold:
                logger.warning(f"Speed above threshold: {speed_kmh:.2f} km/h, filtering out")
                speed_kmh = 0.0  # Filter out instead of capping
            elif speed_kmh < 0:
                speed_kmh = 0.0
            
            speed_kmh = round(speed_kmh, self.decimal_places)
            
            logger.debug(f"Speed calculation: {distance_pixels:.3f}px ({distance_meters:.3f}m) "
                        f"over {time_interval:.3f}s = {speed_kmh} km/h")
            
            return speed_kmh
            
        except Exception as e:
            logger.error(f"Error calculating speed: {str(e)}")
            return round(0.0, self.decimal_places)

    def calculate_speeds_sequence(self, positions: List[Tuple[float, float]], frame_numbers: List[int]) -> List[float]:
        """
        Calculate a sequence of speeds from position and frame data with improved filtering
        
        Args:
            positions: List of (x, y) positions
            frame_numbers: List of corresponding frame numbers
            
        Returns:
            List[float]: List of calculated speeds in km/h
        """
        speeds = []
        valid_count = 0
        
        try:
            if len(positions) < 2 or len(positions) != len(frame_numbers):
                logger.warning("Invalid input sequences")
                return speeds
                
            for i in range(1, len(positions)):
                if any(pd.isna(val) for val in positions[i-1]) or any(pd.isna(val) for val in positions[i]):
                    logger.warning(f"NaN values detected in positions at index {i}, skipping")
                    speeds.append(0.0)
                    continue
                    
                frames_diff = frame_numbers[i] - frame_numbers[i-1]
                
                speed = self.calculate_speed(
                    positions[i-1],
                    positions[i],
                    frames_diff
                )
                
                # Update validation criteria
                if frames_diff >= self.min_frame_diff and speed > 0:
                    valid_count += 1
                
                speeds.append(speed)
            
            # Apply speed smoothing to reduce noise
            if len(speeds) > 3:
                speeds = SpeedDataFilter.smooth_speeds(speeds, window_size=3)
            
            logger.info(f"Processed {len(speeds)} speed calculations, {valid_count} valid measurements")
            return speeds
            
        except Exception as e:
            logger.error(f"Error calculating speed sequence: {str(e)}")
            return []

    def get_speed_statistics(self, speeds: List[float]) -> Dict[str, float]:
        """
        Calculate summary statistics for a sequence of speeds with improved filtering
        
        Args:
            speeds: List of calculated speeds in km/h
            
        Returns:
            Dict[str, float]: Dictionary containing speed statistics
        """
        try:
            # Apply outlier filtering before statistics
            filtered_speeds = SpeedDataFilter.filter_speed_outliers(speeds)
            non_zero_speeds = [s for s in filtered_speeds if s > 0]
            
            if not non_zero_speeds:
                return {
                    'average': round(0.0, self.decimal_places),
                    'maximum': round(0.0, self.decimal_places),
                    'sustained': round(0.0, self.decimal_places),
                    'min_frame_diff': self.min_frame_diff,
                    'valid_count': 0,
                    'pitch_length_meters': self.pitch_length_meters,
                    'meters_per_pixel': self.meters_per_pixel
                }
            
            avg_speed = round(np.mean(non_zero_speeds), self.decimal_places)
            max_speed = round(np.max(non_zero_speeds), self.decimal_places)
            sustained_speed = round(SpeedDataFilter.get_max_sustained_speed(non_zero_speeds), self.decimal_places)
            
            stats = {
                'average': avg_speed,
                'maximum': max_speed,
                'sustained': sustained_speed,
                'min_frame_diff': self.min_frame_diff,
                'valid_count': len(non_zero_speeds),
                'pitch_length_meters': self.pitch_length_meters,
                'meters_per_pixel': self.meters_per_pixel
            }
            
            logger.info(f"Speed statistics calculated: avg={avg_speed}, max={max_speed}, "
                       f"sustained={sustained_speed}, valid_count={len(non_zero_speeds)}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating speed statistics: {str(e)}")
            return {
                'average': round(0.0, self.decimal_places),
                'maximum': round(0.0, self.decimal_places),
                'sustained': round(0.0, self.decimal_places),
                'min_frame_diff': self.min_frame_diff,
                'valid_count': 0,
                'pitch_length_meters': self.pitch_length_meters,
                'meters_per_pixel': self.meters_per_pixel
            }

    def validate_speed(self, speed: float, max_speed: float = None) -> bool:
        """
        Validate if a calculated speed is physically possible
        
        Args:
            speed: Speed in km/h to validate
            max_speed: Maximum possible speed in km/h (uses fixed threshold if None)
            
        Returns:
            bool: True if speed is valid, False otherwise
        """
        try:
            if max_speed is None:
                max_speed = 125.0  # Fixed threshold
                
            if speed <= 0 or speed > max_speed:
                logger.warning(f"Invalid speed detected: {speed} km/h (threshold: {max_speed})")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating speed: {str(e)}")
            return False

    def calibrate(self, known_distance_pixels: int, known_distance_meters: float) -> bool:
        """
        Calibrate the speed calculator using a known distance
        
        Args:
            known_distance_pixels: Distance in pixels
            known_distance_meters: Corresponding distance in meters
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        try:
            if known_distance_pixels <= 0 or known_distance_meters <= 0:
                logger.error("Invalid calibration distances")
                return False
                
            new_meters_per_pixel = known_distance_meters / known_distance_pixels
            
            if new_meters_per_pixel <= 0:
                logger.error("Invalid calibration result")
                return False
                
            self.meters_per_pixel = new_meters_per_pixel
            logger.info(f"Calibration updated: {self.meters_per_pixel} meters/pixel")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during calibration: {str(e)}")
            return False

    def get_calibration_info(self) -> Dict[str, float]:
        """
        Get current calibration information
        
        Returns:
            Dict[str, float]: Dictionary containing calibration parameters
        """
        return {
            'pitch_length_pixels': self.pitch_length_pixels,
            'pitch_length_meters': self.pitch_length_meters,
            'meters_per_pixel': self.meters_per_pixel,
            'fps': self.fps,
            'frame_duration': self.frame_duration
        }

# ==============================================================================
# ROI COORDINATES CLASS
# ==============================================================================

@dataclass
class ROICoordinates:
    """Enhanced data class for Region of Interest coordinates with crease-focused detection"""
    left: int  # Now represents left crease boundary
    right: int  # Now represents right crease boundary
    top: int
    bottom: int
    left_wicket_x: int
    right_wicket_x: int
    # Crease coordinates (now same as main ROI)
    left_crease_x: int
    right_crease_x: int
    crease_top_y: int
    crease_bottom_y: int

    def is_point_inside(self, x: int, y: int) -> bool:
        """Check if a point is inside the ROI (now crease-to-crease region only)"""
        return (self.left_crease_x <= x <= self.right_crease_x and 
                self.crease_top_y <= y <= self.crease_bottom_y)
    
    def is_point_inside_crease_region(self, x: int, y: int) -> bool:
        """Check if a point is inside the crease-to-crease region (same as main ROI now)"""
        return (self.left_crease_x <= x <= self.right_crease_x and 
                self.crease_top_y <= y <= self.crease_bottom_y)

# ==============================================================================
# SPEED DATA FILTER CLASS
# ==============================================================================

class SpeedDataFilter:
    """Class to handle speed data filtering and smoothing with improved algorithms"""
    
    MAX_CRICKET_BALL_SPEED = 125.0  # Fixed threshold at 125 km/h
    
    @staticmethod
    def filter_speed_outliers(speeds: List[float], method: str = 'iqr') -> List[float]:
        """
        Filter outliers from speed data using IQR method
        
        Args:
            speeds (List[float]): List of calculated speeds
            method (str): Method to use ('iqr' or 'zscore')
            
        Returns:
            List[float]: Filtered speeds with outliers removed
        """
        if not speeds:
            return []
            
        try:
            max_threshold = 125.0  # Fixed threshold
            speeds_array = np.array([s for s in speeds if s > 0])  # Remove zero speeds
            
            if len(speeds_array) < 4:  # Need at least 4 points for IQR
                return [s for s in speeds_array if s <= max_threshold]
            
            if method == 'iqr':
                # Use IQR method for better outlier detection
                Q1 = np.percentile(speeds_array, 25)
                Q3 = np.percentile(speeds_array, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Also apply fixed speed threshold
                upper_bound = min(upper_bound, max_threshold)
                
                filtered_speeds = speeds_array[
                    (speeds_array >= lower_bound) & 
                    (speeds_array <= upper_bound)
                ]
            else:
                # Z-score method
                z_scores = np.abs((speeds_array - np.mean(speeds_array)) / np.std(speeds_array))
                filtered_speeds = speeds_array[
                    (z_scores < 2.0) & 
                    (speeds_array <= max_threshold)
                ]
            
            logger.info(f"Filtered {len(speeds_array) - len(filtered_speeds)} outliers from {len(speeds_array)} speeds (threshold: {max_threshold} km/h)")
            return filtered_speeds.tolist()
            
        except Exception as e:
            logger.error(f"Error filtering speed outliers: {str(e)}")
            max_threshold = 125.0  # Fixed threshold
            return [s for s in speeds if 0 < s <= max_threshold]
    
    @staticmethod
    def remove_speed_outliers(speeds: List[float], 
                            frame_numbers: List[int],
                            z_threshold: float = 2.0) -> Tuple[List[float], List[int]]:
        """
        Remove statistical outliers from speed data (legacy method for compatibility)
        
        Args:
            speeds (List[float]): List of calculated speeds
            frame_numbers (List[int]): Corresponding frame numbers
            z_threshold (float): Z-score threshold for outlier detection
            
        Returns:
            Tuple[List[float], List[int]]: Filtered speeds and frame numbers
        """
        if not speeds:
            return [], []
            
        try:
            # Use the new filter method
            filtered_speeds_only = SpeedDataFilter.filter_speed_outliers(speeds)
            
            # Map back to frame numbers
            filtered_speeds = []
            filtered_frames = []
            
            for i, speed in enumerate(speeds):
                if speed in filtered_speeds_only or speed == 0:  # Keep zero speeds for continuity
                    filtered_speeds.append(speed)
                    if i < len(frame_numbers):
                        filtered_frames.append(frame_numbers[i])
            
            return filtered_speeds, filtered_frames
            
        except Exception as e:
            logger.error(f"Error filtering speed outliers: {str(e)}")
            return speeds, frame_numbers

    @staticmethod
    def smooth_speeds(speeds: List[float], 
                     window_size: int = 3) -> List[float]:
        """
        Apply moving average smoothing to speed data with improved edge handling
        
        Args:
            speeds (List[float]): List of speeds to smooth
            window_size (int): Size of moving average window
            
        Returns:
            List[float]: Smoothed speed values
        """
        if not speeds or window_size < 2:
            return speeds
            
        try:
            speeds_array = np.array(speeds)
            
            # Improved smoothing with better edge handling
            if len(speeds_array) <= window_size:
                # For short sequences, use simple averaging
                return speeds
            
            # Apply moving average
            smoothed = []
            half_window = window_size // 2
            
            for i in range(len(speeds_array)):
                if speeds_array[i] == 0:  # Don't smooth zero values
                    smoothed.append(0.0)
                    continue
                
                # Define window bounds
                start_idx = max(0, i - half_window)
                end_idx = min(len(speeds_array), i + half_window + 1)
                
                # Get non-zero values in window
                window_values = speeds_array[start_idx:end_idx]
                non_zero_values = window_values[window_values > 0]
                
                if len(non_zero_values) > 0:
                    smoothed.append(float(np.mean(non_zero_values)))
                else:
                    smoothed.append(speeds_array[i])
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error smoothing speeds: {str(e)}")
            return speeds

    @staticmethod
    def get_max_sustained_speed(speeds: List[float], 
                              min_duration: int = 3) -> float:
        """
        Calculate maximum sustained speed over minimum duration with improved algorithm
        
        Args:
            speeds (List[float]): List of speeds
            min_duration (int): Minimum number of consecutive frames
            
        Returns:
            float: Maximum sustained speed
        """
        if not speeds or len(speeds) < min_duration:
            return 0.0
            
        try:
            # Improved sustained speed calculation
            max_sustained = 0.0
            
            # Filter out zero speeds for sustained calculation
            non_zero_speeds = [s for s in speeds if s > 0]
            
            if len(non_zero_speeds) < min_duration:
                return 0.0
            
            # Calculate rolling average for all possible windows
            for i in range(len(non_zero_speeds) - min_duration + 1):
                window = non_zero_speeds[i:i + min_duration]
                avg_speed = sum(window) / len(window)
                max_sustained = max(max_sustained, avg_speed)
            
            return round(max_sustained, 3)
            
        except Exception as e:
            logger.error(f"Error calculating sustained speed: {str(e)}")
            return 0.0

# ==============================================================================
# CRICKET VIDEO ANALYZER CLASS
# ==============================================================================
        
class CricketVideoAnalyzer:
    """Enhanced main class for cricket video analysis with YOLO model integration and crease detection"""
    
    def __init__(self, model_path: str):
        """Initialize the analyzer with model path"""
        try:
            self.model, _ = load_yolo_model(model_path)
            self.meters_per_pixel = None
            # Cricket specifications
            self.POPPING_CREASE_DISTANCE = 1.22  # 4 feet in meters
            logger.info("Enhanced model loaded successfully with crease detection")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def calculate_meters_per_pixel_from_wickets(self, left_wicket_x: int, right_wicket_x: int) -> float:
        """
        Calculate the meters per pixel conversion factor using wicket distance
        
        Args:
            left_wicket_x: X coordinate of left wicket
            right_wicket_x: X coordinate of right wicket
            
        Returns:
            float: Conversion factor (meters/pixel)
        """
        try:
            pixel_distance = right_wicket_x - left_wicket_x
            if pixel_distance <= 0:
                raise ValueError("Invalid wicket positions detected")
            
            pitch_length = st.session_state.pitch_length_meters
            meters_per_pixel = pitch_length / pixel_distance
            logger.info(f"Calibration: {meters_per_pixel:.6f} meters/pixel using wicket distance")
            
            return meters_per_pixel
            
        except Exception as e:
            logger.error(f"Error calculating meters per pixel from wickets: {str(e)}")
            raise

    @staticmethod
    def _draw_crease_focused_roi_visualization(
        frame: np.ndarray,
        roi_coords: ROICoordinates,
        wicket_boxes: List[Tuple]
    ) -> None:
        """Draw crease-focused ROI visualization with wicket-to-crease connection lines"""
        
        # Draw main ROI (now crease-to-crease region) with thick green border
        cv2.rectangle(
            frame,
            (roi_coords.left_crease_x, roi_coords.crease_top_y),
            (roi_coords.right_crease_x, roi_coords.crease_bottom_y),
            (0, 255, 0), 4  # Thick green border for main ROI
        )
        
        # Draw wicket lines (blue) - only within ROI area, no extensions
        for wicket_x in [roi_coords.left_wicket_x, roi_coords.right_wicket_x]:
            # Draw wicket lines only within the crease area, no extensions
            cv2.line(
                frame,
                (wicket_x, roi_coords.crease_top_y),
                (wicket_x, roi_coords.crease_bottom_y),
                (255, 0, 0), 2
            )
        
        # Draw crease lines (red) - same length as before
        crease_line_thickness = 3
        
        # Left crease line (red)
        cv2.line(
            frame,
            (roi_coords.left_crease_x, roi_coords.crease_top_y),
            (roi_coords.left_crease_x, roi_coords.crease_bottom_y),
            (0, 0, 255), crease_line_thickness
        )
        
        # Right crease line (red)
        cv2.line(
            frame,
            (roi_coords.right_crease_x, roi_coords.crease_top_y),
            (roi_coords.right_crease_x, roi_coords.crease_bottom_y),
            (0, 0, 255), crease_line_thickness
        )
        
        # Draw black connecting lines from wickets to creases
        connection_line_thickness = 2
        
        # Left side: Connect left wicket to left crease
        cv2.line(
            frame,
            (roi_coords.left_wicket_x, roi_coords.crease_top_y),
            (roi_coords.left_crease_x, roi_coords.crease_top_y),
            (0, 0, 0), connection_line_thickness  # Black line
        )
        cv2.line(
            frame,
            (roi_coords.left_wicket_x, roi_coords.crease_bottom_y),
            (roi_coords.left_crease_x, roi_coords.crease_bottom_y),
            (0, 0, 0), connection_line_thickness  # Black line
        )
        
        # Right side: Connect right wicket to right crease
        cv2.line(
            frame,
            (roi_coords.right_wicket_x, roi_coords.crease_top_y),
            (roi_coords.right_crease_x, roi_coords.crease_top_y),
            (0, 0, 0), connection_line_thickness  # Black line
        )
        cv2.line(
            frame,
            (roi_coords.right_wicket_x, roi_coords.crease_bottom_y),
            (roi_coords.right_crease_x, roi_coords.crease_bottom_y),
            (0, 0, 0), connection_line_thickness  # Black line
        )
        
        # Draw wicket boxes (yellow rectangles for better visibility)
        for box in wicket_boxes:
            cv2.rectangle(
                frame,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 255), 2  # Yellow for wickets
            )
            cv2.circle(
                frame,
                (box[4], box[5]),
                5, (0, 255, 255), -1  # Yellow centers
            )

    def calculate_crease_positions(self, left_wicket_x: int, right_wicket_x: int, 
                                 wicket_height: int, center_y: int) -> tuple:
        """
        Calculate crease positions based on wicket locations
        
        Args:
            left_wicket_x: X coordinate of left wicket
            right_wicket_x: X coordinate of right wicket
            wicket_height: Height of wicket boxes
            center_y: Center Y coordinate of wickets
            
        Returns:
            tuple: (left_crease_x, right_crease_x, crease_top_y, crease_bottom_y)
        """
        try:
            # Calculate pitch length in pixels
            pitch_length_pixels = right_wicket_x - left_wicket_x
            
            # Get meters per pixel conversion
            pitch_length_meters = st.session_state.pitch_length_meters
            meters_per_pixel = pitch_length_meters / pitch_length_pixels
            
            # Calculate crease distance in pixels
            crease_distance_pixels = int(self.POPPING_CREASE_DISTANCE / meters_per_pixel)
            
            # Calculate crease positions (4 feet in front of wickets)
            left_crease_x = left_wicket_x + crease_distance_pixels
            right_crease_x = right_wicket_x - crease_distance_pixels
            
            # Ensure creases don't overlap
            if left_crease_x >= right_crease_x:
                # If pitch is too short, place creases at 1/4 and 3/4 of pitch length
                left_crease_x = left_wicket_x + pitch_length_pixels // 4
                right_crease_x = right_wicket_x - pitch_length_pixels // 4
            
            # Calculate crease line length (same as wicket visualization)
            crease_line_length = int(wicket_height * 2.4)  # Same as ROI height calculation
            crease_top_y = center_y - crease_line_length
            crease_bottom_y = center_y + crease_line_length
            
            logger.info(f"Crease positions calculated: left={left_crease_x}, right={right_crease_x}")
            logger.info(f"Crease distance: {crease_distance_pixels} pixels ({self.POPPING_CREASE_DISTANCE}m)")
            
            return left_crease_x, right_crease_x, crease_top_y, crease_bottom_y
            
        except Exception as e:
            logger.error(f"Error calculating crease positions: {str(e)}")
            # Fallback to simple calculation
            pitch_quarter = (right_wicket_x - left_wicket_x) // 4
            return (left_wicket_x + pitch_quarter, 
                   right_wicket_x - pitch_quarter,
                   center_y - wicket_height,
                   center_y + wicket_height)

    def create_roi_between_wickets(self, frame: np.ndarray) -> Tuple[Optional[ROICoordinates], np.ndarray]:
        """Enhanced wicket detection with crease-focused ROI"""
        try:
            results = self.model(frame, conf=0.25)
            frame_with_roi = frame.copy()
            wicket_boxes = []
            
            # Detect wickets
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # Wicket class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        wicket_boxes.append((x1, y1, x2, y2, center_x, center_y))
            
            if len(wicket_boxes) < 2:
                logger.warning("Not enough wickets detected for crease calculation")
                return None, frame_with_roi
                
            # Sort wickets by x-coordinate
            wicket_boxes.sort(key=lambda x: x[0])
            left_wicket = wicket_boxes[0]
            right_wicket = wicket_boxes[-1]
            
            # Calculate basic ROI dimensions
            wicket_height = max(
                left_wicket[3] - left_wicket[1],
                right_wicket[3] - right_wicket[1]
            )
            extra_height = int(wicket_height * 2.4)
            center_y = (min(left_wicket[1], right_wicket[1]) + 
                       max(left_wicket[3], right_wicket[3])) // 2
            
            # Calculate crease positions
            left_crease_x, right_crease_x, crease_top_y, crease_bottom_y = self.calculate_crease_positions(
                left_wicket[4], right_wicket[4], wicket_height, center_y
            )
            
            # Create ROI coordinates with crease-focused detection
            # Main ROI is now the crease-to-crease region
            roi_coords = ROICoordinates(
                left=left_crease_x,  # Main ROI left boundary is now crease
                right=right_crease_x,  # Main ROI right boundary is now crease
                top=max(0, crease_top_y),  # Main ROI top is crease top
                bottom=min(frame.shape[0], crease_bottom_y),  # Main ROI bottom is crease bottom
                left_wicket_x=left_wicket[4],
                right_wicket_x=right_wicket[4],
                left_crease_x=left_crease_x,
                right_crease_x=right_crease_x,
                crease_top_y=max(0, crease_top_y),
                crease_bottom_y=min(frame.shape[0], crease_bottom_y)
            )
            
            # Calculate meters per pixel for speed analysis
            self.meters_per_pixel = self.calculate_meters_per_pixel_from_wickets(
                left_wicket[4], right_wicket[4]
            )
            
            # Draw enhanced visualization with crease-focused ROI
            self._draw_crease_focused_roi_visualization(
                frame_with_roi, roi_coords, wicket_boxes
            )
            
            return roi_coords, frame_with_roi
            
        except Exception as e:
            logger.error(f"Error in crease-focused ROI creation: {str(e)}")
            return None, frame

    @staticmethod
    def _draw_enhanced_roi_visualization(
        frame: np.ndarray,
        roi_coords: ROICoordinates,
        wicket_boxes: List[Tuple]
    ) -> None:
        """Legacy method - redirects to crease-focused visualization"""
        CricketVideoAnalyzer._draw_crease_focused_roi_visualization(frame, roi_coords, wicket_boxes)

# ==============================================================================
# BALL TRACKER CLASS
# ==============================================================================

class BallTracker:
    """Kalman filter-based ball tracking implementation"""
    
    def __init__(self):
        """Initialize the Kalman filter for ball tracking"""
        self.kf = self._setup_kalman_filter()
        self.initialized = False
        self.tracked_positions = []
        self.smoothed_positions = []
    
    @staticmethod
    def _setup_kalman_filter() -> KalmanFilter:
        """Set up Kalman filter parameters"""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        kf.R = np.eye(2) * 50
        kf.Q = np.eye(4) * 0.1
        kf.P *= 1000
        
        return kf

    def update(self, measurement: np.ndarray) -> Tuple[float, float]:
        """Update tracking with new measurement"""
        
        if any(pd.isna(val) for val in measurement):
            logger.warning(f"NaN values detected in measurement: {measurement}")
            if self.initialized:
                return self.smoothed_positions[-1][0], self.smoothed_positions[-1][1]
            else:
                return 0.0, 0.0
                
        if not self.initialized:
            self._initialize_tracking(measurement)
            return measurement[0], measurement[1]
        
        self.kf.predict()
        self.kf.update(measurement)
        
        self.tracked_positions.append((measurement[0], measurement[1]))
        self.smoothed_positions.append((self.kf.x[0], self.kf.x[1]))
        
        return self.kf.x[0], self.kf.x[1]

    def _initialize_tracking(self, measurement: np.ndarray) -> None:
        """Initialize tracking with first measurement"""
        self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
        self.initialized = True
        self.tracked_positions.append((measurement[0], measurement[1]))
        self.smoothed_positions.append((measurement[0], measurement[1]))

# ==============================================================================
# VIDEO PROCESSING FUNCTIONS
# ==============================================================================

def get_video_info(video_path):
    """Get video FPS and frame duration"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1/fps if fps > 0 else 0
        cap.release()
        
        return {
            "fps": f"{fps:.2f}",
            "frame_duration": f"{frame_duration:.4f}"  
        }
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        raise

# ==============================================================================
# TRAJECTORY VISUALIZATION
# ==============================================================================

def create_trajectory_visualization(detections, positions, roi):
    """
    Create professional trajectory visualization focused on crease-to-crease region
    
    Enhanced with crease-focused detection for better analysis.
    """
    try:
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='#f8f8f8')
        ax.set_facecolor('#f8f8f8')
        
        # Draw main ROI (crease-to-crease region) with thick green border
        rect = plt.Rectangle(
            (roi.left_crease_x, roi.crease_top_y), 
            roi.right_crease_x - roi.left_crease_x, 
            roi.crease_bottom_y - roi.crease_top_y, 
            linewidth=4, 
            edgecolor='green', 
            facecolor='rgba(0, 255, 0, 0.1)', 
            alpha=0.8, 
            linestyle='-',
            label='Detection ROI (Crease to Crease)'
        )
        ax.add_patch(rect)
        
        # Draw wicket lines (within ROI only)
        for wicket_name, wicket_x in [("Bowler's Wicket", roi.left_wicket_x), ("Batsman's Wicket", roi.right_wicket_x)]:
            ax.plot(
                [wicket_x, wicket_x],
                [roi.crease_top_y, roi.crease_bottom_y],
                color='blue', 
                linestyle='--', 
                alpha=0.7, 
                linewidth=2,
                label=wicket_name if wicket_x == roi.left_wicket_x else None
            )
        
        # Draw crease lines (ROI boundaries)
        for crease_name, crease_x in [("Bowler's Crease", roi.left_crease_x), ("Batsman's Crease", roi.right_crease_x)]:
            ax.axvline(
                x=crease_x, 
                color='red', 
                linestyle='-', 
                alpha=0.9, 
                linewidth=3,
                label=crease_name if crease_x == roi.left_crease_x else None
            )
        
        # Draw wicket-to-crease connection lines (black)
        for side, (wicket_x, crease_x) in [("Left", (roi.left_wicket_x, roi.left_crease_x)), 
                                          ("Right", (roi.right_wicket_x, roi.right_crease_x))]:
            ax.axhline(
                y=roi.crease_top_y, 
                xmin=(wicket_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                xmax=(crease_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                color='black', 
                linestyle='-', 
                alpha=0.8, 
                linewidth=2,
                label=f'Wicket-Crease Connection (4 feet)' if side == "Left" else None
            )
            ax.axhline(
                y=roi.crease_bottom_y, 
                xmin=(wicket_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                xmax=(crease_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                color='black', 
                linestyle='-', 
                alpha=0.8, 
                linewidth=2
            )

        if raw_positions := [d['position'] for d in detections]:
            x_coords = [p[0] for p in raw_positions]
            y_coords = [p[1] for p in raw_positions]
            
            valid_indices = [i for i, (x, y) in enumerate(zip(x_coords, y_coords)) if not pd.isna(x) and not pd.isna(y)]
            
            if valid_indices:
                x_coords = [x_coords[i] for i in valid_indices]
                y_coords = [y_coords[i] for i in valid_indices]
                
                # All detections are now in crease region, so use consistent styling
                cmap = plt.cm.plasma
                colors = cmap(np.linspace(0, 1, len(x_coords)))
                
                for i in range(len(x_coords)-1):
                    ax.plot(
                        [x_coords[i], x_coords[i+1]], 
                        [y_coords[i], y_coords[i+1]], 
                        color=colors[i], 
                        alpha=0.7, 
                        linewidth=2
                    )
                
                # Detect direction changes (potential bat hits)
                bat_hit_segments = []
                current_segment = []
                is_decreasing = False
                
                for i in range(1, len(x_coords)):
                    if x_coords[i] < x_coords[i-1]:  
                        if not is_decreasing:  
                            current_segment = [(x_coords[i-1], y_coords[i-1])]
                            is_decreasing = True
                        current_segment.append((x_coords[i], y_coords[i]))
                    else:
                        if is_decreasing and len(current_segment) > 1:  
                            bat_hit_segments.append(current_segment)
                        is_decreasing = False
                
                if is_decreasing and len(current_segment) > 1:  
                    bat_hit_segments.append(current_segment)
                
                scatter = ax.scatter(
                    x_coords, 
                    y_coords, 
                    c=range(len(x_coords)), 
                    cmap='plasma',
                    alpha=0.8, 
                    s=60, 
                    edgecolors='white',
                    linewidths=1,
                    label='Ball Positions (Crease Region)'
                )
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Time Sequence', fontsize=12)
                
                # Highlight bat hits
                for segment in bat_hit_segments:
                    seg_x = [p[0] for p in segment]
                    seg_y = [p[1] for p in segment]
                    
                    ax.plot(
                        seg_x, 
                        seg_y, 
                        'orange', 
                        linewidth=6,  
                        alpha=0.9,    
                        solid_capstyle='round',
                        label='Bat Hit' if segment == bat_hit_segments[0] else None
                    )
                    
                    ax.scatter(
                        seg_x[0], 
                        seg_y[0], 
                        c='orange', 
                        s=300,     
                        marker='*', 
                        edgecolors='black',
                        linewidths=2,
                        zorder=10
                    )
        
        if positions:
            valid_positions = [(x, y) for x, y in positions if not pd.isna(x) and not pd.isna(y)]
            
            if valid_positions:
                x_smooth = [p[0] for p in valid_positions]
                y_smooth = [p[1] for p in valid_positions]
                
                points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                norm = plt.Normalize(0, len(segments))
                lc = LineCollection(
                    segments, 
                    cmap='viridis', 
                    norm=norm,
                    linewidth=5,      
                    alpha=0.9,        
                    capstyle='round'  
                )
                lc.set_array(np.arange(len(segments)))
                line = ax.add_collection(lc)
                
                ax.scatter(
                    x_smooth[0], 
                    y_smooth[0], 
                    s=200, 
                    marker='o', 
                    color='darkgreen', 
                    edgecolor='white', 
                    linewidth=3, 
                    label='Start Position',
                    zorder=10
                )
                
                ax.scatter(
                    x_smooth[-1], 
                    y_smooth[-1], 
                    s=200, 
                    marker='s', 
                    color='darkred', 
                    edgecolor='white', 
                    linewidth=3, 
                    label='End Position',
                    zorder=10
                )

        # Enhanced legend box
        legend_box = ax.text(
            0.02, 0.98,
            'CRICKET BALL ANALYSIS\n\n' +
            '• GREEN BOX: Detection ROI (Crease to Crease)\n' +
            '• RED LINES: Crease boundaries (4 feet from wickets)\n' +
            '• BLUE LINES: Wicket positions\n' +
            '• BLACK LINES: Connection lines (4 feet)\n' +
            '• COLORED DOTS: Ball positions over time\n' +
            '• ORANGE SEGMENTS: Potential bat hits\n' +
            '• DETECTION ZONE: 4 feet from each wicket',
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor='white', 
                alpha=0.95,
                edgecolor='green',
                linewidth=2
            ),
            verticalalignment='top',
            fontsize=11,
            fontweight='bold'
        )
        
        # Set limits to focus on crease region with some padding
        padding = 100
        ax.set_xlim(roi.left_crease_x - padding, roi.right_crease_x + padding)
        ax.set_ylim(roi.crease_bottom_y + padding, roi.crease_top_y - padding)
        
        ax.set_xlabel('X Position (pixels)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Position (pixels)', fontsize=14, fontweight='bold')
        ax.set_title('Cricket Ball Trajectory Analysis\n(Detection Zone: 4 feet from wickets)', 
                    pad=20, fontsize=16, fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Position legend to avoid overlapping with data
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')
        
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(
            0.98, 0.02, 
            f"Cricket Ball Analysis: {time_now}", 
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=9, color='gray',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', edgecolor='none')
        )
        
        fig.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"Error creating crease-focused trajectory visualization: {str(e)}")
        raise

# ==============================================================================
# RTSP STREAMING FUNCTIONS
# ==============================================================================

def open_rtsp_stream(rtsp_url):
    """Opens an RTSP stream connection and returns the video capture object"""
    try:
        cv2.setUseOptimized(True)
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  
        
        if not cap.isOpened():
            raise ValueError(f"Failed to connect to RTSP stream: {rtsp_url}")
        
        logger.info(f"Successfully connected to RTSP stream: {rtsp_url}")
        return cap
    
    except Exception as e:
        logger.error(f"Error opening RTSP stream: {str(e)}")
        raise

def get_rtsp_frame(cap):
    """Get a single frame from the RTSP stream"""
    try:
        ret, frame = cap.read()
        
        if not ret:
            logger.warning("Failed to read frame from RTSP stream")
            return None
        
        return frame
    
    except Exception as e:
        logger.error(f"Error reading RTSP frame: {str(e)}")
        return None

def create_temp_video_from_rtsp(rtsp_url, duration=5, output_path=None):
    """
    Capture frames from RTSP stream and create a temporary video file
    """
    try:
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                output_path = temp_file.name
        
        cap = open_rtsp_stream(rtsp_url)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps <= 0:
            fps = 25.0
            logger.warning(f"Invalid FPS detected, using default value: {fps}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        num_frames = int(fps * duration)
        
        frames_captured = 0
        start_time = time.time()
        
        progress_text = st.empty()
        progress_text.info(f"Capturing {duration} seconds of video from RTSP stream...")
        
        progress_bar = st.progress(0)
        
        while frames_captured < num_frames:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            out.write(frame)
            frames_captured += 1
            
            progress = frames_captured / num_frames
            progress_bar.progress(progress)
            progress_text.info(f"Capturing video: {frames_captured}/{num_frames} frames ({progress*100:.1f}%)")
            
            elapsed_time = time.time() - start_time
            if elapsed_time > (duration * 1.2):
                logger.warning(f"Capture time exceeded expected duration by 20%, stopping at {frames_captured} frames")
                break
        
        cap.release()
        out.release()
        
        progress_text.empty()
        progress_bar.empty()
        
        logger.info(f"Temporary video created at {output_path} with {frames_captured} frames")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating temporary video from RTSP: {str(e)}")
        raise



def main():
    """Main application entry point"""
    
    
    if 'ball_detections' not in st.session_state:
        st.session_state.ball_detections = []
        
    if 'ball_tracker' not in st.session_state:
        st.session_state.ball_tracker = BallTracker()
        st.session_state.all_smoothed_positions = []
    
    if 'rtsp_url' not in st.session_state:
        st.session_state.rtsp_url = "rtsp://admin:admin123@172.16.200.1:554/cam/realmonitor?channel=1&subtype=0"
    
    
    if 'max_speed_threshold' not in st.session_state:
        st.session_state.max_speed_threshold = 125.0
        logger.info("Speed automatically set to km/h")
        
    def check_device():
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.sidebar.info(f"🖥️ Running on: {device.upper()}")
        if device == "cuda":
            st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
        return device
        
    
    if 'analyzer' not in st.session_state:
        try:
            
            model_path = r"./best3.pt"
            st.session_state.analyzer = CricketVideoAnalyzer(model_path)
            logger.info("Enhanced Cricket Video Analyzer with crease detection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize analyzer: {str(e)}")
            st.error("Failed to initialize the video analyzer. Please check the model file path.")
            return
    
    
    st.title("🏏 Cricket Ball Speed Analysis")
    st.markdown("""
    ### Professional Cricket Video Analysis Tool
    **Key Features:**
    - **Corrected FPS Processing**: All FPS values supported
    - **Improved Speed Calculation**: Enhanced accuracy with validation
    - **Crease Detection**: 4 feet from wickets
    - **Speed Statistics**: Outlier filtering and smoothing
    """)
    
    check_device()
    
    
    with st.sidebar:
        st.header("Configuration")
        
        input_source = st.radio(
            "Select Video Input Source",
            ["Upload Video File", "Dahua Camera (RTSP)"],
            help="Choose between uploading a video file or connecting to a Dahua camera via RTSP"
        )
        
        video_path = None
        
        if input_source == "Upload Video File":
            video_file = st.file_uploader(
                "Upload Cricket Video",
                type=['mp4', 'avi', 'mov'],
                help="Upload a video file for analysis"
            )
            
            if video_file:
                file_details = {
                    "Filename": video_file.name,
                    "FileType": video_file.type,
                    "FileSize": f"{video_file.size / 1024:.2f} KB"
                }
                st.json(file_details)
                
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(video_file.read())
                        video_path = tfile.name
                        logger.info(f"Video saved temporarily at: {video_path}")
                except Exception as e:
                    logger.error(f"Error saving video file: {str(e)}")
                    st.error("Failed to process the uploaded video file.")
        else:
            rtsp_url = st.text_input(
                "Dahua Camera RTSP URL",
                value=st.session_state.rtsp_url,
                help="Enter the RTSP URL for your Dahua camera"
            )
            
            if rtsp_url != st.session_state.rtsp_url:
                st.session_state.rtsp_url = rtsp_url
            
            capture_duration = st.slider(
                "Capture Duration (seconds)",
                min_value=5,
                max_value=30,
                value=10,
                step=5,
                help="Duration in seconds to capture from the RTSP stream"
            )
            
            if st.button("Test Camera Connection"):
                with st.spinner("Testing connection to camera..."):
                    try:
                        cap = open_rtsp_stream(rtsp_url)
                        ret, frame = cap.read()
                        if ret:
                            st.success("✅ Successfully connected to camera!")
                            st.image(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                caption="Camera Preview",
                                width=300
                            )
                            cap.release()
                        else:
                            st.error("❌ Connected to camera but failed to read frame")
                    except Exception as e:
                        st.error(f"❌ Failed to connect to camera: {str(e)}")
            
            if st.button("Capture Video from Camera", type="primary"):
                with st.spinner(f"Capturing {capture_duration} seconds of video from camera..."):
                    try:
                        video_path = create_temp_video_from_rtsp(rtsp_url, duration=capture_duration)
                        st.success(f"✅ Successfully captured {capture_duration} seconds of video!")
                        
                        video_info = get_video_info(video_path)
                        st.json({
                            "Captured Video": {
                                "Duration": f"{capture_duration} seconds",
                                "FPS": video_info["fps"],
                                "Frame Duration": video_info["frame_duration"],
                                "Temporary File": video_path
                            }
                        })
                    except Exception as e:
                        st.error(f"❌ Failed to capture video: {str(e)}")
                        video_path = None
    
    
    if video_path is not None:
        tab0, tab1, tab2, tab3 = st.tabs([
            "⚙️ Pitch Configuration",
            "🎯 ROI Detection", 
            "🏏 Ball Detection",
            "📊 Speed Analysis"
        ])
        
        
        with tab0:
            st.header("Pitch Configuration")
            st.markdown("""
            Set the actual pitch length for accurate speed calculations and crease detection. 
            Standard cricket pitch length is 20.12 meters (22 yards).
            """)
            
            pitch_length = st.number_input(
                "Enter Pitch Length (meters)",
                min_value=10.0,
                max_value=30.0,
                value=20.12,
                step=0.01,
                help="Standard cricket pitch length is 20.12 meters (22 yards)"
            )
            
            if 'pitch_length_meters' not in st.session_state or st.session_state.pitch_length_meters != pitch_length:
                st.session_state.pitch_length_meters = pitch_length
                logger.info(f"Pitch length set to {pitch_length} meters")
            
            st.subheader("Current Configuration")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Pitch Length:** {pitch_length} meters")
            

            if st.button("Confirm Configuration", type="primary"):
                st.success(f"✅ Configuration saved! Pitch: {pitch_length}m, Proceed to ROI Detection.")

        
        with tab1:
            try:
                if 'pitch_length_meters' not in st.session_state:
                    st.warning("⚠️ Please configure the pitch length in the Pitch Configuration tab first")
                    return
                    
                st.header("ROI Detection")
                st.info("🎯 ROI Detection")
                
                video_info = get_video_info(video_path)
                fps = float(video_info["fps"])
                frame_duration = float(video_info['frame_duration'])
                
                cap = cv2.VideoCapture(video_path)
                ret, first_frame = cap.read()
                cap.release()
                
                if not ret:
                    raise ValueError("Failed to read video frame")
                
                with st.spinner("Detecting wickets and setting ROI..."):
                    roi_coords, processed_frame = st.session_state.analyzer.create_roi_between_wickets(first_frame)
                    
                    if not roi_coords:
                        st.warning("Could not detect wickets in the frame. Please ensure the video shows both wickets clearly.")
                        return
                    
                    st.session_state.roi_frame = processed_frame
                    st.session_state.roi_coordinates = roi_coords
                    
                    pitch_length_pixels = roi_coords.right_wicket_x - roi_coords.left_wicket_x
                    st.session_state.pitch_length_pixels = pitch_length_pixels
                    
                    meters_per_pixel = st.session_state.pitch_length_meters / pitch_length_pixels
                    st.session_state.meters_per_pixel = round(meters_per_pixel, 6)
                    
                    crease_to_crease_pixels = roi_coords.right_crease_x - roi_coords.left_crease_x
                
                st.image(
                    cv2.cvtColor(st.session_state.roi_frame, cv2.COLOR_BGR2RGB),
                    caption="Crease-Focused Detection Zone",
                    width=None
                )
                
                st.subheader("Detection Parameters")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Crease to Crease", f"{crease_to_crease_pixels} pixels")
                with col2:
                    st.metric("Meters per Pixel", f"{st.session_state.meters_per_pixel:.6f}")
                with col3:
                    st.metric("FPS", f"{fps:.2f}")
                with col4:
                    st.metric("Frame Duration", f"{frame_duration:.6f} sec")
                
                st.success("✅ ROI Detection Completed Successfully")
                
            except Exception as e:
                logger.error(f"Error in ROI detection: {str(e)}")
                st.error("An error occurred during ROI detection. Please try again.")
                return
        
        
        with tab2:
            try:
                st.header("Ball Detection")
                st.info("🎯 Ball detection in crease-to-crease region")
                    
                if not all(key in st.session_state for key in ['roi_frame', 'roi_coordinates']):
                    st.warning("⚠️ Please detect the ROI in the previous tab first")
                    return
                        
                st.image(
                    cv2.cvtColor(st.session_state.roi_frame, cv2.COLOR_BGR2RGB),
                    caption="Crease-Focused Detection Zone",
                    width=None  
                )
                    
                with st.expander("Detection Settings"):
                    col1, col2 = st.columns(2)
                    with col1:
                        confidence_threshold = st.slider(
                            "Detection Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.25,
                            step=0.05
                        )
                    with col2:
                        performance_window = st.number_input(
                            "Performance Window (frames)",
                            min_value=10,
                            max_value=100,
                            value=30,
                            help="Number of frames to calculate rolling performance metrics"
                        )
                    
                if st.button("🎯 Start Ball Detection", type="primary"):
                    
                    progress_bar = st.progress(0)
                    metrics_container = st.container()
                    
                    with metrics_container:
                        col1, col2, col3 = st.columns(3)
                        progress_text = col1.empty()
                        fps_metric = col2.empty()
                        detection_count = col3.empty()
                    
                    frame_placeholder = st.empty()
                    performance_placeholder = st.empty()
                        
                    cap = cv2.VideoCapture(video_path)
                    frame_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                        
                    processing_times = []
                    detection_counts = []
                    start_time = time.time()
                        
                    st.session_state.ball_detections = []
                    st.session_state.all_smoothed_positions = []
                    st.session_state.ball_tracker = BallTracker()
                    
                    try:
                        while cap.isOpened():
                            frame_start_time = time.time()
                            
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            roi = st.session_state.roi_coordinates
                            frame_with_detections = frame.copy()
                                
                            
                            cv2.rectangle(
                                frame_with_detections,
                                (roi.left_crease_x, roi.crease_top_y),
                                (roi.right_crease_x, roi.crease_bottom_y),
                                (0, 255, 0), 4  
                            )
                                
                            
                            cv2.putText(
                                frame_with_detections,
                                f"BALL DETECTION",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )
                            
                            
                            for wicket_x in [roi.left_wicket_x, roi.right_wicket_x]:
                                cv2.line(
                                    frame_with_detections,
                                    (wicket_x, roi.crease_top_y),
                                    (wicket_x, roi.crease_bottom_y),
                                    (255, 0, 0), 2
                                )
                            
                            
                            for crease_x in [roi.left_crease_x, roi.right_crease_x]:
                                cv2.line(
                                    frame_with_detections,
                                    (crease_x, roi.crease_top_y),
                                    (crease_x, roi.crease_bottom_y),
                                    (0, 0, 255), 3
                                )
                            
                            
                            cv2.line(
                                frame_with_detections,
                                (roi.left_wicket_x, roi.crease_top_y),
                                (roi.left_crease_x, roi.crease_top_y),
                                (0, 0, 0), 2
                            )
                            cv2.line(
                                frame_with_detections,
                                (roi.left_wicket_x, roi.crease_bottom_y),
                                (roi.left_crease_x, roi.crease_bottom_y),
                                (0, 0, 0), 2
                            )
                            cv2.line(
                                frame_with_detections,
                                (roi.right_wicket_x, roi.crease_top_y),
                                (roi.right_crease_x, roi.crease_top_y),
                                (0, 0, 0), 2
                            )
                            cv2.line(
                                frame_with_detections,
                                (roi.right_wicket_x, roi.crease_bottom_y),
                                (roi.right_crease_x, roi.crease_bottom_y),
                                (0, 0, 0), 2
                            )
                                
                            frame_detections = 0
                            results = st.session_state.analyzer.model(
                                frame,
                                conf=confidence_threshold
                            )
                                
                            for result in results:
                                for box in result.boxes:
                                    if int(box.cls[0]) == 1:  
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        center_x = (x1 + x2) // 2
                                        center_y = (y1 + y2) // 2
                                            
                                        
                                        if roi.is_point_inside(center_x, center_y):
                                            frame_detections += 1
                                            conf = float(box.conf[0])
                                            
                                            st.session_state.ball_detections.append({
                                                'frame': frame_count,
                                                'time': frame_count / fps,
                                                'position': (center_x, center_y),
                                                'confidence': conf,
                                                'in_crease_region': True  
                                            })
                                            
                                            smoothed_x, smoothed_y = st.session_state.ball_tracker.update(np.array([center_x, center_y]))
                                            st.session_state.all_smoothed_positions.append((int(smoothed_x), int(smoothed_y)))
                                                
                                            
                                            box_padding = 40
                                            
                                            cv2.rectangle(
                                                frame_with_detections,
                                                (x1-box_padding, y1-box_padding),
                                                (x2+box_padding, y2+box_padding),
                                                (0, 255, 0), 3  
                                            )
                                            
                                            cv2.circle(
                                                frame_with_detections,
                                                (center_x, center_y),
                                                10, (0, 255, 0), -1  
                                            )
                            
                            
                            if len(st.session_state.all_smoothed_positions) > 1:
                                total_points = len(st.session_state.all_smoothed_positions)
                                
                                for i in range(1, total_points):
                                    prev_pos = st.session_state.all_smoothed_positions[i-1]
                                    curr_pos = st.session_state.all_smoothed_positions[i]
                                    
                                    if any(pd.isna(val) for val in prev_pos) or any(pd.isna(val) for val in curr_pos):
                                        continue
                                    
                                    
                                    alpha = min(1.0, 0.5 + 0.5 * (i / total_points))
                                    thickness = int(3 + 4 * (i / total_points))
                                    color = (0, 255, 0)  
                                    
                                    cv2.line(
                                        frame_with_detections,
                                        (int(prev_pos[0]), int(prev_pos[1])),
                                        (int(curr_pos[0]), int(curr_pos[1])),
                                        color, thickness
                                    )
                                
                                
                                if total_points > 0:
                                    current_pos = st.session_state.all_smoothed_positions[-1]
                                    
                                    if not any(pd.isna(val) for val in current_pos):
                                        cv2.circle(
                                            frame_with_detections,
                                            (int(current_pos[0]), int(current_pos[1])),
                                            15, (0, 255, 0), -1  
                                        )
                                        
                                        cv2.circle(
                                            frame_with_detections,
                                            (int(current_pos[0]), int(current_pos[1])),
                                            15, (0, 0, 0), 3  
                                        )
                            
                            frame_time = time.time() - frame_start_time
                            processing_times.append(frame_time)
                            detection_counts.append(frame_detections)
                            
                            if len(processing_times) > performance_window:
                                processing_times.pop(0)
                                detection_counts.pop(0)
                            
                            recent_fps = 1.0 / (sum(processing_times) / len(processing_times))
                            avg_detections = sum(detection_counts) / len(detection_counts)
                            
                            frame_count += 1
                            progress = frame_count / total_frames
                            
                            progress_bar.progress(progress)
                            progress_text.metric(
                                "Progress",
                                f"{progress*100:.1f}%",
                                f"Frame {frame_count}/{total_frames}"
                            )
                            fps_metric.metric(
                                "Processing Speed",
                                f"{recent_fps:.1f} FPS",
                                f"{1000/recent_fps:.0f} ms/frame"
                            )
                            detection_count.metric(
                                "Enhanced Detections",
                                len(st.session_state.ball_detections),
                                f"{avg_detections:.1f}/frame"
                            )
                            
                            frame_placeholder.image(
                                cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB),
                                caption="Crease Ball Detection",
                                width=None
                            )
                            
                            if frame_count % 5 == 0:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    y=processing_times,
                                    name='Processing Time (s)',
                                    line=dict(color='blue')
                                ))
                                fig.add_trace(go.Scatter(
                                    y=detection_counts,
                                    name='Enhanced Detections',
                                    line=dict(color='green')
                                ))
                                fig.update_layout(
                                    title='Real-time Detection Performance',
                                    height=200,
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    yaxis_title='Count / Time'
                                )
                                performance_placeholder.plotly_chart(fig, use_container_width=True)
                                
                    finally:
                        cap.release()
                        total_time = time.time() - start_time
                        average_fps = frame_count / total_time
                        total_detections = len(st.session_state.ball_detections)
                        
                        st.success("✅ Ball detection completed!")
                        
                        with st.expander("Performance Summary", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            col1.metric(
                                "Average Speed",
                                f"{average_fps:.1f} FPS",
                                f"{1000/average_fps:.0f} ms/frame"
                            )
                            col2.metric(
                                "Total Processing Time",
                                f"{total_time:.1f} s"
                            )
                            col3.metric(
                                "Enhanced Detections",
                                total_detections,
                                f"{total_detections/frame_count:.1f}/frame"
                            )
                
            except Exception as e:
                logger.error(f"Error in enhanced crease ball detection: {str(e)}")
                st.error("An error occurred during enhanced crease ball detection. Please try again.")
                return
        
        
        with tab3:
            try:
                st.header("Speed Analysis")
                st.info("📊 Speed calculations with accuracy and outlier filtering (speeds in excluded from Excel export)")
                
                if not hasattr(st.session_state, 'roi_coordinates') or not hasattr(st.session_state, 'pitch_length_pixels'):
                    st.warning("⚠️ ROI Detection Required", icon="⚠️")
                    st.info("Please complete the ROI detection in the previous tab first.")
                    return
                        
                if not st.session_state.ball_detections:
                    st.warning("⚠️ Ball Detection Required", icon="⚠️")
                    st.info("Please complete the ball detection process before proceeding with speed analysis.")
                    return

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Failed to open video file.")
                    return
                    
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                min_frame_diff = 1  
                    
                meters_per_pixel = st.session_state.pitch_length_meters / st.session_state.pitch_length_pixels  
                frame_duration = 1.0 / fps  
                cap.release()

                source_label = "Dahua Camera (RTSP)" if "rtsp" in str(video_path) else "Uploaded Video File"
                st.info(f"Analyzing crease zone detections from: {source_label}")
                
                
                roi = st.session_state.roi_coordinates
                total_detections = len(st.session_state.ball_detections)
                crease_to_crease_pixels = roi.right_crease_x - roi.left_crease_x
                
                st.subheader("Analysis Parameters")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Crease to Crease", f"{crease_to_crease_pixels} pixels")
                with col2:
                    st.metric("Meters per Pixel", f"{meters_per_pixel:.6f}")
                with col3:
                    st.metric("FPS", f"{fps:.2f}")
            
                
                with st.spinner("Processing crease zone ball trajectory with improved accuracy..."):
                    
                    
                    filtered_detections = st.session_state.ball_detections
                    
                    if len(filtered_detections) < 2:
                        st.warning("Insufficient ball detections for speed analysis.")
                        return

                    logger.info(f"Processing {len(filtered_detections)} enhanced crease zone detections with corrected algorithms")
                    
                    frame_numbers = []
                    timestamps = []
                    positions = []
                    speeds_kmh = []

                    for i in range(1, len(filtered_detections)):
                        prev = filtered_detections[i-1]
                        curr = filtered_detections[i]
                        
                        frame_diff = curr['frame'] - prev['frame']
                        
                        pos1 = prev['position']
                        pos2 = curr['position']
                        
                        if any(pd.isna(val) for val in pos1) or any(pd.isna(val) for val in pos2):
                            logger.warning(f"NaN values detected in positions: {pos1}, {pos2}")
                            continue
                        
                        
                        dx = float(pos2[0]) - float(pos1[0])
                        dy = float(pos2[1]) - float(pos1[1])
                        distance_pixels = np.sqrt(dx**2 + dy**2)  
                        
                        time_interval = frame_diff * frame_duration
                        distance_meters = distance_pixels * meters_per_pixel
                        
                        
                        if time_interval > 0 and frame_diff >= min_frame_diff:
                            speed_ms = distance_meters / time_interval
                            speed_kmh = speed_ms * 3.6
                            
                            
                            max_threshold = 125.0  
                            if speed_kmh > max_threshold:
                                logger.warning(f"Speed above threshold: {speed_kmh:.2f} km/h, filtering out")
                                speed_kmh = 0.0  
                            elif speed_kmh < 0:
                                speed_kmh = 0.0
                        else:
                            speed_kmh = 0.0

                        frame_numbers.append(curr['frame'])
                        timestamps.append(curr['time'])
                        positions.append(pos2)
                        speeds_kmh.append(speed_kmh)

                    if not speeds_kmh:
                        st.warning("No valid speed measurements calculated.")
                        return

                    
                    filtered_speeds = SpeedDataFilter.filter_speed_outliers(speeds_kmh, method='iqr')
                    logger.info(f"Applied outlier filtering: {len(speeds_kmh)} -> {len(filtered_speeds)} speeds")

                    try:
                        excel_data = create_export_file(
                            frame_numbers=frame_numbers,
                            timestamps=timestamps,
                            smoothed_positions=positions,
                            speeds=speeds_kmh,  
                            fps=fps,
                            meters_per_pixel=meters_per_pixel,
                            pitch_length_pixels=st.session_state.pitch_length_pixels
                        )
                    except Exception as e:
                        logger.error(f"Error creating export file: {str(e)}")
                        st.error("Failed to create analysis export file.")
                        return

                    try:
                        with io.BytesIO(excel_data) as bio:
                            df = pd.read_excel(bio, sheet_name='Analysis')
                            stats_start = df[df.iloc[:, 0] == "Summary Statistics"].index[0]
                            stats_df = df.iloc[stats_start+1:stats_start+12]
                            
                            avg_speed = float(stats_df.iloc[0, 1])
                            max_speed = float(stats_df.iloc[1, 1])
                            sustained_speed = float(stats_df.iloc[2, 1])
                            
                            avg_crease_speed = float(stats_df.iloc[3, 1])
                            max_crease_speed = float(stats_df.iloc[4, 1])
                            
                            st.subheader("Speed Metrics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Average Speed",
                                    f"{avg_speed:.3f} km/h",
                                    delta=f"({avg_speed/3.6:.3f} m/s)"
                                )
                            with col2:
                                st.metric(
                                    "Maximum Speed",
                                    f"{max_speed:.3f} km/h",
                                    delta=f"({max_speed/3.6:.3f} m/s)"
                                )
                            with col3:
                                st.metric(
                                    "Sustained Speed",
                                    f"{sustained_speed:.3f} km/h",
                                    delta=f"({sustained_speed/3.6:.3f} m/s)"
                                )
                                
                    except Exception as e:
                        logger.error(f"Error processing enhanced crease zone statistics: {str(e)}")
                        st.error("Failed to process enhanced crease zone speed statistics.")
                        return

                    with st.expander("📊 Calculation Details", expanded=False):
                        st.write("Speed calculation parameters:")
                        st.write(f"- **Crease to crease**: {crease_to_crease_pixels} pixels")
                        st.write(f"- **Meters per pixel**: {meters_per_pixel:.6f}")
                        st.write(f"- **FPS**: {fps:.2f}")
                        st.write(f"- **Frame duration (1/FPS)**: {frame_duration:.6f} seconds")
                        st.write(f"- Pitch length: {st.session_state.pitch_length_meters} meters")
                        st.write(f"- Outlier filtering: IQR method applied")
                        st.write(f"- Video source: {source_label}")

                    st.subheader("📥 Export Analysis")
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    source_type = "camera" if "rtsp" in str(video_path) else "upload"
                    download_filename = f"cricket_analysis_{source_type}_{timestamp}.xlsx"
                    
                    
                    with io.BytesIO(excel_data) as bio:
                        df_temp = pd.read_excel(bio, sheet_name='Analysis')
                        stats_start = df_temp[df_temp.iloc[:, 0] == "Summary Statistics"].index[0]
                        excluded_count = int(df_temp.iloc[stats_start+13, 1])  
                        
                    if excluded_count > 0:
                        st.warning(f"measurements saved in the Excel file")
                    else:
                        st.success("✅ All speed measurements are within excel sheet")
                    
                    st.download_button(
                        label="📊 Download Analysis Excel Report",
                        data=excel_data,
                        file_name=download_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Excel file contains only speed measurements within the 125 km/h threshold"
                    )

                    
                    if len(positions) > 2:
                        try:
                            st.subheader("📈 Speed Analysis Charts")
                            
                            
                            fig_speed, fig_histogram, fig_scatter, fig_box = plot_results(frame_numbers, speeds_kmh, positions)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(fig_speed, use_container_width=True)
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            with col2:
                                st.plotly_chart(fig_histogram, use_container_width=True)
                                st.plotly_chart(fig_box, use_container_width=True)
                                
                        except Exception as e:
                            logger.error(f"Error creating speed charts: {str(e)}")
                            st.warning("Could not create speed analysis charts.")

            except Exception as e:
                logger.error(f"Error in crease zone speed analysis: {str(e)}")
                st.error("An error occurred during crease zone speed analysis. Please check the logs for details.")
                raise



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again or contact support.")

