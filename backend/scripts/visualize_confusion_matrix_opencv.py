"""
Confusion Matrix Visualization using OpenCV
Creates annotated heatmap visualizations for model evaluation.
"""

import numpy as np
import cv2
from typing import List, Tuple


def create_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    output_path: str = None
) -> np.ndarray:
    """
    Create confusion matrix heatmap visualization using OpenCV.
    
    Args:
        cm: Confusion matrix (n_classes x n_classes)
        class_names: List of class labels
        title: Plot title
        output_path: Optional path to save image
    
    Returns:
        BGR image array
    """
    n_classes = len(class_names)
    
    # Create larger canvas for labels
    cell_size = 120
    margin_top = 100
    margin_left = 150
    margin_right = 50
    margin_bottom = 50
    
    img_width = margin_left + (n_classes * cell_size) + margin_right
    img_height = margin_top + (n_classes * cell_size) + margin_bottom
    
    # Create white background
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Normalize confusion matrix for color mapping
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Draw cells with color intensity
    for i in range(n_classes):
        for j in range(n_classes):
            # Calculate position
            x = margin_left + (j * cell_size)
            y = margin_top + (i * cell_size)
            
            # Color based on value (green for correct, red for errors)
            if i == j:
                # Diagonal (correct predictions) - green scale
                intensity = int(255 * (1 - cm_normalized[i, j]))
                color = (intensity, 255, intensity)  # Green
            else:
                # Off-diagonal (errors) - red scale
                intensity = int(255 * (1 - cm_normalized[i, j]))
                color = (intensity, intensity, 255)  # Red
            
            # Draw cell
            cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), 
                         color, -1)
            cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), 
                         (0, 0, 0), 2)
            
            # Add count text
            count_text = str(cm[i, j])
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       1.2, 2)[0]
            text_x = x + (cell_size - text_size[0]) // 2
            text_y = y + (cell_size + text_size[1]) // 2
            cv2.putText(img, count_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # Add percentage
            if cm[i, j] > 0:
                pct_text = f"{cm_normalized[i, j]*100:.1f}%"
                pct_size = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, 1)[0]
                pct_x = x + (cell_size - pct_size[0]) // 2
                pct_y = y + cell_size - 15
                cv2.putText(img, pct_text, (pct_x, pct_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    
    # Add row labels (True labels)
    for i, label in enumerate(class_names):
        y = margin_top + (i * cell_size) + cell_size // 2
        cv2.putText(img, label, (10, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add column labels (Predicted labels)
    for j, label in enumerate(class_names):
        x = margin_left + (j * cell_size) + cell_size // 2
        # Rotate text would require more complex rendering, use horizontal
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                     0.8, 2)[0]
        text_x = x - label_size[0] // 2
        cv2.putText(img, label, (text_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add axis labels
    cv2.putText(img, "True Label", (10, margin_top - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    title_x = (img_width - title_size[0]) // 2
    cv2.putText(img, title, (title_x, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    
    pred_label_y = margin_top - 10
    pred_text = "Predicted Label"
    pred_size = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    pred_x = margin_left + ((n_classes * cell_size) - pred_size[0]) // 2
    cv2.putText(img, pred_text, (pred_x, pred_label_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Add legend
    legend_y = img_height - 30
    cv2.rectangle(img, (margin_left, legend_y), 
                 (margin_left + 30, legend_y + 20), (100, 255, 100), -1)
    cv2.putText(img, "Correct", (margin_left + 40, legend_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    legend_x2 = margin_left + 150
    cv2.rectangle(img, (legend_x2, legend_y), 
                 (legend_x2 + 30, legend_y + 20), (100, 100, 255), -1)
    cv2.putText(img, "Misclassified", (legend_x2 + 40, legend_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Save if path provided
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"✓ Confusion matrix saved to {output_path}")
    
    return img


def calculate_metrics_overlay(cm: np.ndarray, class_names: List[str]) -> str:
    """Calculate precision, recall, F1 for overlay text."""
    metrics = []
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append(f"{name}: P={precision:.2f} R={recall:.2f} F1={f1:.2f}")
    
    return "\n".join(metrics)


def demo_confusion_matrix():
    """Demo: Create confusion matrix visualization."""
    print("="*60)
    print("OpenCV Confusion Matrix Visualization Demo")
    print("="*60)
    
    # Example confusion matrix for drone failure prediction
    cm = np.array([
        [1234, 12, 5],      # Normal flights
        [8, 234, 15],       # Motor failures
        [3, 18, 456]        # Vibration failures
    ])
    
    class_names = ['Normal', 'Motor Fail', 'Vibration']
    
    print("\nGenerating confusion matrix visualization...")
    
    # Create visualization
    img = create_confusion_matrix_heatmap(
        cm,
        class_names,
        title="Drone Failure Classification",
        output_path="confusion_matrix.png"
    )
    
    # Calculate and display metrics
    print("\n📊 Classification Metrics:")
    print(calculate_metrics_overlay(cm, class_names))
    
    # Calculate overall accuracy
    accuracy = np.trace(cm) / cm.sum()
    print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")
    print(f"✓ Total Samples: {cm.sum()}")
    
    # Display the image
    print("\nDisplaying visualization (press any key to close)...")
    cv2.imshow("Confusion Matrix", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    # Check if OpenCV is installed
    try:
        import cv2
        demo_confusion_matrix()
    except ImportError:
        print("❌ OpenCV not installed. Install with:")
        print("   pip install opencv-python")
