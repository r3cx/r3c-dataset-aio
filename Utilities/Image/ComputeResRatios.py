import math

def calculate_constant_area_resolutions(target_w, target_h, ratios):
    """
    Calculates the dimensions for a given list of aspect ratios 
    such that the total area (H' * W') equals the target area (H * W).
    """
    results = []
    # Calculate the target pixel area
    target_area = target_w * target_h
    
    for label, (ratio_w, ratio_h) in ratios.items():
        # Numerical aspect ratio (Width / Height)
        aspect_ratio = ratio_w / ratio_h
        
        # Calculate exact dimensions based on constant area
        # W' = sqrt(Area * aspect)
        # H' = sqrt(Area / aspect)
        exact_w = math.sqrt(target_area * aspect_ratio)
        exact_h = math.sqrt(target_area / aspect_ratio)
        
        # Round to the nearest even pixel (often preferred for video/images)
        # Change to round() if you don't need even numbers
        new_w = round(exact_w / 2) * 2
        new_h = round(exact_h / 2) * 2
        
        # Calculate actual area and variance from target
        actual_area = new_w * new_h
        variance = ((actual_area - target_area) / target_area) * 100
        
        results.append({
            "Ratio": label,
            "Target Aspect": f"{ratio_w}:{ratio_h}",
            "Resolution": f"{new_w}x{new_h}",
            "Total Pixels": actual_area,
            "Variance": f"{variance:+.2f}%"
        })
        
    return results

# --- Configuration ---
# Target base dimensions (e.g., a 1536x1536 canvas has 2,359,296 pixels)
BASE_WIDTH = 1280
BASE_HEIGHT = 1024

# Common target aspect ratios (Width, Height)
target_ratios = {
    "Square": (1, 1),
    "Widescreen (Landscape)": (16, 9),
    "Vertical (Stories/Reels)": (9, 16),
    "Standard (Landscape)": (4, 3),
    "Standard (Portrait)": (3, 4),
    "Photo (Landscape)": (3, 2),
    "Photo (Portrait)": (2, 3),
    "Ultrawide": (21, 9),
    "Instagram Post (Portrait)": (4, 5),
    "Photo (Wide Portrait)": (10, 16),
    "Photo (Wider Portrait)": (11, 16)
}

# --- Execution ---
output_resolutions = calculate_constant_area_resolutions(BASE_WIDTH, BASE_HEIGHT, target_ratios)

# Display results
print(f"Target Megapixels: {(BASE_WIDTH * BASE_HEIGHT) / 1e6:.2f}MP ({BASE_WIDTH * BASE_HEIGHT:,} pixels)\n")
print(f"{'Ratio Type':<30} | {'Aspect':<8} | {'Resolution':<12} | {'Pixel Count':<12} | {'Area Dev.'}")
print("-" * 75)
for res in output_resolutions:
    print(f"{res['Ratio']:<30} | {res['Target Aspect']:<8} | {res['Resolution']:<12} | {res['Total Pixels']:<12,} | {res['Variance']}")