import numpy as np

def calculate_luminance(rgb):
    # Normalize RGB values to the range 0-1
    rgb = [component / 255.0 for component in rgb]
    
    # Apply gamma correction to each component
    rgb = [
        comp / 12.92 if comp <= 0.03928 else ((comp + 0.055) / 1.055) ** 2.4 
        for comp in rgb
    ]
    
    # Calculate relative luminance
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return luminance

def calculate_contrast_ratio(color1, color2):
    # Calculate luminance for both colors
    luminance1 = calculate_luminance(color1)
    luminance2 = calculate_luminance(color2)
    
    # Ensure luminance1 is the lighter color
    if luminance1 < luminance2:
        luminance1, luminance2 = luminance2, luminance1
    
    # Calculate the contrast ratio
    contrast_ratio = (luminance1 + 0.05) / (luminance2 + 0.05)
    return contrast_ratio

def contrast_matrix_flat_list(palette):
    num_colors = len(palette)
    flat_list = []
    
    # Fill the matrix with contrast ratios in a flat list format
    for i in range(num_colors):
        for j in range(num_colors):
            if i == j:
                flat_list.append("0")  # Contrast of a color with itself
            else:
                ratio = calculate_contrast_ratio(palette[i], palette[j])
                flat_list.append(f"{int(ratio * 10):.0f}")  # Scale and convert to integer for simplicity
    
    return flat_list

# Example usage
palette = [
            [
                15,
                15,
                21
            ],
            [
                192,
                72,
                94
            ],
            [
                52,
                182,
                214
            ],
            [
                49,
                97,
                160
            ],
            [
                101,
                33,
                70
            ],
            [
                202,
                212,
                170
            ],
            [
                35,
                87,
                116
            ]
        ]
contrast_flat_list = contrast_matrix_flat_list(palette)

# Display the flat list of contrast ratios
print(contrast_flat_list)
