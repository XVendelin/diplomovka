"""
Image extraction utility - extracts polygon regions from images.
Python equivalent of MATLAB's druhy function.
"""
import numpy as np
from PIL import Image
from matplotlib.path import Path


# ============================================================================
# ORIENTATION FIX CONFIGURATION
# ============================================================================
# Change this if maps don't match MATLAB orientation:
#
# Options:
#   'none'              - No transformation (current default)
#   'transpose'         - Transpose the result (swap rows/cols)
#   'flip_vertical'     - Flip vertically (upside down)
#   'flip_horizontal'   - Flip horizontally (left-right)
#   'rotate_90'         - Rotate 90 degrees counterclockwise
#   'rotate_180'        - Rotate 180 degrees
#   'rotate_270'        - Rotate 270 degrees (90 clockwise)
#
# To find the correct setting:
#   1. Run: python diagnose_image_orientation.py
#   2. Compare with MATLAB output
#   3. Set ORIENTATION_FIX to the matching option
# ============================================================================

ORIENTATION_FIX = 'transpose'  # ← CHANGE THIS IF NEEDED

# ============================================================================


def druhy(filename, coords):
    """
    Extract a polygon region from an image.

    Parameters:
    -----------
    filename : str
        Path to the image file
    coords : np.ndarray
        Polygon coordinates as [N x 2] array of [row, col] pairs

    Returns:
    --------
    result : np.ndarray
        Matrix with points outside polygon set to 1, inside points
        contain original image values

    Example:
    --------
    coords = np.array([[250, 370],  # [row, col]
                       [220, 550],
                       [450, 550],
                       [450, 350]])
    result = druhy('image.jpg', coords)
    """
    # Read image
    img = Image.open(filename)

    # Convert to grayscale if needed
    if img.mode == 'RGB':
        img = img.convert('L')

    # Convert to numpy array and normalize to [0, 1]
    I = np.array(img, dtype=np.float64) / 255.0

    # Extract coordinates
    rows = coords[:, 0]
    cols = coords[:, 1]

    rmin = int(np.min(rows))
    rmax = int(np.max(rows))
    cmin = int(np.min(cols))
    cmax = int(np.max(cols))

    # Crop to bounding box
    cropped = I[rmin:rmax + 1, cmin:cmax + 1]

    # Shift coordinates relative to cropped region
    shifted_rows = rows - rmin
    shifted_cols = cols - cmin

    # Create meshgrid for polygon check
    height, width = cropped.shape
    cc, rr = np.meshgrid(np.arange(width), np.arange(height))

    # Create polygon path
    polygon_points = np.column_stack([shifted_cols, shifted_rows])
    polygon_path = Path(polygon_points)

    # Check which points are inside polygon
    points = np.column_stack([cc.flatten(), rr.flatten()])
    inside = polygon_path.contains_points(points).reshape(height, width)

    # Initialize result with ones
    result = np.ones_like(cropped)

    # Copy original values for points inside polygon
    result[inside] = cropped[inside]

    # Apply orientation fix if configured
    result = _apply_orientation_fix(result, ORIENTATION_FIX)

    return result


def _apply_orientation_fix(array, fix_type):
    """
    Apply orientation transformation to match MATLAB output.

    Parameters:
    -----------
    array : np.ndarray
        Input array
    fix_type : str
        Type of fix to apply

    Returns:
    --------
    np.ndarray
        Transformed array
    """
    if fix_type == 'none':
        return array
    elif fix_type == 'transpose':
        return array.T
    elif fix_type == 'flip_vertical':
        return np.flipud(array)
    elif fix_type == 'flip_horizontal':
        return np.fliplr(array)
    elif fix_type == 'rotate_90':
        return np.rot90(array, k=1)
    elif fix_type == 'rotate_180':
        return np.rot90(array, k=2)
    elif fix_type == 'rotate_270':
        return np.rot90(array, k=3)
    else:
        raise ValueError(f"Unknown orientation fix: {fix_type}. "
                        f"Valid options: 'none', 'transpose', 'flip_vertical', "
                        f"'flip_horizontal', 'rotate_90', 'rotate_180', 'rotate_270'")


def test_orientations(filename, coords):
    """
    Test all orientation options and display them.
    Useful for finding which orientation matches MATLAB.

    Parameters:
    -----------
    filename : str
        Path to image file
    coords : np.ndarray
        Polygon coordinates

    Returns:
    --------
    dict
        Dictionary mapping orientation names to extracted maps
    """
    import matplotlib.pyplot as plt

    orientations = [
        'none', 'transpose', 'flip_vertical', 'flip_horizontal',
        'rotate_90', 'rotate_180', 'rotate_270'
    ]

    results = {}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Save original setting
    global ORIENTATION_FIX
    original_fix = ORIENTATION_FIX

    for i, orientation in enumerate(orientations):
        ORIENTATION_FIX = orientation
        try:
            result = druhy(filename, coords)
            results[orientation] = result

            axes[i].imshow(result, cmap='gray')
            axes[i].set_title(orientation.replace('_', ' ').title())
            axes[i].axis('equal')
        except Exception as e:
            axes[i].set_title(f'{orientation}\nERROR')
            axes[i].text(0.5, 0.5, str(e)[:50], ha='center', va='center',
                        fontsize=8, wrap=True)

    # Hide last subplot
    axes[7].axis('off')

    # Restore original setting
    ORIENTATION_FIX = original_fix

    plt.suptitle('Image Extraction - All Orientations\n'
                 'Compare with MATLAB output to find correct orientation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('orientation_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison to 'orientation_comparison.png'")
    print(f"\nCurrent setting: ORIENTATION_FIX = '{original_fix}'")
    print("\nTo change orientation:")
    print("  1. Look at 'orientation_comparison.png'")
    print("  2. Find which matches your MATLAB output")
    print("  3. Edit image_extraction.py and change ORIENTATION_FIX")
    plt.show()

    return results


# Quick test function
if __name__ == '__main__':
    print("=" * 70)
    print("IMAGE EXTRACTION ORIENTATION TEST")
    print("=" * 70)
    print(f"\nCurrent orientation setting: '{ORIENTATION_FIX}'")
    print("\nTo test all orientations:")
    print("  python -c \"from image_extraction import test_orientations; "
          "import numpy as np; test_orientations('image.jpg', "
          "np.array([[280, 400], [280, 520], [400, 520], [400, 400]]))\"")
    print("\nOr use the diagnostic script:")
    print("  python diagnose_image_orientation.py")
    print("=" * 70)