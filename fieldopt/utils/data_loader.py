import numpy as np

def read_initial_field(file_path, center_path, get_difference):
    """
    Read initial field data from specified file path.

    File format of the initial field:
    - Line 1: x,y,z coordinate minimum values (min_x min_y min_z)
    - Line 2: x,y,z coordinate maximum values (max_x max_y max_z)
    - Line 3: A float value t_end
    - Line 4 and beyond: Data for each point (x y z value1 value2)

    Args:
        file_path (str): Path to the initial field file.

    Returns:
        tuple: A tuple containing:
            - bounding_box (list): List containing min and max coordinates [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            - t_end (float): t_end value defined in the file.
            - points (np.ndarray): ndarray with shape (N, 3) containing all point coordinates.
            - field1_values (np.ndarray): ndarray with shape (N, 1) containing values for the first field.
            - field2_values (np.ndarray): ndarray with shape (N, 1) containing values for the second field.
    """

    with open(center_path, 'r') as f:
        real_bounds = f.readline()
        real_bounds = np.array(real_bounds.strip().split(','), dtype=np.float32)
        min_bound = real_bounds[0:3]
        max_bound = real_bounds[3:6]
        bounding_box = [min_bound.tolist(), max_bound.tolist()]
        centers = np.loadtxt(f, dtype=np.float32, delimiter=',')

    with open(file_path, 'r') as f:
        # Read first three lines for metadata
        min_bound_str = f.readline()
        max_bound_str = f.readline()
        t_end_str = f.readline()

        # Parse bounding box
        min_bound = np.array(min_bound_str.strip().split(','), dtype=np.float32)
        max_bound = np.array(max_bound_str.strip().split(','), dtype=np.float32)
        # bounding_box = [min_bound.tolist(), max_bound.tolist()]

        # Parse t_end
        t_end = float(t_end_str)

        # Use numpy to efficiently read remaining data, specifying comma as delimiter
        data = np.loadtxt(f, dtype=np.float32, delimiter=',')

    # points = data[:, :3]
    points = centers
    field1_values = data[:, 3:4]
    field2_values = data[:, 4:5]

    # 将field1和field2中所有大于t_end的值加上t_end/2
    field1_values[field1_values > t_end] += t_end/4
    field2_values[field2_values > t_end] += t_end/4

    if get_difference:
        field2_values = field2_values - field1_values

    return bounding_box, t_end, points, field1_values, field2_values

if __name__ == '__main__':
    file_path = (
        'model_data/initial_fields/bracket/'
        'bracket_res100_initial_fields.txt'
    )
    center_path = (
        'model_data/initial_fields/bracket/'
        'bracket_res100_centers.txt'
    )
    
    try:
        bbox, t, pts, vals1, vals2 = read_initial_field(
            file_path,
            center_path,
            get_difference=False,
        )

        print("Bounding Box:", bbox)
        print("t_end:", t)
        print("\nNumber of points:", pts.shape[0])
        print("Points shape:", pts.shape)
        print("Field 1 values shape:", vals1.shape)
        print("Field 2 values shape:", vals2.shape)

        print("\nExample data for first point:")
        print("  Point coordinates:", pts[0])
        print("  Field 1 value:", vals1[0])
        print("  Field 2 value:", vals2[0])
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the path.")
    except Exception as e:
        print(f"Error occurred while reading file: {e}")
