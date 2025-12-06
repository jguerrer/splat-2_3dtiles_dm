import numpy as np
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
# Choose reference automatically from average camera position
AUTO_REFERENCE = True

# Or set your own reference manually like this:
# REFERENCE = np.array([518900.0, 2750000.0, 1700.0])


# ----------------------------------------------------------
# Read points3D.txt
# ----------------------------------------------------------
def read_points3d(path="points3D.txt"):
    points = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue
            elems = line.split()
            point_id = int(elems[0])
            xyz = np.array(list(map(float, elems[1:4])))
            rgb = list(map(int, elems[4:7]))
            error = float(elems[7])
            track = list(map(int, elems[8:]))
            points[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track
            }
    return points

# ----------------------------------------------------------
# Read images.txt (poses)
# ----------------------------------------------------------
def read_images(path="images.txt"):
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()

        if line.startswith("#") or len(line) == 0:
            idx += 1
            continue

        # ---- First line (pose) ----
        elems = line.split()
        image_id = int(elems[0])
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        cam_id = int(elems[8])
        name = elems[9]

        # ---- Second line (2D-3D correspondences) ----
        idx += 1
        obs_line = lines[idx].strip()
        obs = list(map(float, obs_line.split()))

        images[image_id] = {
            "qvec": np.array([qw, qx, qy, qz]),
            "tvec": np.array([tx, ty, tz]),
            "camera_id": cam_id,
            "name": name,
            "obs_line": obs_line
        }

        idx += 1

    return images

# ----------------------------------------------------------
# Compute camera centers in world coordinates
# ----------------------------------------------------------
def compute_camera_centers(images):
    centers = {}
    for img_id, img in images.items():
        q = img["qvec"]
        t = img["tvec"]

        Rwc = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # COLMAP uses q=(w,x,y,z)
        C = -Rwc.T @ t
        centers[img_id] = C
    return centers

# ----------------------------------------------------------
# Write shifted images.txt
# ----------------------------------------------------------
def write_images_shifted(images, centers_shifted, out="images_shifted.txt"):
    with open(out, "w") as f:
        f.write("# Shifted COLMAP images file\n")

        for img_id, img in images.items():
            q = img["qvec"]
            name = img["name"]
            cam_id = img["camera_id"]

            # Compute new T
            q_colmap = np.array([q[0], q[1], q[2], q[3]])
            Rwc = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            C_shift = centers_shifted[img_id]
            t_new = -Rwc @ C_shift

            f.write(f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} {t_new[0]} {t_new[1]} {t_new[2]} {cam_id} {name}\n")
            f.write(img["obs_line"] + "\n")

# ----------------------------------------------------------
# Write shifted points3D
# ----------------------------------------------------------
def write_points_shifted(points, shift, out="points3D_shifted.txt"):
    with open(out, "w") as f:
        f.write("# Shifted COLMAP points3D file\n")
        for pid, p in points.items():
            xyz = p["xyz"] - shift
            rgb = p["rgb"]
            err = p["error"]
            track = " ".join(map(str, p["track"]))
            line = f"{pid} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {err} {track}\n"
            f.write(line)

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
path = r"H:\deep_matrix\tower_images_1040_124\tower_images_1040_124\colmap_highest_ecef\sparse\0\\"
points = read_points3d(path + "points3D.txt")
images = read_images(path + "images.txt")
print(f"Read {len(points)} points and {len(images)} images.")


# Compute camera centers in world coordinates, so they are translated
centers = compute_camera_centers(images) #

print("Camera centers (world coordinates):")
for img_id, C in centers.items():
    print(f"Image {img_id}: Center = {C}")  
# Choose reference
if AUTO_REFERENCE:
    all_centers = np.array(list(centers.values()))
    REFERENCE = all_centers.mean(axis=0)
    print("Using average camera center as reference:", REFERENCE)

# # Shift camera centers
centers_shifted = {img_id: C - REFERENCE for img_id, C in centers.items()}

# # Write outputs
write_images_shifted(images, centers_shifted, out=path + "images_shifted.txt")
write_points_shifted(points, REFERENCE, out=path + "points3D_shifted.txt"   )

print("Done. Files written: images_shifted.txt, points3D_shifted.txt")
