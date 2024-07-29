import os
import cv2
import numpy as np
import scipy

def Homography_function(M_pt, corners):

    num_points = len(M_pt)
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        A[2*i, :] = [-M_pt[i, 0], -M_pt[i, 1], -1, 0, 0, 0, M_pt[i, 0]*corners[i, 0], M_pt[i, 1]*corners[i, 0], corners[i, 0]]
        A[2*i + 1, :] = [0, 0, 0, -M_pt[i, 0], -M_pt[i, 1], -1, M_pt[i, 0]*corners[i, 1], M_pt[i, 1]*corners[i, 1], corners[i, 1]]

    _, _, V = np.linalg.svd(A)

    H = V[-1, :].reshape((3, 3))
    
    return H

def checkerboard_homography(images, M_pt, Output_folder_checkerboard):
    
    Homography_list = []
    Corners_list = []
    images_copy = np.copy(images)

    for idx, image in enumerate(images_copy):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image_gray, (9, 6))
        
        if ret:
            corners = corners.reshape(-1, 2)
            # Homography = Homography_function(M_pt, corners)
            Homography, _ = cv2.findHomography(M_pt, corners, cv2.RANSAC, 5.0)
            Corners_list.append(corners)
            Homography_list.append(Homography)

            cv2.drawChessboardCorners(image, (9, 6), corners, ret)

            Output_folder_checkerboard_path = os.path.join(Output_folder_checkerboard, f"Checkerboard_{idx+1}.jpg")
            cv2.imwrite(Output_folder_checkerboard_path, image)

    return Homography_list, Corners_list

def v(i, j, H):
    
    v_ij = np.array([
        H[0][i] * H[0][j],
        H[0][i] * H[1][j] + H[1][i] * H[0][j],
        H[1][i] * H[1][j],
        H[2][i] * H[0][j] + H[0][i] * H[2][j],
        H[2][i] * H[1][j] + H[1][i] * H[2][j],
        H[2,][i] * H[2][j]
    ])
    
    return v_ij

def Intrinsic_Matrix(Homography):
    
    V = []
    
    for h in Homography:
        V.append(v(0, 1, h))
        V.append(v(0, 0, h) - v(1, 1, h))
    
    V = np.array(V)
    _, _, vt = np.linalg.svd(V)

    b = vt[-1][:]
    B11, B12, B22, B13, B23, B33 = b

    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lamda_ = B33 - ((B13**2) + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lamda_ / B11)
    beta = np.sqrt(lamda_ * B11 / (B11 * B22 - (B12**2)))
    gamma = -(B12 * (alpha**2) * beta / lamda_)
    u0 = (gamma * v0 / beta) - (B13 * (alpha**2) / lamda_)
    A = np.array([[alpha, gamma, u0],
                  [0    , beta , v0],
                  [0    , 0    , 1 ]])
    
    return A

def Rotation_Matrix(Intrinsic_A, Homography):
    
    R = []
    
    A_inv = np.linalg.inv(Intrinsic_A)
    
    for h in Homography:
        h1 = h[:, 0]
        h2 = h[:, 1]
        h3 = h[:, 2]
        
        lamda_ = 1/scipy.linalg.norm((A_inv @ h1), 2)
        r1 = lamda_ * (A_inv @ h1) 
        r2 = lamda_ * (A_inv @ h2) 
        # r3 = np.cross(r1, r2)
        t = lamda_ * (A_inv @ h3)
        R_plus_t = np.vstack((r1, r2, t)).T
        
        R.append(R_plus_t)
        
    return R

def loss(Intrinsic_A, Corners, M_pt, Rotation_R):
    
    alpha, gamma, beta, u0, v0, k1, k2 = Intrinsic_A
    
    intrinsic_matrix = np.array([[alpha, gamma, u0],
                                 [0    , beta , v0],
                                 [k1   , k2   , 1 ]])

    num_corners = len(Corners)
    final_errors = np.zeros(num_corners)

    for i in range(num_corners):
        corner = Corners[i]
        rotation_matrix = Rotation_R[i]

        homography = intrinsic_matrix @ rotation_matrix

        total_error = 0
        for j in range(len(M_pt)):
            world_pt = np.array([M_pt[j][0], M_pt[j][1], 1])

            Projected_Normalized_Img_Coordinates = rotation_matrix @ world_pt
            x, y = Projected_Normalized_Img_Coordinates[0] / Projected_Normalized_Img_Coordinates[2], Projected_Normalized_Img_Coordinates[1] / Projected_Normalized_Img_Coordinates[2]
            
            Projected_Pixel_Img_Coordinates = homography @ world_pt
            u, v = Projected_Pixel_Img_Coordinates[0] / Projected_Pixel_Img_Coordinates[2], Projected_Pixel_Img_Coordinates[1] / Projected_Pixel_Img_Coordinates[2]

            mij = np.array([corner[j][0], corner[j][1], 1], dtype=np.float32)

            u_cap = u + (u - u0) * (k1 * ((x**2) + (y**2)) + k2 * (((x**2) + (y**2))**2))
            v_cap = v + (v - v0) * (k1 * ((x**2) + (y**2)) + k2 * (((x**2) + (y**2))**2))
            mij_cap = np.array([u_cap, v_cap, 1], dtype=np.float32)

            error = scipy.linalg.norm((mij - mij_cap), 2)
            total_error += error

        final_errors[i] = total_error / len(M_pt)

    return final_errors

def Optimize(Intrinsic_A, Corners, M_pt, Rotation_R):
    
    alpha = Intrinsic_A[0, 0]
    gamma = Intrinsic_A[0, 1]
    u0 = Intrinsic_A[0, 2]
    beta = Intrinsic_A[1, 1]
    v0 = Intrinsic_A[1, 2]
    k1 = Intrinsic_A[2, 0]
    k2 = Intrinsic_A[2, 1]
    
    optimized = scipy.optimize.least_squares(fun=loss, x0 = [alpha, gamma, beta, u0, v0, k1, k2], method = 'lm', args=(Corners, M_pt, Rotation_R))
    
    [alpha_optimized, gamma_optimized, beta_optimized, u0_optimized, v0_optimized, k1_optimized, k2_optimized] = optimized.x
    
    A_optimized = np.array([[alpha_optimized, gamma_optimized, u0_optimized],
                            [0              , beta_optimized , v0_optimized],
                            [0              , 0              , 1           ]])
    
    return A_optimized, k1_optimized, k2_optimized

def error(A_optimized, Distortion, Rotation_R_optimized, Corners, M_pt):
    
    u0 = A_optimized[0][2]
    v0 = A_optimized[1][2]
    k1, k2= Distortion[0], Distortion[1]

    reprojected_error = []
    reprojected_pts = []
    
    num_corners = len(Corners)
    
    for i in range(num_corners):
        corner = Corners[i]
        rotation_matrix = Rotation_R_optimized[i]
        
        homography = A_optimized @ rotation_matrix

        total_error = 0
        reprojected_corners = []
        for j in range(len(M_pt)):
            world_pt = np.array([ M_pt[j][0],  M_pt[j][1], 1])

            Projected_Normalized_Img_Coordinates = rotation_matrix @ world_pt
            x, y = Projected_Normalized_Img_Coordinates[0] / Projected_Normalized_Img_Coordinates[2], Projected_Normalized_Img_Coordinates[1] / Projected_Normalized_Img_Coordinates[2]

            Projected_Pixel_Img_Coordinates = homography @ world_pt
            u, v = Projected_Pixel_Img_Coordinates[0] / Projected_Pixel_Img_Coordinates[2], Projected_Pixel_Img_Coordinates[1] / Projected_Pixel_Img_Coordinates[2]

            mij = np.array([corner[j][0], corner[j][1], 1])

            u_cap = u + (u - u0) * (k1 * ((x**2) + (y**2)) + k2 * (((x**2) + (y**2))**2))
            v_cap = v + (v - v0) * (k1 * ((x**2) + (y**2)) + k2 * (((x**2) + (y**2))**2))
            
            reprojected_corners.append([int(u_cap[0]), int(v_cap[0])])
            mij_cap = np.array([u_cap[0], v_cap[0], 1])
            error = scipy.linalg.norm((mij - mij_cap), 2)
           
            total_error += error

        reprojected_error.append(total_error)
        reprojected_pts.append(reprojected_corners)

    reprojected_error_avg = np.sum(np.array(reprojected_error)) / (len(Corners) * M_pt.shape[0])
    reprojected_error_list = np.array(reprojected_error) / (len(Corners) * M_pt.shape[0])
    
    return reprojected_error_avg, reprojected_error_list, reprojected_pts

def main():
    folder_path = "Calibration_Imgs"
    output_folder = "Results"
    output_folder_checkerboard = os.path.join(output_folder, "Checkerboard")
    output_folder_final = os.path.join(output_folder, "Final")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_checkerboard, exist_ok=True)
    os.makedirs(output_folder_final, exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [cv2.imread(os.path.join(folder_path, f)) for f in image_files]

    size = 21.5

    M_x, M_y = np.meshgrid(range(9), range(6))
    M_pt = np.hstack((M_x.reshape(-1, 1), M_y.reshape(-1, 1))).astype(np.float32) * size
    
    Homography, Corners = checkerboard_homography(images, M_pt, output_folder_checkerboard)
    
    Intrinsic_A = Intrinsic_Matrix(Homography)
    
    Rotation_R = Rotation_Matrix(Intrinsic_A, Homography)
    
    A_optimized, k1_optimized, k2_optimized = Optimize(Intrinsic_A, Corners, M_pt, Rotation_R)
    
    print(f"k1 Optimized: ", k1_optimized)
    print(f"k2 Optimized: ", k2_optimized)
    
    Rotation_R_optimized = Rotation_Matrix(A_optimized, Homography)
    
    Distortion = np.array([0, 0]).reshape(2, 1)
    reprojected_error_avg, reprojected_error_list, reprojected_pts = error(Intrinsic_A, Distortion, Rotation_R, Corners, M_pt)
    print("Error before Optimization: ")
    print(reprojected_error_list)
    print(f"Average Error before optimization : ", reprojected_error_avg)
    
    Distortion = np.array([k1_optimized, k2_optimized]).reshape(2, 1)
    reprojected_error_avg, reprojected_error_list, reprojected_pts = error(A_optimized, Distortion, Rotation_R_optimized, Corners, M_pt)
    print("Error after Optimization: ")
    print(reprojected_error_list)
    print(f"Average Error after optimization : ", reprojected_error_avg)
    
    A_matrix = np.array(A_optimized, np.float32).reshape(3,3)
    print("Intrinsic Matrix A Optimized: ")
    print(A_matrix)
    
    distortion = np.array([Distortion[0][0],Distortion[1][0], 0, 0], np.float32)
    print(f"Distortion: ", distortion)
    
    
    for idx, image_points in enumerate(reprojected_pts):
        image = cv2.undistort(images[idx], A_matrix, distortion)
        for point in image_points:
            image = cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), 10)

        Output_folder_final_path = os.path.join(output_folder_final, f"Final_{idx+1}.jpg")
        cv2.imwrite(Output_folder_final_path, image)

if __name__ == "__main__":
    main()
