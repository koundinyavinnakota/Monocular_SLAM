import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import Utils.MathUtils as mutils
import Utils.ImageUtils as imutils
import Utils.MiscUtils as miscutils


def orb_detect_kpts(image, no_kpts=None):
    orb = cv2.ORB_create(no_kpts)
    kp = orb.detect(image ,None)
    kp, des = orb.compute(image, kp)
    kpt_image = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)
    kpt_image = cv2.rotate(kpt_image, cv2.ROTATE_90_CLOCKWISE)
    return kpt_image


def get_orb_matches(image1, image2):
    query_img = image1
    train_img = image2
    
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(queryDescriptors, trainDescriptors)
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Initialize lists
    matches_image1 = list()
    matches_image2 = list()

    for mat in matches:
        query_idx = mat.queryIdx
        train_idx = mat.trainIdx
    
        # Get the coordinates: x - columns, y - rows
        x1, y1 = queryKeypoints[query_idx].pt
        x2, y2 = trainKeypoints[train_idx].pt
        
        matches_image1.append((x1, y1))
        matches_image2.append((x2, y2))
        
    matches_image1 = np.array(matches_image1)
    matches_image2 = np.array(matches_image2)
    
    return matches_image1, matches_image2
        

def build_projection_matrix(R, C):
    I = np.identity(3)
    IC = np.column_stack((I, -C))
    P = np.dot(R, IC)
    return P


def matrix_cofactor(matrix):
    try:
        determinant = np.linalg.det(matrix)
        if(determinant!=0):
            cofactor = None
            cofactor = np.linalg.inv(matrix).T * determinant
            # return cofactor matrix of the given matrix
            return cofactor
        else:
            raise Exception("singular matrix")
    except Exception as e:
        print("could not find cofactor matrix due to", e)
        
        
def build_world(F, map_points1, map_points2, P1):
    mtrx = np.trace(np.dot(F, F.T)/2)*np.identity(len(F)) - np.dot(F, F.T)
    h14 = np.sqrt(mtrx[0, 0])
    h24 = mtrx[0, 1] / h14
    h34 = mtrx[0, 2] / h14
    T_config1 = np.array([[0, -h34, h24], [h34, 0, -h14], [-h24, h14, 0]])
    T_config2 = (-1)*np.array(T_config1)
    
    C_config1 = np.transpose([h14, h24, h34])*(1/(h14**2 + h24**2 + h34**2))
    # print(C_config1)
    C_config2 = (-1)*C_config1
    
    cofactor_F = matrix_cofactor(F).T
    R_config1 = (cofactor_F.T - np.dot(T_config1, F))*(1/(h14**2 + h24**2 + h34**2))
    R_config2 = (cofactor_F.T - np.dot(T_config2, F))*(1/(h14**2 + h24**2 + h34**2))
    
    RT_combinations = [(C_config1, R_config1), (C_config1, R_config2), (C_config2, R_config1), (C_config2, R_config2)]
    best_config = None
    best_3d_points = None
    best_count = 0
    for comb in RT_combinations:
        # C_init = np.transpose([0, 0, 0])
        # R_init = np.identity(3)
        C2, R2 = comb
        P2 = build_projection_matrix(R2, C2)
        # print(best_matches_current)
        # print(P1, P2)
        X_3D = list()
        count = 0
        for i in range(len(map_points1)):
            x_3d = cv2.triangulatePoints(P1, P2, map_points1[i], map_points2[i])
            x_3d = x_3d / x_3d[-1]
            X_3D.append(x_3d[0:3])
            if x_3d[-1] > 0:
                count += 1
        if count > best_count:
            best_count = count
            # print(best_count)
            best_config = comb
            best_3d_points = X_3D
            
    return P2, best_config, best_3d_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--InputFilePath', default='Data/camera_calibrate.mp4', help='Path to the input file')
    parser.add_argument('-l', '--LoadOrNot', default=0, help='1 to load an existing file or 0 to recompute the Fundamental Matrices')
    parser.add_argument('-s', '--SaveFileName', default='Output_Files/result.avi', help='Name of the output file')
    
    args = parser.parse_args()
    input_path = args.InputFilePath
    load_data = bool(int(args.LoadOrNot))
    save_path = args.SaveFileName   
    
    K = np.array([[2.98969355e+03, 0.00000000e+00, 1.54772397e+03],
                  [0.00000000e+00, 2.98787392e+03, 2.01202879e+03],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    if load_data:
        with open('Output_Files/Frames.npy', 'rb') as f:
            all_frames = np.load(f, allow_pickle=True)
        with open('Output_Files/matches.npy', 'rb') as f:
            all_best_matches = np.load(f, allow_pickle=True)
        with open('Output_Files/fundamental_matrices.npy', 'rb') as f:
            all_F = np.load(f, allow_pickle=True)

        for i in range(len(all_frames)-1):
            frame1, frame2 = all_frames[i], all_frames[i+1]
            best_matches1, best_matches2 = np.array_split(all_best_matches[i], 2, axis=1)
            # print(best_matches1)
            plot_best_matches2 = imutils.get_plot_points(best_matches2, frame1.shape)
            imutils.plot_matches(frame1, frame2, best_matches1, plot_best_matches2)
            # funda_matrix = all_F[i]
            # mtrx = np.trace(np.dot(funda_matrix, funda_matrix.T)/2)*np.identity(len(funda_matrix)) - np.dot(funda_matrix, funda_matrix.T)
            # print(mtrx)
        
        X_3D_set = list()
        psn_set = list()
        P_set = list()
        curr_psn = [0, 0, 0]
        for i in range(len(all_best_matches)-1):
            best_matches1, best_matches2 = np.array_split(all_best_matches[i], 2, axis=1)
            F = all_F[i]
            E = mutils.get_essential_matrix(F, K)
            # print(cv2.recoverPose(E, best_matches1, best_matches2, K))
            _, R, t, _ = cv2.recoverPose(E, best_matches1, best_matches2, K)
            t = t.reshape(1, 3)[0]
            # t  = t / t[-1]
            curr_psn += t
            # print(curr_psn)
            psn_set.append(t)
        
        # print(psn_set)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for psn in psn_set:
            ax.scatter3D(psn[0], psn[1], psn[2], color='red')
        plt.show()
            
        # X_3D_set = list()
        # psn_set = list()
        # P_set = list()
        # for i in range(len(all_frames)-1):
        #     try:
        #         if i == 0:
        #             C_init = np.transpose([0, 0, 0])
        #             R_init = np.identity(3)
        #             P1 = build_projection_matrix(R_init, C_init)
        #             best_matches1, best_matches2 = np.array_split(all_best_matches[i], 2, axis=1)
        #             F = all_F[i]
        #             curr_P, best_config, best_3d_pts = build_world(F, best_matches1, best_matches2, P1)
        #             X_3D_set.append(best_3d_pts)
        #             psn_set.append(best_config)
        #             P_set.append(curr_P)
                    
        #         else:
        #             P1 = P_set[i-1]
        #             best_matches1, best_matches2 = np.array_split(all_best_matches[i], 2, axis=1)
        #             F = all_F[i]
        #             curr_P, best_config, best_3d_pts = build_world(F, best_matches1, best_matches2, P1)
        #             X_3D_set.append(best_3d_pts)
        #             psn_set.append(best_config)
        #             P_set.append(curr_P)
        #     except:
        #         print("Detected a Singular Matrix!")
        #         continue
        # print(len(X_3D_set), len(psn_set), len(P_set))
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # for X_3D in X_3D_set:
        #     for i in range(len(X_3D)):
        #         ax.scatter3D(X_3D[i][0], X_3D[i][1], X_3D[i][2], color = "green")
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.title("World Coordinates")
        
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # for psn in psn_set:
        #     print(psn)
        # #     for i in range(len(psn)):
        # #         print(translation)
        # #         ax.scatter3D(translation[0], translation[1], translation[2], color='red')
        # # plt.show()
            
    else:    
        cap = cv2.VideoCapture(input_path)
        all_frames = list()
        ret = True
        count = 0
        disp_duration = 3
        while ret:
            ret, frame = cap.read()
            if ret:
                all_frames.append(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
            else:
                break
        
        all_frames = all_frames[:10]

        points_master = list()
        all_F = list()
        # reqd_frames = list()
        for i in range(0, len(all_frames)-10):
            if i % 1 == 0:
                print(i)
                frame1, frame2 = all_frames[i], all_frames[i+1]
                pt_set1, pt_set2 = get_orb_matches(frame1, frame2)
                F, best_idxs = mutils.get_inliers(pt_set1, pt_set2, n_iterations=500, error_thresh=0.005)
                # F,mask = cv2.findFundamentalMat(pt_set1, pt_set2, cv2.FM_RANSAC)
                # best_pts1 = pt_set1[mask.ravel() == 1]
                # best_pts2 = pt_set2[mask.ravel() == 1]
                
                # reqd_frames.append(all_frames[i])
                best_pts1, best_pts2 = pt_set1[best_idxs], pt_set2[best_idxs]
                points_master.append(np.hstack((best_pts1, best_pts2)))
                all_F.append(F)
        
        array_path1 = 'Output_Files/Frames.npy'
        array_path2 = 'Output_Files/matches.npy'
        array_path3 = 'Output_Files/fundamental_matrices.npy'
        # miscutils.save_np_array(reqd_frames, array_path1)
        miscutils.save_np_array(points_master, array_path2)
        miscutils.save_np_array(all_F, array_path3)
            

if __name__ == "__main__":
    main()