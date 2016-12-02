//
//  helper.cpp
//  eos
//
//  Created by Vidit Singh on 18/10/16.
//
//

#include "helper.hpp"
#include <iomanip>
#include <sys/stat.h>
#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;
using std::string;
using std::vector;
using cv::Vec4f;
using cv::Vec2f;
using cv::Mat;
using std::cout;
using std::endl;


string createString(string base, int indx, int lead_zeros = 4){
    std::ostringstream ss;
    ss << base << "_" << std::setfill('0') << std::setw(lead_zeros) << indx << ".jpg";
    return ss.str();
    
}

bool fileExists (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}


int main(int argc, char *argv[]){
    
    
    vector<Mat> affine_camera_matrices;
    vector<vector<Vec2f>> landmarks_set;
    vector<vector<int>> vertex_ids_set;
    int right[16] = {1,2,3,19,20,21,22,23,24,31,32,33,41,42,43,48};
    int left[16] = {7,8,9,25,26,27,28,29,30,35,36,37,38,39,45,46};
    vector<vector<int>> drop_landmarks_set;
    
    
    string modelfile("/Users/vidit/Semester3/Project/FittingModel/eos/share/sfm_shape_3448.bin");
    string mappingsfile("/Users/vidit/Semester3/Project/FittingModel/eos/share/ibug2did.txt");
    
    //    fs::path parent_dir("/Users/vidit/Semester3/Project/TestImages/Leonardo_Di_Caprio");
    //    const fs::path parent_dir("/Users/vidit/Semester3/Project/TestImages/Christian_Bale");
    //    fs::path parent_dir("/Users/vidit/Semester3/Project/TestImages/Laughing");
    
    fs::path parent_dir("/Users/vidit/Semester3/Project/test");
    fs::path imagelist = parent_dir / "single_face_images.txt";
    
    //if(!fs::exists(parent_dir/"models"))
    fs::create_directory(parent_dir/"models");
    std::ifstream labels;
    labels.open(parent_dir.string() + "/labels.txt");
    
    std::ofstream csvfile;
    csvfile.open(parent_dir.string() + "/test_alpha_coeff.csv");
    csvfile << "category" << ",";
    for (int i = 1; i<=63; i++) {
        csvfile << "feat" << i << ","  ;
    }
    csvfile<<"\n";
    int cat = 0;
    string label;
    
    while (std::getline(labels, label)) {
        
        
        
        int num_images = 15;
        for (int i = 11; i <= num_images; i++) {
            
            
            fs::path imagefile = parent_dir / "images" / createString(label, i);
            
            fs::path landmarksfile = parent_dir / "landmarks" / fs::path(imagefile.stem().string()+  "_0.pts");
            
            fs::path outputfile = parent_dir / "models" / fs::path(imagefile.stem().string()+"_out");
            
            
            if(fileExists(landmarksfile.string())){
                
                
                
                std::shared_ptr<model_data> data = fit_model(modelfile, imagefile.string(), landmarksfile.string(), mappingsfile, outputfile.string());
                
                if (data->yaw_angle < -30) {
                    vector<int> v(right, right + sizeof right / sizeof right[0]);
                    drop_landmarks_set.push_back(v);
                }else if(data->yaw_angle > 30){
                    vector<int> v(left, left + sizeof left / sizeof left[0]);
                    drop_landmarks_set.push_back(v);
                }else{
                    vector<int> v;
                    drop_landmarks_set.push_back(v);
                }
                csvfile << cat << ",";
                for(int i = 0; i < data->fitted_coeffs.size(); i++)
                    csvfile << data->fitted_coeffs[i] << "," ;
                csvfile << "\n";
                
                affine_camera_matrices.push_back(data->affine_from_ortho);
                landmarks_set.push_back(data->image_points);
                vertex_ids_set.push_back(data->vertex_indices);
            }
            
            
            
        }
        
        
        Helper process;
        vector<float> alphas =  process.regressToFitPoses(eos::morphablemodel::load_model(modelfile), affine_camera_matrices, landmarks_set, vertex_ids_set, drop_landmarks_set);
//        csvfile << cat << ",";
//        for(int i = 0; i < alphas.size(); i++)
//            csvfile << alphas[i] << "," ;
//        csvfile << "\n";
        
        eos::render::Mesh mesh = eos::morphablemodel::load_model(modelfile).draw_sample(alphas, vector<float>());
        
        fs::path final_model = parent_dir/"models"/(label+".obj");
        eos::render::write_textured_obj(mesh, final_model.string());
       
        drop_landmarks_set.clear();
        affine_camera_matrices.clear();
        landmarks_set.clear();
        vertex_ids_set.clear();
        
        
        cat++;
    }
    
    csvfile.close();
}



vector<float> Helper::regressToFitPoses(eos::morphablemodel::MorphableModel morphable_model, vector<Mat> affine_camera_matrices, vector< vector<Vec2f> > landmarks_set, vector<vector<int>> vertex_ids_set,vector<vector<int>>  drop_landmarks_set, Mat base_face, float lambda){
    
    int num_poses =  affine_camera_matrices.size();
    int num_coeffs_to_fit = morphable_model.get_shape_model().get_num_principal_components();
    int num_landmarks_ =  landmarks_set[0].size();
    
    
    
    Mat Af = Mat::zeros(num_coeffs_to_fit,num_coeffs_to_fit, CV_32FC1);
    Mat bf = Mat::zeros(num_coeffs_to_fit, 1, CV_32FC1);
    
    for (int count = 0 ; count < num_poses; count++) {
        
        Mat affine_camera_matrix = affine_camera_matrices[count];
        vector<Vec2f> landmarks = landmarks_set[count];
        vector<int> vertex_ids = vertex_ids_set[count];
        
        int num_landmarks = static_cast<int>(landmarks.size());
        
        if (base_face.empty())
        {
            base_face = morphable_model.get_shape_model().get_mean();
        }
        
        // $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
        // And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
        Mat V_hat_h = Mat::zeros(4 * num_landmarks, num_coeffs_to_fit, CV_32FC1);
        int row_index = 0;
        for (int i = 0; i < num_landmarks; ++i) {
            Mat basis_rows = morphable_model.get_shape_model().get_normalised_pca_basis(vertex_ids[i]); // In the paper, the not-normalised basis might be used? I'm not sure, check it. It's even a mess in the paper. PH 26.5.2014: I think the normalised basis is fine/better.
            //basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
            basis_rows.colRange(0, num_coeffs_to_fit).copyTo(V_hat_h.rowRange(row_index, row_index + 3));
            row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
        }
        // Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
        Mat P = Mat::zeros(3 * num_landmarks, 4 * num_landmarks, CV_32FC1);
        for (int i = 0; i < num_landmarks; ++i) {
            Mat submatrix_to_replace = P.colRange(4 * i, (4 * i) + 4).rowRange(3 * i, (3 * i) + 3);
            affine_camera_matrix.copyTo(submatrix_to_replace);
        }
        // The variances: Add the 2D and 3D standard deviations.
        // If the user doesn't provide them, we choose the following:
        // 2D (detector) standard deviation: In pixel, we follow [1] and choose sqrt(3) as the default value.
        // 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different variance for different vertices.
        // The 3D variance has to be projected to 2D (for details, see paper [1]) so the units do match up.
        float sigma_squared_2D = std::pow(std::sqrt(3.0f), 2) + std::pow((0.0f), 2);
        Mat Sigma = Mat::zeros(3 * num_landmarks, 3 * num_landmarks, CV_32FC1);
        for (int i = 0; i < 3 * num_landmarks; ++i) {
            Sigma.at<float>(i, i) = 1.0f / std::sqrt(sigma_squared_2D); // the higher the sigma_squared_2D, the smaller the diagonal entries of Sigma will be
        }
        Mat Omega = Sigma.t() * Sigma; // just squares the diagonal
        // The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
        Mat y = Mat::ones(3 * num_landmarks, 1, CV_32FC1);
        for (int i = 0; i < num_landmarks; ++i) {
            y.at<float>(3 * i, 0) = landmarks[i][0];
            y.at<float>((3 * i) + 1, 0) = landmarks[i][1];
            //y.at<float>((3 * i) + 2, 0) = 1; // already 1, stays (homogeneous coordinate)
        }
        // The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
        Mat v_bar = Mat::ones(4 * num_landmarks, 1, CV_32FC1);
        for (int i = 0; i < num_landmarks; ++i) {
            //cv::Vec4f model_mean = morphable_model.get_shape_model().get_mean_at_point(vertex_ids[i]);
            cv::Vec4f model_mean(base_face.at<float>(vertex_ids[i] * 3), base_face.at<float>(vertex_ids[i] * 3 + 1), base_face.at<float>(vertex_ids[i] * 3 + 2), 1.0f);
            v_bar.at<float>(4 * i, 0) = model_mean[0];
            v_bar.at<float>((4 * i) + 1, 0) = model_mean[1];
            v_bar.at<float>((4 * i) + 2, 0) = model_mean[2];
            //v_bar.at<float>((4 * i) + 3, 0) = 1; // already 1, stays (homogeneous coordinate)
            // note: now that a Vec4f is returned, we could use copyTo?
        }
        // Bring into standard regularised quadratic form with diagonal distance matrix Omega
        Mat A = P * V_hat_h; // camera matrix times the basis
        Mat b = P * v_bar - y; // camera matrix times the mean, minus the landmarks.
        //Mat c_s; // The x, we solve for this! (the variance-normalised shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$
        //int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
        
        if (drop_landmarks_set[count].size() > 0) {
            
            Mat drop_landmark = Mat::ones(3 * num_landmarks, 3 * num_landmarks, CV_32FC1);
            Mat zeros = Mat::zeros(3, 3, CV_32FC1);
            
            for (auto i : drop_landmarks_set[count]) {
                Mat submatrix_to_replace = drop_landmark.colRange(3 * i, (3 * i) + 3).rowRange(3 * i, (3 * i) + 3);
                zeros.copyTo(submatrix_to_replace);
            }
            cout << cv::determinant(drop_landmark) << endl;
            A = drop_landmark * A;
            b = drop_landmark * b;
        }
        
        Mat AtOmegaA = A.t() * Omega * A;
        Af += AtOmegaA;
        bf += -A.t() * Omega.t() * b;
        
    }
    
    const int num_shape_pc = num_coeffs_to_fit;
    Af = Af + num_poses* lambda * Mat::eye(num_shape_pc, num_shape_pc, CV_32FC1);
    
    
    // Solve using OpenCV:
    Mat c_s; // Note/Todo: We get coefficients ~ N(0, sigma) I think. They are not multiplied with the eigenvalues.
    bool non_singular = cv::solve(Af, bf, c_s, cv::DECOMP_SVD); // DECOMP_SVD calculates the pseudo-inverse if the matrix is not invertible.
    
    return std::vector<float>(c_s);
    
    
    
}

void otherfunc(){
    //    eos::render::Mesh mesh = data->mesh;
    //    vector<Vec4f> vertices =  mesh.vertices;
    //    vector<Vec4f> landmark_vertices ;
    //    vector<Vec4f> pca_basis;
    //    for(auto indx : data->vertex_indices){
    //        landmark_vertices.push_back(mesh.vertices.at(indx));
    //        cout << data->shape_model.get_normalised_pca_basis().size()<< endl;
    //        pca_basis.push_back(data->shape_model.get_normalised_pca_basis(indx));
    //
    //    }
    //
    //    std::cout << "No. of Mesh Vertices: " <<mesh.vertices.size() << std::endl;
    //    std::cout << "No. of landmarks: " <<landmark_vertices.size() << std::endl;
    //
    //    std::cout << "3D to 2D mat size" << cv::Mat(landmark_vertices[0]).size() <<std::endl;
    //    std::cout << "affine mat size" << data->affine_from_ortho.size() <<std::endl;
    //
    //    int indx = 0;
    //    Mat imagepoint_t  = Mat(landmark_vertices.size(),2,CV_32FC1);
    //    Mat imagepoints2D = Mat(landmark_vertices.size(),3,CV_32FC1);
    //
    //    for (auto val: landmark_vertices){
    //
    //
    //        Mat value = data->affine_from_ortho * Mat(val);
    //        imagepoints2D.row(indx) = value.t();
    //        imagepoint_t.row(indx) = Mat(data->image_points[indx]).t();
    //
    //        indx++;
    //    }
    //
    //    imagepoints2D.col(0) /= imagepoints2D.col(2);
    //    imagepoints2D.col(1) /= imagepoints2D.col(2);
    //
    //    imagepoints2D =  imagepoints2D(cv::Range(0,landmark_vertices.size()),cv::Range(0,2));
    //
    //    Mat dist = imagepoints2D-imagepoint_t;
    //
    //
    //
    //    dist  = dist.mul(dist);
    //    cv::reduce(dist, dist, 1, CV_REDUCE_SUM);
    //    cv::sqrt(dist, dist);
    //    cout << dist << endl;
    
    
    
    //
    //    //fs::path parent_dir("/Users/vidit/Semester3/Project/TestImages/Leonardo_Di_Caprio");
    //    fs::path parent_dir("/Users/vidit/Semester3/Project/TestImages/Christian_Bale");
    //    //    fs::path parent_dir("/Users/vidit/Semester3/Project/TestImages/Laughing");
    //    //if(!fs::exists(parent_dir/"models"))
    //    fs::create_directory(parent_dir/"models");
    //
    //    int num_images = 5;
    //    for (int i = 1; i <= num_images; i++) {
    //
    //
    //        fs::path imagefile = parent_dir / "images" / createString(parent_dir.stem().string(), i);
    //
    //        fs::path landmarksfile = parent_dir / "landmarks" / fs::path(imagefile.stem().string()+  "_0.pts");
    //
    //        fs::path outputfile = parent_dir / "models" / fs::path(imagefile.stem().string()+"_out");
    //
    //
    //        if(i!=4){
    //
    //
    //
    //            std::shared_ptr<model_data> data = fit_model(modelfile, imagefile.string(), landmarksfile.string(), mappingsfile, outputfile.string());
    //
    //            if (data->yaw_angle < -30) {
    //                vector<int> v(right, right + sizeof right / sizeof right[0]);
    //                drop_landmarks_set.push_back(v);
    //            }else if(data->yaw_angle > 30){
    //                vector<int> v(left, left + sizeof left / sizeof left[0]);
    //                drop_landmarks_set.push_back(v);
    //            }else{
    //                vector<int> v;
    //                drop_landmarks_set.push_back(v);
    //            }
    //            
    //            affine_camera_matrices.push_back(data->affine_from_ortho);
    //            landmarks_set.push_back(data->image_points);
    //            vertex_ids_set.push_back(data->vertex_indices);
    //        }
    //        
    //        
    //        
    //    }
    
    
    
}
