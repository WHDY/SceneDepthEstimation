#include<iostream>
#include<string>
#include<vector>
#include<cmath>
#include<opencv2\opencv.hpp>
#include<opencv2\core\core.hpp>
#include<opencv2\core\utility.hpp>
#include<opencv2\imgcodecs\imgcodecs.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;


//#define ImagePath ".\\left\\"
//#define ImagePathL "C:\\Users\\lenovo\\Desktop\\stereo\\left\\"
//#define ImagePathR "C:\\Users\\lenovo\\Desktop\\stereo\\right\\"
#define ImagePathL "C:\\Users\\lenovo\\Desktop\\stereo\\calibration4\\left\\"
#define ImagePathR "C:\\Users\\lenovo\\Desktop\\stereo\\calibration4\\right\\"
#define BoardWith  9
#define BoardHeight  6
#define SquareSize  0.024
#define OutputFileName ".\\calibrate_result.xml"


/*   find the coordinates of inner corrorners in images;   */
void FindImageCoordinates(vector<string>& image_list, vector<vector<Point2f>>& image_points, Size board_size, Size& image_size) {
	for (int i = 0; i < image_list.size(); ++i) {
		Mat image;
		image = imread(image_list[i], IMREAD_GRAYSCALE);

		if (!image.empty()) {
			image_size = image.size();

			vector<Point2f> pointbuf;
			bool find = findChessboardCorners(image, board_size, pointbuf,
				CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

			if (find) {
				cornerSubPix(image, pointbuf, Size(11, 11),
					Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
				image_points.push_back(pointbuf);
				drawChessboardCorners(image, board_size, Mat(pointbuf), find);
				//imshow("inner corroners", image);
				int test = 0; //for test 
			}
		}

	}
	return;
}


/*   calculate the world coordinates of inner corroners of board.   */
void CalculateWorldCoordinates(Size board_size, float square_size, vector<vector<Point3f>>& object_points ) {
	for (int i = 0; i < board_size.height; ++i) {
		for (int j = 0; j < board_size.width; ++j)
			object_points[0].push_back(Point3f(i*square_size, j*square_size, 0));
	}
	return;
}


/*   Implement zhang's method  by myself   */
void Calibrate(vector<vector<Point3f>>& object_points, vector<vector<Point2f>>& img_points, \
					Size board_size, Mat& intrinsic_matrix, Mat& distcoffs, vector<Mat>& rvecs, vector<Mat>& tvecs)
{
	/*   first step, calculate the Homography of each image.   */
	Mat b = Mat::zeros(Size(1,object_points[0].size() * 2),CV_64F);
	Mat A = Mat::zeros(Size(8,object_points[0].size() * 2),CV_64F);
	vector<Mat> H;
	int numberOfImage = img_points.size();
	for ( int i = 0; i < numberOfImage; ++i )
	{
		int numberOfPoints = img_points[i].size();
		for (int j = 0; j < numberOfPoints; ++j)
		{
			b.ptr<double>(2 * j)[0] = img_points[i][j].x;
			b.ptr<double>(2 * j+1)[0] = img_points[i][j].y;
			
			A.ptr<double>(2*j)[0] = object_points[i][j].x;
			A.ptr<double>(2*j)[1] = object_points[i][j].y;
			A.ptr<double>(2*j)[2] = 1;
			A.ptr<double>(2*j)[6] = -object_points[i][j].x * img_points[i][j].x;
			A.ptr<double>(2*j)[7] = -object_points[i][j].y * img_points[i][j].x;

			A.ptr<double>(2*j+1)[3] = object_points[i][j].x;
			A.ptr<double>(2*j+1)[4] = object_points[i][j].y;
			A.ptr<double>(2*j+1)[5] = 1;
			A.ptr<double>(2*j+1)[6] = -object_points[i][j].x * img_points[i][j].y;
			A.ptr<double>(2*j+1)[7] = -object_points[i][j].y * img_points[i][j].y;
		}

		Mat h = Mat(Size(1, 8), CV_64F);
		Mat U, W, VT;
		SVD::compute(A, W, U, VT, SVD::MODIFY_A | SVD::FULL_UV);

		Mat w = Mat::zeros(Size(8, 8), CV_64F);
		for (int k = 0; k < 8; k++)
			w.ptr<double>(k)[k] = W.ptr<double>(k)[0];
		Mat Un = U(Range::all(), Range::Range(0,8));
		Mat UnT = Un.t();

		Mat tempt = w*VT;
		tempt = tempt.inv();
		h = tempt*UnT*b;
		H.push_back(h);
	}

	/*   second step, get the closed-form solution     */
	Mat V = Mat::zeros(Size(6, numberOfImage * 3), CV_64F);
	double lamda = 1, gama = 0;
	double a,u0, v0, fx, fy;
	for (int i = 0; i < numberOfImage; ++i)
	{
		
		V.ptr<double>(i*3)[0] = H[i].ptr<double>(0)[0] * H[i].ptr<double>(1)[0];
		V.ptr<double>(i*3)[1] = H[i].ptr<double>(0)[0] * H[i].ptr<double>(4)[0] + H[i].ptr<double>(3)[0] * H[i].ptr<double>(1)[0];
		V.ptr<double>(i*3)[2] = H[i].ptr<double>(3)[0] * H[i].ptr<double>(4)[0];
		V.ptr<double>(i*3)[3] = H[i].ptr<double>(6)[0] * H[i].ptr<double>(1)[0] + H[i].ptr<double>(0)[0] * H[i].ptr<double>(7)[0];
		V.ptr<double>(i*3)[4] = H[i].ptr<double>(3)[0] * H[i].ptr<double>(7)[0] + H[i].ptr<double>(6)[0] * H[i].ptr<double>(4)[0];
		V.ptr<double>(i*3)[5] = H[i].ptr<double>(6)[0] * H[i].ptr<double>(7)[0] ;

		V.ptr<double>(i*3+1)[0] = H[i].ptr<double>(0)[0] * H[i].ptr<double>(0)[0] - H[i].ptr<double>(1)[0] * H[i].ptr<double>(1)[0];
		V.ptr<double>(i*3+1)[1] = H[i].ptr<double>(0)[0] * H[i].ptr<double>(3)[0]*2 - H[i].ptr<float>(1)[0] * H[i].ptr<double>(4)[0] * 2;
		V.ptr<double>(i*3+1)[2] = H[i].ptr<double>(3)[0] * H[i].ptr<double>(3)[0] - H[i].ptr<double>(4)[0] * H[i].ptr<double>(4)[0];
		V.ptr<double>(i*3+1)[3] = H[i].ptr<double>(6)[0] * H[i].ptr<double>(0)[0]*2 - H[i].ptr<double>(7)[0] * H[i].ptr<double>(1)[0] * 2;
		V.ptr<double>(i*3+1)[4] = H[i].ptr<double>(3)[0] * H[i].ptr<double>(6)[0]*2 - H[i].ptr<double>(4)[0] * H[i].ptr<double>(7)[0] * 2;
		V.ptr<double>(i*3+1)[5] = H[i].ptr<double>(6)[0] * H[i].ptr<double>(6)[0] - H[i].ptr<double>(7)[0] * H[i].ptr<double>(7)[0];

		V.ptr<double>(i * 3 + 2)[1] = 1;
	}
	Mat U, W, VT;
	SVD::compute(V, W, U, VT, SVD::MODIFY_A | SVD::FULL_UV);
	VT = VT.t();
	Mat B = VT(Range::all(), Range::Range(5, 6));

/*
	v0 = (B.ptr<double>(1)[0] * B.ptr<double>(3)[0] - B.ptr<double>(0)[0] * B.ptr<double>(4)[0]) / (B.ptr<double>(0)[0] * B.ptr<double>(2)[0] - B.ptr<double>(1)[0] * B.ptr<double>(1)[0]);
	lamda = B.ptr<double>(5)[0] - (B.ptr<double>(3)[0] * B.ptr<double>(3)[0] + v0*(B.ptr<double>(1)[0] * B.ptr<double>(3)[0] - B.ptr<double>(0)[0] * B.ptr<double>(4)[0])) / B.ptr<double>(0)[0];
	fx = sqrt(lamda / B.ptr<double>(0)[0]);
	fy = sqrt(lamda*B.ptr<double>(0)[0] / (B.ptr<double>(0)[0] * B.ptr<double>(2)[0] - B.ptr<double>(1)[0] * B.ptr<double>(1)[0]));
	gama = -B.ptr<double>(1)[0] * fx*fx*fy / lamda;
	u0 = gama*v0 / fx - B.ptr<double>(3)[0] * fx*fx / lamda;
*/

	u0 = -B.ptr<double>(3)[0] / B.ptr<double>(0)[0];
	v0 = -B.ptr<double>(4)[0] / B.ptr<double>(2)[0];
	a = B.ptr<double>(5)[0] + u0*B.ptr<double>(3)[0] + v0*B.ptr<double>(4)[0];
	fx = sqrt(a / B.ptr<double>(0)[0]);
	fy = sqrt(a / B.ptr<double>(2)[0]);

	intrinsic_matrix.ptr<double>(0)[0] = fx;
	intrinsic_matrix.ptr<double>(1)[1] = fy;
	intrinsic_matrix.ptr<double>(0)[2] = u0;
	intrinsic_matrix.ptr<double>(1)[2] = v0;
	intrinsic_matrix.ptr<double>(0)[1] = gama;

	double lamda_1, lamda_2;
	Mat M_inv = intrinsic_matrix.inv();
	Mat h1 = Mat::zeros(Size(1, 3), CV_64F);
	Mat h2 = Mat::zeros(Size(1, 3), CV_64F);
	Mat h3 = Mat::zeros(Size(1, 3), CV_64F);
	Mat R,r1, r2, r3;
	for (int i = 0; i < numberOfImage; ++i)
	{
		h1.ptr<double>(0)[0] = H[i].ptr<double>(0)[0];
		h2.ptr<double>(0)[0] = H[i].ptr<double>(1)[0];
		h3.ptr<double>(0)[0] = H[i].ptr<double>(2)[0];
		h1.ptr<double>(1)[0] = H[i].ptr<double>(3)[0];
		h2.ptr<double>(1)[0] = H[i].ptr<double>(4)[0];
		h3.ptr<double>(1)[0] = H[i].ptr<double>(5)[0];
		h1.ptr<double>(2)[0] = H[i].ptr<double>(6)[0];
		h2.ptr<double>(2)[0] = H[i].ptr<double>(7)[0];
		h3.ptr<double>(2)[0] = 1;

		Mat temp,tempt1, tempt2,t;
		tempt1 = M_inv*h1;
		tempt2 = M_inv*h2;
		normalize(tempt1, r1);
		normalize(tempt2, r2);
		r3 = r1.cross(r2);
		lamda_1 = r1.ptr<double>(0)[0] / tempt1.ptr<double>(0)[0];
		lamda_2 = r1.ptr<double>(0)[0] / tempt1.ptr<double>(0)[0];
		t = (lamda_1 + lamda_2) / 2 * M_inv*h3;
		hconcat(r1, r2, R);
		hconcat(R, r3, R);

		Rodrigues(R, temp);
		rvecs.push_back(temp);
		tvecs.push_back(t);
	}	
	
	/*   Third step, refine the parameters with LM algorithm   */
	int numberOfPointsPerImage = board_size.height * board_size.width;
	vector<Mat> rotation_vectors(numberOfImage);
	vector<double> rotation_angles(numberOfImage);
	for (int i = 0; i < numberOfImage; ++i) {
		normalize(rvecs[i], rotation_vectors[i]);
		rotation_angles[i] = rvecs[i].ptr<double>(0)[0] / rotation_vectors[i].ptr<double>(0)[0];
	}

	Mat parameters = Mat::zeros(Size(1, 4 + 4 + 7 * numberOfImage), CV_64F);
	Mat Jacobian = Mat::zeros(Size(4 + 4 + 7 * numberOfImage, 2 * numberOfPointsPerImage * numberOfImage), CV_64F);
	Mat JTJ, G, F, L, hlm;
	Mat f = Mat::zeros(Size(1, 2 * numberOfPointsPerImage * numberOfImage), CV_64F);
	double rho;

	//initial parameters
	parameters.ptr<double>(0)[0] = intrinsic_matrix.ptr<double>(0)[0];
	parameters.ptr<double>(1)[0] = intrinsic_matrix.ptr<double>(1)[1];
	parameters.ptr<double>(2)[0] = intrinsic_matrix.ptr<double>(0)[2];
	parameters.ptr<double>(3)[0] = intrinsic_matrix.ptr<double>(1)[2];
	for (int i = 8; i < parameters.rows; i = i + 7) {
		parameters.ptr<double>(i)[0] = rotation_vectors[(i - 8) / 7].ptr<double>(0)[0];
		parameters.ptr<double>(i + 1)[0] = rotation_vectors[(i - 8) / 7].ptr<double>(1)[0];
		parameters.ptr<double>(i + 2)[0] = rotation_vectors[(i - 8) / 7].ptr<double>(2)[0];
		parameters.ptr<double>(i + 3)[0] = rotation_angles[(i - 8) / 7];
		parameters.ptr<double>(i + 4)[0] = tvecs[(i - 8) / 7].ptr<double>(0)[0];
		parameters.ptr<double>(i + 5)[0] = tvecs[(i - 8) / 7].ptr<double>(1)[0];
		parameters.ptr<double>(i + 6)[0] = tvecs[(i - 8) / 7].ptr<double>(2)[0];
	}

	//initial Jacobian matrix and Gradient matrix
	int points[54];
	for (int i = 0; i <  board_size.height * board_size.width; ++i)
		points[i] = i;

	int order = 0;
	for (int i = 0; i < numberOfImage; i = ++i) {
		for (int j = 0; j <  numberOfPointsPerImage; ++j) {
			double xw = object_points[i][points[j]].x;
			double yw = object_points[i][points[j]].y;
			double u_t = img_points[i][points[j]].x;
			double v_t = img_points[i][points[j]].y;

			double fx = parameters.ptr<double>(0)[0];
			double fy = parameters.ptr<double>(1)[0];
			double u0 = parameters.ptr<double>(2)[0];
			double v0 = parameters.ptr<double>(3)[0];
			double k1 = parameters.ptr<double>(4)[0];
			double k2 = parameters.ptr<double>(5)[0];
			double p1 = parameters.ptr<double>(6)[0];
			double p2 = parameters.ptr<double>(7)[0];
			double kx = parameters.ptr<double>(i * 7 + 8)[0];
			double ky = parameters.ptr<double>(i * 7 + 9)[0];
			double kz = parameters.ptr<double>(i * 7 + 10)[0];
			double theta = parameters.ptr<double>(i * 7 + 11)[0];
			double tx = parameters.ptr<double>(i * 7 + 12)[0];
			double ty = parameters.ptr<double>(i * 7 + 13)[0];
			double tz = parameters.ptr<double>(i * 7 + 14)[0];
			double r = sqrt(kx*kx + ky*ky + kz*kz);

			double h1 = (kx*kx + cos(theta)*ky*ky + cos(theta)*kz*kz)*xw + (-sin(theta)*kz*r + (1 - cos(theta))*kx*ky)*yw + tx*r*r;
			double h2 = (cos(theta)*kx*kx + ky*ky + cos(theta)*kz*kz)*yw + (sin(theta)*kz*r + (1 - cos(theta))*kx*ky)*xw + ty*r*r;
			double h0 = (-sin(theta)*ky*r + (1 - cos(theta))*kx*kz)*xw + (sin(theta)*kx*r + (1 - cos(theta))*ky*kz)*yw + tz*r*r;

			double x = h1 / h0;
			double y = h2 / h0;
			double R = x*x + y*y;

			double x_s = x + k1*x*R + k2*x*R*R + 2 * p1*x*y + p2*(2 * x*x + R);
			double y_s = y + k1*y*R + k2*y*R*R + 2 * p2*x*y + p1*(2 * y*y + R);

			double u = fx*x_s + u0;
			double v = fy*y_s + v0;

			double ph1_pkx = 2 * xw*kx - sin(theta)*kz*yw*kx / r + (1 - cos(theta))*ky*yw + 2 * tx*kx;
			double ph1_pky = 2 * xw*ky*cos(theta) - sin(theta)*kz*ky*yw / r + (1 - cos(theta))*kx*yw + 2 * tx*ky;
			double ph1_pkz = 2 * cos(theta)*xw*kz - sin(theta)*yw*r - sin(theta)*kz*kz*yw / r + 2 * tx*kz;
			double ph1_ptheta = -xw*ky*ky*sin(theta) - xw*kz*kz*sin(theta) - cos(theta)*kz*r*yw + kx*ky*yw*sin(theta);
			double ph1_ptx = r*r;
			double ph1_pty = 0;
			double ph1_ptz = 0;

			double ph2_pkx = 2 * cos(theta)*yw*kx + sin(theta)*kz*kx*xw / r + (1 - cos(theta))*xw*ky + 2 * ty*kx;
			double ph2_pky = 2 * yw*ky + sin(theta)*kz*ky*xw / r + (1 - cos(theta))*kx*xw + 2 * ty*ky;
			double ph2_pkz = 2 * cos(theta)*yw*kz + sin(theta)*xw*r + sin(theta)*xw*kz*kz / r + 2 * ty*kz;
			double ph2_ptheta = -yw*kx*kx*sin(theta) - yw*kz*kz*sin(theta) + xw*cos(theta)*kz*r + xw*sin(theta)*kx*ky;
			double ph2_ptx = 0;
			double ph2_pty = r*r;
			double ph2_ptz = 0;

			double ph0_pkx = -sin(theta)*xw*ky*kx / r + (1 - cos(theta))*xw*kz + sin(theta)*yw*r + sin(theta)*kx*kx*yw / r + 2 * tz*kx;
			double ph0_pky = -sin(theta)*r*xw - sin(theta)*ky*ky*xw / r + sin(theta)*kx*ky*yw / r + (1 - cos(theta))*kz*yw + 2 * tz*ky;
			double ph0_pkz = -sin(theta)*ky*xw*kz / r + (1 - cos(theta))*kx*xw + sin(theta)*kx*yw*kz / r + (1 - cos(theta))*ky*yw + 2 * tz*kz;
			double ph0_ptheta = -cos(theta)*ky*r*xw + sin(theta)*kx*kz*xw + cos(theta)*kx*r*yw + sin(theta)*ky*kz*yw;
			double ph0_ptx = 0;
			double ph0_pty = 0;
			double ph0_ptz = r*r;

			double px_pkx = (ph1_pkx*h0 - ph0_pkx*h1) / (h0*h0);
			double py_pkx = (ph2_pkx*h0 - ph0_pkx*h2) / (h0*h0);
			double px_pky = (ph1_pky*h0 - ph0_pky*h1) / (h0*h0);
			double py_pky = (ph2_pky*h0 - ph0_pky*h2) / (h0*h0);
			double px_pkz = (ph1_pkz*h0 - ph0_pkz*h1) / (h0*h0);
			double py_pkz = (ph2_pkz*h0 - ph0_pkz*h2) / (h0*h0);
			double px_ptheta = (ph1_ptheta*h0 - ph0_ptheta*h1) / (h0*h0);
			double py_ptheta = (ph2_ptheta*h0 - ph0_ptheta*h2) / (h0*h0);
			double px_ptx = (ph1_ptx*h0 - ph0_ptx*h1) / (h0*h0);
			double py_ptx = (ph2_ptx*h0 - ph0_ptx*h2) / (h0*h0);
			double px_pty = (ph1_pty*h0 - ph0_pty*h1) / (h0*h0);
			double py_pty = (ph2_pty*h0 - ph0_pty*h2) / (h0*h0);
			double px_ptz = (ph1_ptz*h0 - ph0_ptz*h1) / (h0*h0);
			double py_ptz = (ph2_ptz*h0 - ph0_ptz*h2) / (h0*h0);


			Jacobian.ptr<double>(order)[0] = x_s;
			Jacobian.ptr<double>(order)[1] = 0;
			Jacobian.ptr<double>(order)[2] = 1;
			Jacobian.ptr<double>(order)[3] = 0;
			Jacobian.ptr<double>(order)[4] = fx*x*R;
			Jacobian.ptr<double>(order)[5] = fx*x*R*R;
			Jacobian.ptr<double>(order)[6] = fx * 2 * x*y;
			Jacobian.ptr<double>(order)[7] = fx*(2 * x*x + R);
			Jacobian.ptr<double>(order)[i * 7 + 8] = fx*(px_pkx + k1*R*px_pkx + k1*x*(2 * x*px_pkx + 2 * y*py_pkx) + k2*R*R*px_pkx + 2 * k2*x*R*(2 * x*px_pkx + 2 * y*py_pkx) + 2 * p1*x*py_pkx + 2 * p1*y*px_pkx + 6 * p2*x*px_pkx + 2 * p2*y*py_pkx);
			Jacobian.ptr<double>(order)[i * 7 + 9] = fx*(px_pky + k1*R*px_pky + k1*x*(2 * x*px_pky + 2 * y*py_pky) + k2*R*R*px_pky + 2 * k2*x*R*(2 * x*px_pky + 2 * y*py_pky) + 2 * p1*x*py_pky + 2 * p1*y*px_pky + 6 * p2*x*px_pky + 2 * p2*y*py_pky);
			Jacobian.ptr<double>(order)[i * 7 + 10] = fx*(px_pkz + k1*R*px_pkz + k1*x*(2 * x*px_pkz + 2 * y*py_pkz) + k2*R*R*px_pkz + 2 * k2*x*R*(2 * x*px_pkz + 2 * y*py_pkz) + 2 * p1*x*py_pkz + 2 * p1*y*px_pkz + 6 * p2*x*px_pkz + 2 * p2*y*py_pkz);
			Jacobian.ptr<double>(order)[i * 7 + 11] = fx*(px_ptheta + k1*R*px_ptheta + k1*x*(2 * x*px_ptheta + 2 * y*py_ptheta) + k2*R*R*px_ptheta + 2 * k2*x*R*(2 * x*px_ptheta + 2 * y*py_ptheta) + 2 * p1*x*py_ptheta + 2 * p1*y*px_ptheta + 6 * p2*x*px_ptheta + 2 * p2*y*py_ptheta);
			Jacobian.ptr<double>(order)[i * 7 + 12] = fx*(px_ptx + k1*R*px_ptx + k1*x*(2 * x*px_ptx + 2 * y*py_ptx) + k2*R*R*px_ptx + 2 * k2*x*R*(2 * x*px_ptx + 2 * y*py_ptx) + 2 * p1*x*py_ptx + 2 * p1*y*px_ptx + 6 * p2*x*px_ptx + 2 * p2*y*py_ptx);
			Jacobian.ptr<double>(order)[i * 7 + 13] = fx*(px_pty + k1*R*px_pty + k1*x*(2 * x*px_pty + 2 * y*py_pty) + k2*R*R*px_pty + 2 * k2*x*R*(2 * x*px_pty + 2 * y*py_pty) + 2 * p1*x*py_pty + 2 * p1*y*px_pty + 6 * p2*x*px_pty + 2 * p2*y*py_pty);
			Jacobian.ptr<double>(order)[i * 7 + 14] = fx*(px_ptz + k1*R*px_ptz + k1*x*(2 * x*px_ptz + 2 * y*py_ptz) + k2*R*R*px_ptz + 2 * k2*x*R*(2 * x*px_ptz + 2 * y*py_ptz) + 2 * p1*x*py_ptz + 2 * p1*y*px_ptz + 6 * p2*x*px_ptz + 2 * p2*y*py_ptz);

			Jacobian.ptr<double>(order + 1)[0] = 0;
			Jacobian.ptr<double>(order + 1)[1] = y_s;
			Jacobian.ptr<double>(order + 1)[2] = 0;
			Jacobian.ptr<double>(order + 1)[3] = 1;
			Jacobian.ptr<double>(order + 1)[4] = fy*R*y;
			Jacobian.ptr<double>(order + 1)[5] = fy*y*R*R;
			Jacobian.ptr<double>(order + 1)[6] = fy*(2 * y*y + R);
			Jacobian.ptr<double>(order + 1)[7] = fy * 2 * x*y;
			Jacobian.ptr<double>(order + 1)[i * 7 + 8] = fy*(py_pkx + k1*R*py_pkx + k1*y*(2 * x*px_pkx + 2 * y*py_pkx) + k2*R*R*py_pkx + 2 * k2*y*R*(2 * x*px_pkx + 2 * y*py_pkx) + 2 * p2*x*py_pkx + 2 * p2*y*px_pkx + 2 * p1*x*px_pkx + 6 * p1*y*py_pkx);
			Jacobian.ptr<double>(order + 1)[i * 7 + 9] = fy*(py_pky + k1*R*py_pky + k1*y*(2 * x*px_pky + 2 * y*py_pky) + k2*R*R*py_pky + 2 * k2*y*R*(2 * x*px_pky + 2 * y*py_pky) + 2 * p2*x*py_pky + 2 * p2*y*px_pky + 2 * p1*x*px_pky + 6 * p1*y*py_pky);
			Jacobian.ptr<double>(order + 1)[i * 7 + 10] = fy*(py_pkz + k1*R*py_pkz + k1*y*(2 * x*px_pkz + 2 * y*py_pkz) + k2*R*R*py_pkz + 2 * k2*y*R*(2 * x*px_pkz + 2 * y*py_pkz) + 2 * p2*x*py_pkz + 2 * p2*y*px_pkz + 2 * p1*x*px_pkz + 6 * p1*y*py_pkz);
			Jacobian.ptr<double>(order + 1)[i * 7 + 11] = fy*(py_ptheta + k1*R*py_ptheta + k1*y*(2 * x*px_ptheta + 2 * y*py_ptheta) + k2*R*R*py_ptheta + 2 * k2*y*R*(2 * x*px_ptheta + 2 * y*py_ptheta) + 2 * p2*x*py_ptheta + 2 * p2*y*px_ptheta + 2 * p1*x*px_ptheta + 6 * p1*y*py_ptheta);
			Jacobian.ptr<double>(order + 1)[i * 7 + 12] = fy*(py_ptx + k1*R*py_ptx + k1*y*(2 * x*px_ptx + 2 * y*py_ptx) + k2*R*R*py_ptx + 2 * k2*y*R*(2 * x*px_ptx + 2 * y*py_ptx) + 2 * p2*x*py_ptx + 2 * p2*y*px_ptx + 2 * p1*x*px_ptx + 6 * p1*y*py_ptx);
			Jacobian.ptr<double>(order + 1)[i * 7 + 13] = fy*(py_pty + k1*R*py_pty + k1*y*(2 * x*px_pty + 2 * y*py_pty) + k2*R*R*py_pty + 2 * k2*y*R*(2 * x*px_pty + 2 * y*py_pty) + 2 * p2*x*py_pty + 2 * p2*y*px_pty + 2 * p1*x*px_pty + 6 * p1*y*py_pty);
			Jacobian.ptr<double>(order + 1)[i * 7 + 14] = fy*(py_ptz + k1*R*py_ptz + k1*y*(2 * x*px_ptz + 2 * y*py_ptz) + k2*R*R*py_ptz + 2 * k2*y*R*(2 * x*px_ptz + 2 * y*py_ptz) + 2 * p2*x*py_ptz + 2 * p2*y*px_ptz + 2 * p1*x*px_ptz + 6 * p1*y*py_ptz);

			f.ptr<double>(order)[0] = u - u_t;
			f.ptr<double>(order + 1)[0] = v - v_t;
			order = order + 2;
		}

	}
	F = 0.5*f.t()*f;
	L = F.clone();
	G = Jacobian.t()*f;
	JTJ = Jacobian.t()*Jacobian;

	//iterate to refinement the parameters
	int k = 0, kmax = 100, kp = 2;
	double e1 = 1e-20;
	double e2 = 1e-20;
	int find = (norm(G, NORM_INF) < e1);
	double u = JTJ.ptr<double>(0)[0];
	for (int i = 1; i < JTJ.rows; ++i) {
		if (u < JTJ.ptr<double>(i)[i])
			u = JTJ.ptr<double>(i)[i];
	}
	u = u*1e-6;
	Mat new_parameters;
	while (!find && k < kmax) {
		k = k + 1;
		Mat temp = JTJ.clone();
		for (int i = 0; i < JTJ.rows; ++i)
			temp.ptr<double>(i)[i] += u;
		hlm = temp.inv()*(-G);

		if (norm(hlm, NORM_L2) < e2*(norm(parameters, NORM_L2) + e2))
			find = true;
		else {
			new_parameters = parameters.clone();
			new_parameters = parameters + hlm;
			Mat new_f = f.clone();
			order = 0;
			for (int i = 0; i < numberOfImage; i = ++i) {
				for (int j = 0; j < numberOfPointsPerImage; ++j) {
					double xw = object_points[i][points[j]].x;
					double yw = object_points[i][points[j]].y;
					double u_t = img_points[i][points[j]].x;
					double v_t = img_points[i][points[j]].y;

					double fx = new_parameters.ptr<double>(0)[0];
					double fy = new_parameters.ptr<double>(1)[0];
					double u0 = new_parameters.ptr<double>(2)[0];
					double v0 = new_parameters.ptr<double>(3)[0];
					double k1 = new_parameters.ptr<double>(4)[0];
					double k2 = new_parameters.ptr<double>(5)[0];
					double p1 = new_parameters.ptr<double>(6)[0];
					double p2 = new_parameters.ptr<double>(7)[0];
					double kx = new_parameters.ptr<double>(i * 7 + 8)[0];
					double ky = new_parameters.ptr<double>(i * 7 + 9)[0];
					double kz = new_parameters.ptr<double>(i * 7 + 10)[0];
					double theta = new_parameters.ptr<double>(i * 7 + 11)[0];
					double tx = new_parameters.ptr<double>(i * 7 + 12)[0];
					double ty = new_parameters.ptr<double>(i * 7 + 13)[0];
					double tz = new_parameters.ptr<double>(i * 7 + 14)[0];
					double r = sqrt(kx*kx + ky*ky + kz*kz);

					double h1 = (kx*kx + cos(theta)*ky*ky + cos(theta)*kz*kz)*xw + (-sin(theta)*kz*r + (1 - cos(theta))*kx*ky)*yw + tx*r*r;
					double h2 = (cos(theta)*kx*kx + ky*ky + cos(theta)*kz*kz)*yw + (sin(theta)*kz*r + (1 - cos(theta))*kx*ky)*xw + ty*r*r;
					double h0 = (-sin(theta)*ky*r + (1 - cos(theta))*kx*kz)*xw + (sin(theta)*kx*r + (1 - cos(theta))*ky*kz)*yw + tz*r*r;

					double x = h1 / h0;
					double y = h2 / h0;
					double R = x*x + y*y;

					double x_s = x + k1*x*R + k2*x*R*R + 2 * p1*x*y + p2*(2 * x*x + R);
					double y_s = y + k1*y*R + k2*y*R*R + 2 * p2*x*y + p1*(2 * y*y + R);

					double u = fx*x_s + u0;
					double v = fy*y_s + v0;

					new_f.ptr<double>(order)[0] = u - u_t;
					new_f.ptr<double>(order + 1)[0] = v - v_t;
					order = order + 2;
				}
			}
			Mat new_F = 0.5*new_f.t()*new_f;
			Mat new_L = F + hlm.t()*Jacobian.t()*f + 0.5*hlm.t()*Jacobian.t()*Jacobian*hlm;
			rho = (F.ptr<double>(0)[0] - new_F.ptr<double>(0)[0]) / (L.ptr<double>(0)[0] - new_L.ptr<double>(0)[0]);
			if (rho > 0) {
				parameters = new_parameters.clone();
				order = 0;
				for (int i = 0; i < numberOfImage; i = ++i) {
					for (int j = 0; j < numberOfPointsPerImage; ++j) {
						double xw = object_points[i][points[j]].x;
						double yw = object_points[i][points[j]].y;
						double u_t = img_points[i][points[j]].x;
						double v_t = img_points[i][points[j]].y;

						double fx = parameters.ptr<double>(0)[0];
						double fy = parameters.ptr<double>(1)[0];
						double u0 = parameters.ptr<double>(2)[0];
						double v0 = parameters.ptr<double>(3)[0];
						double k1 = parameters.ptr<double>(4)[0];
						double k2 = parameters.ptr<double>(5)[0];
						double p1 = parameters.ptr<double>(6)[0];
						double p2 = parameters.ptr<double>(7)[0];
						double kx = parameters.ptr<double>(i * 7 + 8)[0];
						double ky = parameters.ptr<double>(i * 7 + 9)[0];
						double kz = parameters.ptr<double>(i * 7 + 10)[0];
						double theta = parameters.ptr<double>(i * 7 + 11)[0];
						double tx = parameters.ptr<double>(i * 7 + 12)[0];
						double ty = parameters.ptr<double>(i * 7 + 13)[0];
						double tz = parameters.ptr<double>(i * 7 + 14)[0];
						double r = sqrt(kx*kx + ky*ky + kz*kz);

						double h1 = (kx*kx + cos(theta)*ky*ky + cos(theta)*kz*kz)*xw + (-sin(theta)*kz*r + (1 - cos(theta))*kx*ky)*yw + tx*r*r;
						double h2 = (cos(theta)*kx*kx + ky*ky + cos(theta)*kz*kz)*yw + (sin(theta)*kz*r + (1 - cos(theta))*kx*ky)*xw + ty*r*r;
						double h0 = (-sin(theta)*ky*r + (1 - cos(theta))*kx*kz)*xw + (sin(theta)*kx*r + (1 - cos(theta))*ky*kz)*yw + tz*r*r;

						double x = h1 / h0;
						double y = h2 / h0;
						double R = x*x + y*y;

						double x_s = x + k1*x*R + k2*x*R*R + 2 * p1*x*y + p2*(2 * x*x + R);
						double y_s = y + k1*y*R + k2*y*R*R + 2 * p2*x*y + p1*(2 * y*y + R);

						double u = fx*x_s + u0;
						double v = fy*y_s + v0;

						double ph1_pkx = 2 * xw*kx - sin(theta)*kz*yw*kx / r + (1 - cos(theta))*ky*yw + 2 * tx*kx;
						double ph1_pky = 2 * xw*ky*cos(theta) - sin(theta)*kz*ky*yw / r + (1 - cos(theta))*kx*yw + 2 * tx*ky;
						double ph1_pkz = 2 * cos(theta)*xw*kz - sin(theta)*yw*r - sin(theta)*kz*kz*yw / r + 2 * tx*kz;
						double ph1_ptheta = -xw*ky*ky*sin(theta) - xw*kz*kz*sin(theta) - cos(theta)*kz*r*yw + kx*ky*yw*sin(theta);
						double ph1_ptx = r*r;
						double ph1_pty = 0;
						double ph1_ptz = 0;

						double ph2_pkx = 2 * cos(theta)*yw*kx + sin(theta)*kz*kx*xw / r + (1 - cos(theta))*xw*ky + 2 * ty*kx;
						double ph2_pky = 2 * yw*ky + sin(theta)*kz*ky*xw / r + (1 - cos(theta))*kx*xw + 2 * ty*ky;
						double ph2_pkz = 2 * cos(theta)*yw*kz + sin(theta)*xw*r + sin(theta)*xw*kz*kz / r + 2 * ty*kz;
						double ph2_ptheta = -yw*kx*kx*sin(theta) - yw*kz*kz*sin(theta) + xw*cos(theta)*kz*r + xw*sin(theta)*kx*ky;
						double ph2_ptx = 0;
						double ph2_pty = r*r;
						double ph2_ptz = 0;

						double ph0_pkx = -sin(theta)*xw*ky*kx / r + (1 - cos(theta))*xw*kz + sin(theta)*yw*r + sin(theta)*kx*kx*yw / r + 2 * tz*kx;
						double ph0_pky = -sin(theta)*r*xw - sin(theta)*ky*ky*xw / r + sin(theta)*kx*ky*yw / r + (1 - cos(theta))*kz*yw + 2 * tz*ky;
						double ph0_pkz = -sin(theta)*ky*xw*kz / r + (1 - cos(theta))*kx*xw + sin(theta)*kx*yw*kz / r + (1 - cos(theta))*ky*yw + 2 * tz*kz;
						double ph0_ptheta = -cos(theta)*ky*r*xw + sin(theta)*kx*kz*xw + cos(theta)*kx*r*yw + sin(theta)*ky*kz*yw;
						double ph0_ptx = 0;
						double ph0_pty = 0;
						double ph0_ptz = r*r;

						double px_pkx = (ph1_pkx*h0 - ph0_pkx*h1) / (h0*h0);
						double py_pkx = (ph2_pkx*h0 - ph0_pkx*h2) / (h0*h0);
						double px_pky = (ph1_pky*h0 - ph0_pky*h1) / (h0*h0);
						double py_pky = (ph2_pky*h0 - ph0_pky*h2) / (h0*h0);
						double px_pkz = (ph1_pkz*h0 - ph0_pkz*h1) / (h0*h0);
						double py_pkz = (ph2_pkz*h0 - ph0_pkz*h2) / (h0*h0);
						double px_ptheta = (ph1_ptheta*h0 - ph0_ptheta*h1) / (h0*h0);
						double py_ptheta = (ph2_ptheta*h0 - ph0_ptheta*h2) / (h0*h0);
						double px_ptx = (ph1_ptx*h0 - ph0_ptx*h1) / (h0*h0);
						double py_ptx = (ph2_ptx*h0 - ph0_ptx*h2) / (h0*h0);
						double px_pty = (ph1_pty*h0 - ph0_pty*h1) / (h0*h0);
						double py_pty = (ph2_pty*h0 - ph0_pty*h2) / (h0*h0);
						double px_ptz = (ph1_ptz*h0 - ph0_ptz*h1) / (h0*h0);
						double py_ptz = (ph2_ptz*h0 - ph0_ptz*h2) / (h0*h0);


						Jacobian.ptr<double>(order)[0] = x_s;
						Jacobian.ptr<double>(order)[1] = 0;
						Jacobian.ptr<double>(order)[2] = 1;
						Jacobian.ptr<double>(order)[3] = 0;
						Jacobian.ptr<double>(order)[4] = fx*x*R;
						Jacobian.ptr<double>(order)[5] = fx*x*R*R;
						Jacobian.ptr<double>(order)[6] = fx * 2 * x*y;
						Jacobian.ptr<double>(order)[7] = fx*(2 * x*x + R);
						Jacobian.ptr<double>(order)[i * 7 + 8] = fx*(px_pkx + k1*R*px_pkx + k1*x*(2 * x*px_pkx + 2 * y*py_pkx) + k2*R*R*px_pkx + 2 * k2*x*R*(2 * x*px_pkx + 2 * y*py_pkx) + 2 * p1*x*py_pkx + 2 * p1*y*px_pkx + 6 * p2*x*px_pkx + 2 * p2*y*py_pkx);
						Jacobian.ptr<double>(order)[i * 7 + 9] = fx*(px_pky + k1*R*px_pky + k1*x*(2 * x*px_pky + 2 * y*py_pky) + k2*R*R*px_pky + 2 * k2*x*R*(2 * x*px_pky + 2 * y*py_pky) + 2 * p1*x*py_pky + 2 * p1*y*px_pky + 6 * p2*x*px_pky + 2 * p2*y*py_pky);
						Jacobian.ptr<double>(order)[i * 7 + 10] = fx*(px_pkz + k1*R*px_pkz + k1*x*(2 * x*px_pkz + 2 * y*py_pkz) + k2*R*R*px_pkz + 2 * k2*x*R*(2 * x*px_pkz + 2 * y*py_pkz) + 2 * p1*x*py_pkz + 2 * p1*y*px_pkz + 6 * p2*x*px_pkz + 2 * p2*y*py_pkz);
						Jacobian.ptr<double>(order)[i * 7 + 11] = fx*(px_ptheta + k1*R*px_ptheta + k1*x*(2 * x*px_ptheta + 2 * y*py_ptheta) + k2*R*R*px_ptheta + 2 * k2*x*R*(2 * x*px_ptheta + 2 * y*py_ptheta) + 2 * p1*x*py_ptheta + 2 * p1*y*px_ptheta + 6 * p2*x*px_ptheta + 2 * p2*y*py_ptheta);
						Jacobian.ptr<double>(order)[i * 7 + 12] = fx*(px_ptx + k1*R*px_ptx + k1*x*(2 * x*px_ptx + 2 * y*py_ptx) + k2*R*R*px_ptx + 2 * k2*x*R*(2 * x*px_ptx + 2 * y*py_ptx) + 2 * p1*x*py_ptx + 2 * p1*y*px_ptx + 6 * p2*x*px_ptx + 2 * p2*y*py_ptx);
						Jacobian.ptr<double>(order)[i * 7 + 13] = fx*(px_pty + k1*R*px_pty + k1*x*(2 * x*px_pty + 2 * y*py_pty) + k2*R*R*px_pty + 2 * k2*x*R*(2 * x*px_pty + 2 * y*py_pty) + 2 * p1*x*py_pty + 2 * p1*y*px_pty + 6 * p2*x*px_pty + 2 * p2*y*py_pty);
						Jacobian.ptr<double>(order)[i * 7 + 14] = fx*(px_ptz + k1*R*px_ptz + k1*x*(2 * x*px_ptz + 2 * y*py_ptz) + k2*R*R*px_ptz + 2 * k2*x*R*(2 * x*px_ptz + 2 * y*py_ptz) + 2 * p1*x*py_ptz + 2 * p1*y*px_ptz + 6 * p2*x*px_ptz + 2 * p2*y*py_ptz);

						Jacobian.ptr<double>(order + 1)[0] = 0;
						Jacobian.ptr<double>(order + 1)[1] = y_s;
						Jacobian.ptr<double>(order + 1)[2] = 0;
						Jacobian.ptr<double>(order + 1)[3] = 1;
						Jacobian.ptr<double>(order + 1)[4] = fy*R*y;
						Jacobian.ptr<double>(order + 1)[5] = fy*y*R*R;
						Jacobian.ptr<double>(order + 1)[6] = fy*(2 * y*y + R);
						Jacobian.ptr<double>(order + 1)[7] = fy * 2 * x*y;
						Jacobian.ptr<double>(order + 1)[i * 7 + 8] = fy*(py_pkx + k1*R*py_pkx + k1*y*(2 * x*px_pkx + 2 * y*py_pkx) + k2*R*R*py_pkx + 2 * k2*y*R*(2 * x*px_pkx + 2 * y*py_pkx) + 2 * p2*x*py_pkx + 2 * p2*y*px_pkx + 2 * p1*x*px_pkx + 6 * p1*y*py_pkx);
						Jacobian.ptr<double>(order + 1)[i * 7 + 9] = fy*(py_pky + k1*R*py_pky + k1*y*(2 * x*px_pky + 2 * y*py_pky) + k2*R*R*py_pky + 2 * k2*y*R*(2 * x*px_pky + 2 * y*py_pky) + 2 * p2*x*py_pky + 2 * p2*y*px_pky + 2 * p1*x*px_pky + 6 * p1*y*py_pky);
						Jacobian.ptr<double>(order + 1)[i * 7 + 10] = fy*(py_pkz + k1*R*py_pkz + k1*y*(2 * x*px_pkz + 2 * y*py_pkz) + k2*R*R*py_pkz + 2 * k2*y*R*(2 * x*px_pkz + 2 * y*py_pkz) + 2 * p2*x*py_pkz + 2 * p2*y*px_pkz + 2 * p1*x*px_pkz + 6 * p1*y*py_pkz);
						Jacobian.ptr<double>(order + 1)[i * 7 + 11] = fy*(py_ptheta + k1*R*py_ptheta + k1*y*(2 * x*px_ptheta + 2 * y*py_ptheta) + k2*R*R*py_ptheta + 2 * k2*y*R*(2 * x*px_ptheta + 2 * y*py_ptheta) + 2 * p2*x*py_ptheta + 2 * p2*y*px_ptheta + 2 * p1*x*px_ptheta + 6 * p1*y*py_ptheta);
						Jacobian.ptr<double>(order + 1)[i * 7 + 12] = fy*(py_ptx + k1*R*py_ptx + k1*y*(2 * x*px_ptx + 2 * y*py_ptx) + k2*R*R*py_ptx + 2 * k2*y*R*(2 * x*px_ptx + 2 * y*py_ptx) + 2 * p2*x*py_ptx + 2 * p2*y*px_ptx + 2 * p1*x*px_ptx + 6 * p1*y*py_ptx);
						Jacobian.ptr<double>(order + 1)[i * 7 + 13] = fy*(py_pty + k1*R*py_pty + k1*y*(2 * x*px_pty + 2 * y*py_pty) + k2*R*R*py_pty + 2 * k2*y*R*(2 * x*px_pty + 2 * y*py_pty) + 2 * p2*x*py_pty + 2 * p2*y*px_pty + 2 * p1*x*px_pty + 6 * p1*y*py_pty);
						Jacobian.ptr<double>(order + 1)[i * 7 + 14] = fy*(py_ptz + k1*R*py_ptz + k1*y*(2 * x*px_ptz + 2 * y*py_ptz) + k2*R*R*py_ptz + 2 * k2*y*R*(2 * x*px_ptz + 2 * y*py_ptz) + 2 * p2*x*py_ptz + 2 * p2*y*px_ptz + 2 * p1*x*px_ptz + 6 * p1*y*py_ptz);

						f.ptr<double>(order)[0] = u - u_t;
						f.ptr<double>(order + 1)[0] = v - v_t;
						order = order + 2;
					}

				}
				F = 0.5*f.t()*f;
				L = F.clone();
				G = Jacobian.t()*f;
				JTJ = Jacobian.t()*Jacobian;
				find = (norm(G, NORM_INF) < e1);
				u = u*((0.333333 > (1 - (2 * rho - 1)*(2 * rho - 1)*(2 * rho - 1)) ? 0.333333 : (1 - (2 * rho - 1)*(2 * rho - 1)*(2 * rho - 1))));
				kp = 2;
			}
			else {
				u = u*kp;
				kp = kp * 2;
			}

		}
	}

	intrinsic_matrix.ptr<double>(0)[0] = parameters.ptr<double>(0)[0];
	intrinsic_matrix.ptr<double>(1)[1] = parameters.ptr<double>(1)[0];
	intrinsic_matrix.ptr<double>(0)[2] = parameters.ptr<double>(2)[0];
	intrinsic_matrix.ptr<double>(1)[2] = parameters.ptr<double>(3)[0];
	distcoffs.ptr<double>(0)[0] = parameters.ptr<double>(4)[0];
	distcoffs.ptr<double>(0)[1] = parameters.ptr<double>(5)[0];
	distcoffs.ptr<double>(0)[2] = parameters.ptr<double>(6)[0];
	distcoffs.ptr<double>(0)[3] = parameters.ptr<double>(7)[0];

	for (int i = 0; i < numberOfImage; ++i) {
		rvecs[i].ptr<double>(0)[0] = parameters.ptr<double>(i * 7 + 8)[0] * parameters.ptr<double>(i * 7 + 11)[0];
		rvecs[i].ptr<double>(1)[0] = parameters.ptr<double>(i * 7 + 9)[0] * parameters.ptr<double>(i * 7 + 11)[0];
		rvecs[i].ptr<double>(2)[0] = parameters.ptr<double>(i * 7 + 10)[0] * parameters.ptr<double>(i * 7 + 11)[0];
		tvecs[i].ptr<double>(0)[0] = parameters.ptr<double>(i * 7 + 12)[0];
		tvecs[i].ptr<double>(1)[0] = parameters.ptr<double>(i * 7 + 13)[0];
		tvecs[i].ptr<double>(2)[0] = parameters.ptr<double>(i * 7 + 14)[0];
	}

	return;
}


/*   stereo calibrate   */
void MyStereoCalibrate(Mat& intrinsic_matrixL, Mat& intrinsic_matrixR, vector<Mat>& rvecsL, vector<Mat>& rvecsR,
						vector<Mat>& tvecsL, vector<Mat>& tvecsR, Mat& essential_matrix, Mat& fundmental_matrix, Mat& R, Mat& T) {
	
	essential_matrix = Mat::zeros(Size(3, 3), CV_64F);
	fundmental_matrix = Mat::zeros(Size(3, 3), CV_64F);
	R = Mat::zeros(Size(3, 3), CV_64F);
	T = Mat::zeros(Size(1, 3), CV_64F);
	Mat Td = Mat::zeros(Size(1, 3), CV_64F);
	//Mat Rd = Mat::zeros(Size(3, 3), CV_64F);
	Mat S = Mat::zeros(Size(3, 3), CV_64F);
	for (int i = 0; i < rvecsL.size(); ++i) {
		Mat tempL, tempR, temp_Rot, temp_Rotd, temp_T, temp_Td;
		Rodrigues(rvecsL[i], tempL);
		Rodrigues(rvecsR[i], tempR);
		temp_Rot = tempR*tempL.t();
		temp_Rotd = tempL*tempR.t();
		temp_T = tvecsR[i] - temp_Rot*tvecsL[i];
		temp_Td = tvecsL[i] - temp_Rotd*tvecsR[i];

		R += temp_Rot;
		T += temp_T;
		Td += temp_Td;
		//Rd += temp_Rotd;
	}

	R = R / rvecsL.size();
	T = T / rvecsL.size();
	Td = Td / rvecsL.size();
	//Rd = Rd / rvecsL.size();

	S.ptr<double>(0)[1] = -Td.ptr<double>(2)[0];
	S.ptr<double>(0)[2] = Td.ptr<double>(1)[0];
	S.ptr<double>(1)[0] = Td.ptr<double>(2)[0];
	S.ptr<double>(1)[2] = -Td.ptr<double>(0)[0];
	S.ptr<double>(2)[0] = -Td.ptr<double>(1)[0];
	S.ptr<double>(2)[1] = Td.ptr<double>(0)[0];

	essential_matrix = S.t()*R.t();
	fundmental_matrix = (intrinsic_matrixL.t()).inv()*essential_matrix*intrinsic_matrixR.inv();
	double lamda = fundmental_matrix.ptr<double>(2)[2];
	fundmental_matrix = fundmental_matrix / lamda;

	return;
}


/*   Calculate the erros   */
double computeReprojectionErrors(vector<vector<Point3f>>& object_points, vector<vector<Point2f>>& image_points,
	vector<Mat>& rvecs, vector<Mat>& tvecs, Mat & intrinsic_matrix, Mat & distCoeffs,  vector<double> & reprojErrs) {
	vector<Point2f> imagePoints2;
	int totalPoints = 0;
	double totalErr = 0, err;
	reprojErrs.resize(image_points.size());

	for (int i = 0; i < object_points.size(); ++i) {
		projectPoints(Mat(object_points[i]), rvecs[i], tvecs[i], intrinsic_matrix, distCoeffs, imagePoints2);
		err = norm(Mat(image_points[i]), Mat(imagePoints2), NORM_L2);
		int n = object_points.size();
		reprojErrs[i] = sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}

	return sqrt(totalErr / totalPoints);
}


/*   Undistort the images   */
void UndistorImages(Mat& intrinsic_mitrax, Mat& distcoffs, vector<string>& image_list, Size img_size) {
	Mat img, NewImg, map1, map2;
	initUndistortRectifyMap(intrinsic_mitrax, distcoffs, Mat(), \
		getOptimalNewCameraMatrix(intrinsic_mitrax, distcoffs, img_size, 1, img_size, 0), img_size, CV_16SC2, map1, map2);
	for (int i = 0; i < (int)image_list.size(); i++)
	{
		img = imread(image_list[i], 1);
		if (img.empty())	continue;
		remap(img, NewImg, map1, map2, INTER_LINEAR);
		//imshow("Image View", NewImg);
		int test = 0; //for test;
	}
}


/*   Save the parameters   */
void saveCameraParameters(string OutputFile, Size board_size, Size image_size, float square_size,
											Mat& intrinsic_matrix, Mat& distCoeffs, double totalAvgErr) {

	FileStorage fs(OutputFile, FileStorage::WRITE);

	fs << "image_width" << image_size.width;
	fs << "image_height" << image_size.height;
	fs << "board_width" << board_size.width;
	fs << "board_height" << board_size.height;
	fs << "square_size" << square_size;

	fs << "camera_matrix" << intrinsic_matrix;
	fs << "distortion_coefficients" << distCoeffs;

	fs << "avg_reprojection_error" << totalAvgErr;

	fs.release();

	return;
}


/*   SGM algorithm   */
void MySGM(Mat& left, Mat& right, Mat& left_disp, Mat& right_disp) {
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
	int cn = left.channels();
	int sgbmWinSize = 5;
	sgbm->setPreFilterCap(63);
	sgbm->setBlockSize(sgbmWinSize);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(32);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(StereoSGBM::MODE_SGBM);

	Mat left_disp8, right_disp8;
	sgbm->compute(left, right, left_disp);
	sgbm->compute(right, left, right_disp);
	left_disp.convertTo(left_disp8, CV_8U);
	right_disp.convertTo(right_disp8, CV_8U);

	return;
}


int main(int argc, char* argv[])
{
	Size board_size, image_sizeL, image_sizeR;
	board_size.height = BoardHeight;
	board_size.width = BoardWith;
	string OutputFile = OutputFileName;

	double square_size = SquareSize;

	vector<vector<Point2f>> image_pointsL;
	vector<vector<Point2f>> image_pointsR;
	vector<vector<Point3f>> object_points(1);

	/*   Complete the images' path   */
	vector<string> image_listL;
	vector<string> image_listR;

	/*   
	for (int i = 1; i < 15; i++) {
		string pathL = ImagePathL;
		string pathR = ImagePathR;
		if (i < 10) {
			pathL = pathL + "left0" + char(i + 48) + ".jpg";
			pathR = pathR + "right0" + char(i + 48) + ".jpg";
		}
		else {
			pathL = pathL + "left1" + char(i % 10 + 48) + ".jpg";
			pathR = pathR + "right1" + char(i % 10 + 48) + ".jpg";
		}
		image_listL.push_back( pathL );
		image_listR.push_back( pathR );
	}
	*/

	for (int i = 1; i < 10; i++) {
		string pathL = ImagePathL;
		string pathR = ImagePathR;
		if (i < 10) {
			pathL = pathL + "left_" + char(i + 48) + ".jpg";
			pathR = pathR + "right_" + char(i + 48) + ".jpg";
		}
		else {
			pathL = pathL + "left_1" + char(i%10+ 48) + ".jpg";
			pathR = pathR + "right_1" + char(i%10+ 48) + ".jpg";
		}
		image_listL.push_back(pathL);
		image_listR.push_back(pathR);
	}
	cout << image_listL.size() << endl;

	FindImageCoordinates(image_listL, image_pointsL, board_size, image_sizeL); /*  To extract the corners  */
	FindImageCoordinates(image_listR, image_pointsR, board_size, image_sizeR);
	CalculateWorldCoordinates(board_size, square_size, object_points); /*   Calculate the coordinates of the corners in world coordinate system   */
	object_points.resize(image_pointsL.size(), object_points[0]);

	Mat intrinsic_matrixL = Mat::eye(3, 3, CV_64F);  /*   Intrinsic matrix   */
	Mat distCoeffsL = Mat::zeros(8, 1, CV_64F);      /*   Distortion coefficient matrix   */
	vector<Mat> rvecsL, tvecsL;                       /*   extrinsic matrix   */

	Mat intrinsic_matrixR = Mat::eye(3, 3, CV_64F);
	Mat distCoeffsR = Mat::zeros(8, 1, CV_64F);
	vector<Mat> rvecsR, tvecsR; 

	/*   Calibration by Opencv function   */
	calibrateCamera(object_points, image_pointsL, image_sizeL, intrinsic_matrixL,
		distCoeffsL, rvecsL, tvecsL, CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);
	calibrateCamera(object_points, image_pointsR, image_sizeR, intrinsic_matrixR,
		distCoeffsR, rvecsR, tvecsR, CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);   


	/*   Calibration by Self-implementing function   */
	Calibrate(object_points, image_pointsL, board_size, intrinsic_matrixL, distCoeffsL, rvecsL, tvecsL);   
	Calibrate(object_points, image_pointsR, board_size, intrinsic_matrixR, distCoeffsR, rvecsR, tvecsR);
	
	/* 
	intrinsic_matrixL.ptr<double>(0)[0] = 362.2;
	intrinsic_matrixL.ptr<double>(1)[1] = 363.5;
	intrinsic_matrixL.ptr<double>(0)[2] = 406.58;
	intrinsic_matrixL.ptr<double>(1)[2] = 234.35;
	distCoeffsL.ptr<double>(0)[0] = -0.250348;
	distCoeffsL.ptr<double>(0)[1] = 0.050579;
	distCoeffsL.ptr<double>(0)[2] = -0.000705;
	distCoeffsL.ptr<double>(0)[3] = -0.008526;

	intrinsic_matrixR.ptr<double>(0)[0] = 365.14;
	intrinsic_matrixR.ptr<double>(1)[1] = 365.13;
	intrinsic_matrixR.ptr<double>(0)[2] = 389.32;
	intrinsic_matrixR.ptr<double>(1)[2] = 234.95;
	distCoeffsR.ptr<double>(0)[0] = -0.303773;
	distCoeffsR.ptr<double>(0)[1] = 0.079930;
	distCoeffsR.ptr<double>(0)[2] = 0.000052;
	distCoeffsR.ptr<double>(0)[3] = -0.000673;
	*/

	Mat essential_matrix , fundmental_matrix, R, T;   /*   Essential Matrix, fundmental Matrix, Rotation Matrix, Displacement vector   */
	//MyStereoCalibrate(intrinsic_matrixL,intrinsic_matrixR, rvecsL, rvecsR,tvecsL, tvecsR, 
	//														essential_matrix, fundmental_matrix, R, T);

	   
	vector<double> perViewError;   
	double rms = stereoCalibrate(object_points, image_pointsL, image_pointsR, intrinsic_matrixL, distCoeffsL, \
							intrinsic_matrixR, distCoeffsR, image_sizeL, R, T, essential_matrix, fundmental_matrix, \
							CALIB_FIX_INTRINSIC, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));   

	
	 
	/*   Computes rectification transforms for each head of a calibrated stereo camera. */
    Mat rmap[2][2];
	Rect  Roi1, Roi2;
	Mat Rrl, Rrr, new_intrinsic_matrixL, new_intrinsic_matrixR, Q;
	stereoRectify(intrinsic_matrixL, distCoeffsL, intrinsic_matrixR, distCoeffsR, image_sizeL, R, T,
		Rrl, Rrr, new_intrinsic_matrixL, new_intrinsic_matrixR, Q, CALIB_ZERO_DISPARITY, -1, image_sizeL, &Roi1, &Roi2);
	initUndistortRectifyMap(intrinsic_matrixL, distCoeffsL, Rrl, new_intrinsic_matrixL, image_sizeL, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(intrinsic_matrixR, distCoeffsR, Rrr, new_intrinsic_matrixR, image_sizeR, CV_16SC2, rmap[1][0], rmap[1][1]);
	
	/*  
	for (int i = 0; i < image_listL.size(); ++i) {
		Mat img1 = imread(image_listL[i],-1), img1r;
		Mat img2 = imread(image_listR[i],-1), img2r;
		
		Mat img(image_sizeL.height, image_sizeL.width * 2, CV_8UC1);

		remap(img1, img1r, rmap[0][0], rmap[0][1], INTER_LINEAR);
		remap(img2, img2r, rmap[1][0], rmap[1][1], INTER_LINEAR);

		Mat imgPart1 = img(Rect(0, 0, image_sizeL.width, image_sizeL.height));
		Mat imgPart2 = img(Rect(image_sizeL.width, 0, image_sizeL.width, image_sizeL.height));
		resize(img1r, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
		resize(img2r, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);

		//画横线
		for (int i = 0; i < img.rows; i += 32)
			line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);

		//显示行对准的图形
		Mat smallImg;//由于我的分辨率1:1显示太大，所以缩小显示
		//resize(img, smallImg, Size(), 0.5, 0.5, CV_INTER_AREA);
		//imshow("rectified", img);

	}   */


	vector<string> test_listL, test_listR;
	for (int i = 0; i < 9; i++) {
		string pathL = "C:\\Users\\lenovo\\Desktop\\stereo\\test_u\\";
		string pathR = "C:\\Users\\lenovo\\Desktop\\stereo\\test_u\\";
		if (i < 10) {
			pathL = pathL + "left_" + char(i + 48) + ".jpg";
			pathR = pathR + "right_" + char(i + 48) + ".jpg";
		}
		else {
			pathL = pathL + "left_1" + char(i % 10 + 48) + ".jpg";
			pathR = pathR + "right_1" + char(i % 10 + 48) + ".jpg";
		}
		test_listL.push_back(pathL);
		test_listR.push_back(pathR);
	}

	for (int i = 0; i < test_listL.size(); ++i) {
		Mat img1 = imread(test_listL[i], -1), img1r;
		Mat img2 = imread(test_listR[i], -1), img2r;

		Mat img(image_sizeL.height, image_sizeL.width * 2, CV_8UC1);

		remap(img1, img1r, rmap[0][0], rmap[0][1], INTER_LINEAR);
		remap(img2, img2r, rmap[1][0], rmap[1][1], INTER_LINEAR);

		Mat imgPart1 = img(Rect(0, 0, image_sizeL.width, image_sizeL.height));
		Mat imgPart2 = img(Rect(image_sizeL.width, 0, image_sizeL.width, image_sizeL.height));
		resize(img1r, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
		resize(img2r, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);

		//画横线
		for (int i = 0; i < img.rows; i += 32)
			line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);

		string pathL = "";
		string pathR = "";
		string pathLR = "";
		pathL = pathL + "C:\\Users\\lenovo\\Desktop\\ture_test\\left_" + char(i + 48) + ".jpg";
		pathR = pathR + "C:\\Users\\lenovo\\Desktop\\ture_test\\right_" + char(i + 48) + ".jpg";
		pathLR = pathLR + "C:\\Users\\lenovo\\Desktop\\ture_test\\epipolar_str" + char(i + 48) + ".jpg";
		imwrite(pathL, img1r);
		imwrite(pathR, img2r);
		imwrite(pathLR, img);


		//显示行对准的图形
		Mat smallImg;//由于我的分辨率1:1显示太大，所以缩小显示
					 //resize(img, smallImg, Size(), 0.5, 0.5, CV_INTER_AREA);
					 //imshow("rectified", img);

	}


	/*   Calculate the errors   */
	vector<double> reprojErrsL, reprojErrsR;
	double AvgErrL = 0, AvgErrR = 0;
	AvgErrL = computeReprojectionErrors(object_points, image_pointsL, rvecsL, tvecsL, intrinsic_matrixL, distCoeffsL, reprojErrsL);
	AvgErrR = computeReprojectionErrors(object_points, image_pointsR, rvecsR, tvecsR, intrinsic_matrixR, distCoeffsR, reprojErrsR);

	/*   Undistort the images   */
	UndistorImages(intrinsic_matrixL, distCoeffsL, image_listL, image_sizeL);
	UndistorImages(intrinsic_matrixR, distCoeffsR, image_listR, image_sizeR);

	/*   Save the parameters   */
	saveCameraParameters(OutputFile, board_size, image_sizeL, square_size, intrinsic_matrixL, distCoeffsL, AvgErrL);

	system("pause");
	return 0;
}
