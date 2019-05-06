#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace QtOcv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->Scan->setEnabled(false);
    ui->tumorarea->setEnabled(false);
    ui->image1->setScaledContents(true);
    ui->image2->setScaledContents(true);
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::glcm(Mat &img,float &energy,float &contrast,float &homogenity,float &IDM,float &entropy,float &mean1)
{

    int row=img.rows,col=img.cols;
    cv::Mat gl=cv::Mat::zeros(256,256,CV_32FC1);

    //creating glcm matrix with 256 levels,radius=1 and in the horizontal direction
    for(int i=0;i<row;i++)
        for(int j=0;j<col-1;j++)
            gl.at<float>(img.at<uchar>(i,j),img.at<uchar>(i,j+1))=gl.at<float>(img.at<uchar>(i,j),img.at<uchar>(i,j+1))+1;

    // normalizing glcm matrix for parameter determination
    gl=gl+gl.t();
    gl=gl/sum(gl)[0];

    float a[256][256];
    for(int i=0;i<256;i++)
        for(int j=0;j<256;j++)
        {
            energy=gl.at<float>(i,j)*gl.at<float>(i,j);
            a[i][j]=energy;
            //        cout<<energy;
            //finding parameters
            contrast=contrast+(i-j)*(i-j)*gl.at<float>(i,j);
            homogenity=homogenity+gl.at<float>(i,j)/(1+abs(i-j));
            if(i!=j)
                IDM=IDM+gl.at<float>(i,j)/((i-j)*(i-j));                      //Taking k=2;
            if(gl.at<float>(i,j)!=0)
                entropy=entropy-gl.at<float>(i,j)*log10(gl.at<float>(i,j));
            mean1=mean1+0.5*(i*gl.at<float>(i,j)+j*gl.at<float>(i,j));
        }
}

void MainWindow::on_SelectImage_clicked()
{
    mFilename = QFileDialog::getOpenFileName();

    QPixmap pix(mFilename);
    ui->image1->setPixmap(pix);
    ui->Scan->setEnabled(true);
    ui->tumorarea->setEnabled(false);
    ui->image2->clear();
    ui->Result->setText(" ");

}

void MainWindow::on_Scan_clicked()
{
    Ptr<RTrees> rt = StatModel::load<RTrees>("tumorDetection.xml");
    vector< vector <Point> > img;
    Mat src,grey,bw,mor,dil,t2Brain,final,testimage;
    vector<Rect> diseaseArea;
    vector<Point> temp;
    int index,min,max=0;
    Rect bbox;
    src=imread(mFilename.toStdString());
    //imshow("image",src);
    cvtColor(src,grey,CV_BGR2GRAY);

    threshold(grey,bw,0,255,CV_THRESH_OTSU);

    mor=getStructuringElement(MORPH_RECT,Size(9,9),Point(4,4));
    dilate(bw,dil,mor);
    // imshow("dilated",dil);
    findContours(dil,img,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

   // findContours(dil,img,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    Scalar color= Scalar(0,255,0);

    for(int i=0;i<img.size();i++)
    {
        temp=img[i];
        min=temp.size();
        if(min>max)
        {
            max=min;
            index=i;
        }
    }
    //drawContours(src,img,index,color,2);
    //imshow("contour",src);

    bbox=boundingRect(img[index]);
    t2Brain=grey(bbox);
    final=src(bbox);

    cv::resize(t2Brain,grey,Size(300,400));
    //imshow("croped Brain area",grey);

    int ROW=grey.rows;
    int COL=grey.cols;
    cout<<"ros="<<ROW<<endl;
    int width=60,height=80;
    int stride=40;
    Mat cropedImage;

    for(int i=0;i<ROW-height;i+=stride)
    {
        for(int j=0;j<COL-width;j+=stride)
        {

            float energy=0,contrast=0,homogenity=0,IDM=0,entropy=0,mean1=0;
            float mean=0,variance=0,standeredDeviation=0,sumOfPixels=0;
            vector<float> features;

            cropedImage=grey(Rect(j,i,width,height));
            imshow("cropedimages",cropedImage);
            for(int row=0;row<height;row++)
            {
                for(int col=0;col<width;col++)
                {
                    sumOfPixels += cropedImage.at<uchar>(row,col);

                }
            }
            mean=sumOfPixels/(height*width);

            for(int row=0;row<height;row++)
            {
                for(int col=0;col<width;col++)
                {
                    int intensity=cropedImage.at<uchar>(row,col);
                    variance += ((intensity-mean)*(intensity-mean));


                }
            }
            cout<<variance<<endl;
            variance /= (height*width);

            standeredDeviation = sqrt(variance);

            glcm(cropedImage,energy,contrast,homogenity,IDM,entropy,mean1);

            cout<<"mean="<<mean1<<endl;
            features.push_back(mean1);

            cout<<"standeredDeviation="<<standeredDeviation<<endl;
            features.push_back(standeredDeviation);

            cout<<"variance="<<variance<<endl;
            features.push_back(variance);

            cout<<"energy="<<energy<<endl;
            features.push_back(energy);

            cout<<"contrast="<<contrast<<endl;
            features.push_back(contrast);

            cout<<"homogenity="<<homogenity<<endl;
            features.push_back(homogenity);

            cout<<"IDM="<<IDM<<endl;
            features.push_back(IDM);

            cout<<"entropy="<<entropy<<endl;
            features.push_back(entropy);
            Mat rowdata=Mat(features);
            Mat vec = rowdata.reshape(0,1);
            vec.convertTo(testimage,CV_32F);
            // int predicted = svm->predict(testimage);
            float predicted = rt->predict(testimage);
            if((int)predicted==1)
            {   diseaseArea.push_back(Rect(j,i,width,height));
                //rectangle(final,Rect(j,i,width,height),Scalar(255,0,0),3);
            }
        }
    }
    if(diseaseArea.size()>3)
    {
        for(int i=0;i<diseaseArea.size();i++)
        {
            rectangle(final,diseaseArea[i],Scalar(255,0,0),3);
        }
        imshow("disease affected",final);
        ui->Result->setText("Tumor Detected");
        mFinalPix=mat2Image(final);

        ui->tumorarea->setEnabled(true);

    }
    else
    {
        //  imshow("Normal",final);
        ui->Result->setText("No Disease Effected");
    }

}

void MainWindow::on_tumorarea_clicked()
{
    QPixmap tempPixmap=tempPixmap.fromImage(mFinalPix);
    ui->image2->setPixmap(tempPixmap);

}
