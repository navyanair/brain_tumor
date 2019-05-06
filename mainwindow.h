#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "QFileDialog"
#include "QPixmap"
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include "cvmatandqimage.h"
#include <iostream>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_SelectImage_clicked();

    void on_Scan_clicked();

    void on_tumorarea_clicked();

private:
    Ui::MainWindow *ui;
    void glcm(cv::Mat &img,float &energy,float &contrast,float &homogenity,float &IDM,float &entropy,float &mean1);
    QString mFilename;
    QImage mFinalPix;
};

#endif // MAINWINDOW_H
