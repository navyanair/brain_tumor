
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = brainTumorGUI
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    cvmatandqimage.cpp

HEADERS  += mainwindow.h \
    cvmatandqimage.h

FORMS    += mainwindow.ui


INCLUDEPATH += "F:\opencv\install\include"
debug
{
LIBS += -LF:\opencv\lib    \
-llibopencv_world320d
}

release
{
LIBS += -LF:\opencv\lib    \
-llibopencv_world320
}
