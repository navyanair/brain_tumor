
QT       += core

QT       -= gui

TARGET = braintumour_training
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    databasereader.cpp


INCLUDEPATH += "C:\software\openCV\install\include"
debug
{
LIBS += -LC:\software\openCV\lib\
-llibopencv_world320d
}

release
{
LIBS += -LC:\software\openCV\lib\
-llibopencv_world320
}
HEADERS += \
    databasereader.h \
    tinydir.h
