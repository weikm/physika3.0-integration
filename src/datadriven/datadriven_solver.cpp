#include "datadriven_solver.h"
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>
// #include <gl_particle_render/glWindow/glWidGet.h>
// #include <gl_particle_render/renderer/cuParticleRenderer.h>

namespace Physika {

// 初始化Python解释器和导入NumPy API
int DataDrivenSolver::initializePython() {
    Py_Initialize();
    if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return -1; // 失败时返回-1
    }
    return 0; // 成功时返回0
}

// 清理Python解释器
void DataDrivenSolver::finalizePython() {
    Py_Finalize();
}

DataDrivenSolver::DataDrivenSolver(const std::string& module, const std::string& className,
                                   const std::string& weightsPath, const std::string& sceneName,
                                   const std::string& outputDir, const int numIterations,  bool writePly) {
    initializePython();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("sys.path.append('../')");

    this->numIterations = numIterations;
    this->writePly = writePly;
    PyObject* pName = PyUnicode_FromString(module.c_str());
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to load module");
    }

    PyObject* pClass = PyObject_GetAttrString(pModule, className.c_str());
    Py_DECREF(pModule);

    if (!pClass || !PyCallable_Check(pClass)) {
        PyErr_Print();
        throw std::runtime_error("Failed to load Worker class");
    }

    PyObject* pArgs = PyTuple_Pack(3, PyUnicode_FromString(weightsPath.c_str()),
                                        PyUnicode_FromString(sceneName.c_str()),
                                        PyUnicode_FromString(outputDir.c_str()));
    this->pInstance = PyObject_CallObject(pClass, pArgs);
    Py_DECREF(pArgs);

    if (!this->pInstance) {
        PyErr_Print();
        throw std::runtime_error("Failed to create Worker instance");
    }
}

DataDrivenSolver::~DataDrivenSolver() {
    if (this->pInstance) {
        Py_DECREF(this->pInstance);
    }
    finalizePython();
}

void DataDrivenSolver::steps() {
    // 实现调用Python中的steps方法
    PyObject_CallMethod(this->pInstance, "steps", "ii", this->numIterations, this->writePly);
    if (PyErr_Occurred()) {
        PyErr_Print();
    }
}

PyObject* DataDrivenSolver::getFluids() {
    // 实现调用Python中的get_fluids方法
    PyObject* result = PyObject_CallMethod(this->pInstance, "get_fluids", NULL);
    if (!result) {
        PyErr_Print();
    }
    return result;
}

PyObject* DataDrivenSolver::stepOne(PyObject* pPos, PyObject* pVel, int step) {

    // 实现调用Python中的step_one方法
    PyObject* newResult = PyObject_CallMethod(this->pInstance, "step_one", "OOii", pPos, pVel, step, this->writePly);
    if (!newResult) {
        PyErr_Print();
    } 
    return newResult;
}

void DataDrivenSolver::checkPyObjType(PyObject* object) {
    PyObject* pTypeName = PyObject_Str(PyObject_Type(object));
    PyObject* pTypeCString = PyUnicode_AsEncodedString(pTypeName, "utf-8", "Error ~");
    const char* typeName = PyBytes_AS_STRING(pTypeCString);
    std::cout << "Type of PyObject: " << typeName << std::endl;
    Py_DECREF(pTypeName);
    Py_DECREF(pTypeCString);
}

void DataDrivenSolver::printNpyShape(PyObject* object, const char* objName) {
    if (!PyArray_Check(object)) {
        std::cerr << objName << " is not a numpy array." << std::endl;
        return;
    }
    int ndim = PyArray_NDIM((PyArrayObject*)object);
    npy_intp* shape = PyArray_DIMS((PyArrayObject*)object);

    std::cout << "Shape of " << objName << ": (";
    for (int i = 0; i < ndim; ++i) {
        std::cout << shape[i];
        if (i < ndim - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}

void DataDrivenSolver::evaluate(const std::string& datasetDir) {
    std::cout << "evaluating the whole sequence errs" << std::endl;
    PyObject_CallMethod(this->pInstance, "evaluate_whole_sequence", "s", datasetDir.c_str());
}

}

