#ifndef DATADRIVEN_SOLVER_H
#define DATADRIVEN_SOLVER_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// #include </usr/include/python3.8/Python.h>
// #include </usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy/arrayobject.h>

#include <Python.h>
#include <arrayobject.h>
#include <string>
namespace Physika{
class DataDrivenSolver {
public:

    int numIterations = 200; // 默认单次仿真200时间步
    int writePly = 0; // 默认不保存npz
    DataDrivenSolver(const std::string& module, const std::string& className,
                     const std::string& weightsPath, const std::string& sceneName,
                     const std::string& outputDir, const int numIterations, bool writePly=false);
    ~DataDrivenSolver();

    void steps();
    PyObject* getFluids();
    PyObject* stepOne(PyObject* pPos, PyObject* pVel, int step);
    void evaluate(const std::string& datasetDir);
    static void checkPyObjType(PyObject* object);
    static void printNpyShape(PyObject* object, const char* objName);

private:
    PyObject* pInstance;
    int initializePython();
    void finalizePython();
};

#endif // DATADRIVENSOLVER_H
}